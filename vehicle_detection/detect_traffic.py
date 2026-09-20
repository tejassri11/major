"""
detect_traffic.py
==================
Runs YOLOv8 vehicle detection on traffic.mp4.
Counts queued (stationary) vehicles per lane approach.
Outputs the same log-scaled values used by the PPO observation vector.

Controls:
    Q  — quit
    P  — pause / unpause
    R  — re-draw ROIs interactively (left-click = add point, right-click = finish lane)
    S  — save current frame with overlay to output/
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import os
import time

# ── Config ────────────────────────────────────────────────────────────────────
# Resolve the video and model relative to this script's folder so the detector
# finds its assets no matter which directory it is launched from.
_HERE               = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH          = os.path.join(_HERE, "traffic.mp4")
YOLO_MODEL          = os.path.join(_HERE, "yolov8s.pt")   # small model — better accuracy than nano
CONF                = 0.25               # lower threshold catches more vehicles
VEHICLE_CLASSES     = [2, 3, 5, 7]        # car, motorcycle, bus, truck

# Indian IRC Passenger Car Unit (PCU) values for detected classes
PCU_MAP = {
    3: 0.5,   # motorcycle / 2-wheeler
    2: 1.0,   # car / auto-rickshaw (classified as car in basic COCO)
    5: 3.0,   # bus
    7: 3.0,   # truck
}

STATIONARY_PX       = 10                  # max pixel drift to be classed as queued
STATIONARY_FRAMES   = 8                   # consecutive still frames required
PROCESS_EVERY_N     = 1                   # run YOLO every frame for full coverage

# Match sumo_env.py exactly so values slot into the PPO obs vector unchanged
OBS_MAX_QUEUE    = 15.0
LOG_OBS_MAX      = float(np.log1p(OBS_MAX_QUEUE))
OBS_MAX_WAIT     = 300.0
LOG_OBS_MAX_WAIT = float(np.log1p(OBS_MAX_WAIT))

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── ROI definitions ───────────────────────────────────────────────────────────
# Pixel coordinates on the 2160×3840 frame.
# These cover the queue zone just before the stop line at the bottom
# of the intersection.  Adjust with R key if needed.
ROIS = {
    "lane_L1": [(270, 2050), (570, 2050), (510, 2500), (130, 2500)],   # far-left lane
    "lane_L2": [(570, 2050), (850, 2050), (820, 2500), (510, 2500)],   # centre-left lane
    "lane_R1": [(850, 2050), (1130, 2050), (1140, 2500), (820, 2500)], # centre-right lane
    "lane_R2": [(1130, 2050), (1430, 2050), (1490, 2500), (1140, 2500)], # far-right lane
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def point_in_roi(cx, cy, polygon):
    pts = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0


def log_scale_queue(count):
    return min(float(np.log1p(count) / LOG_OBS_MAX), 1.0)


def draw_overlay(frame, rois, queue_counts, boxes, track_stationary):
    out = frame.copy()
    frame_h, frame_w = out.shape[:2]

    # ROI polygons — label outside the polygon (above it) to keep road clear
    for lane_id, poly in rois.items():
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 210, 255), 3)
        cx = int(np.mean([p[0] for p in poly]))
        top_y = min(p[1] for p in poly) - 20   # label just above the ROI box
        count = queue_counts.get(lane_id, 0)
        scaled = log_scale_queue(count)
        label = f"{lane_id}  q={count}  obs={scaled:.2f}"
        # semi-transparent background for readability
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(out, (cx - lw // 2 - 8, top_y - lh - 8),
                      (cx + lw // 2 + 8, top_y + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (cx - lw // 2, top_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 210, 255), 2)

    # Bounding boxes — red = queued, green = moving
    if boxes is not None and len(boxes):
        ids = boxes.id.int() if boxes.id is not None else [None] * len(boxes.xyxy)
        for box, tid in zip(boxes.xyxy, ids):
            x1, y1, x2, y2 = map(int, box)
            stationary = track_stationary.get(int(tid), False) if tid is not None else False
            colour = (0, 50, 255) if stationary else (50, 220, 50)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 3)

    # Summary panel placed in the black letterbox at the very top of the frame
    # (above the road content which starts ~y=1050 in the 3840px frame)
    panel_y0  = 20
    panel_x0  = 20
    row_h     = 60
    panel_h   = 80 + row_h * len(rois)
    panel_w   = 680
    cv2.rectangle(out, (panel_x0, panel_y0),
                  (panel_x0 + panel_w, panel_y0 + panel_h), (20, 20, 20), -1)
    cv2.rectangle(out, (panel_x0, panel_y0),
                  (panel_x0 + panel_w, panel_y0 + panel_h), (0, 210, 255), 2)
    cv2.putText(out, "QUEUE LENGTHS", (panel_x0 + 15, panel_y0 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

    y = panel_y0 + 90
    for lane_id, count in queue_counts.items():
        bar_max = panel_w - 40
        bar_w   = min(int(count / OBS_MAX_QUEUE * bar_max), bar_max)
        # background bar
        cv2.rectangle(out, (panel_x0 + 20, y),
                      (panel_x0 + 20 + bar_max, y + 38), (50, 50, 50), -1)
        # filled bar
        if bar_w > 0:
            cv2.rectangle(out, (panel_x0 + 20, y),
                          (panel_x0 + 20 + bar_w, y + 38), (0, 180, 255), -1)
        cv2.putText(out, f"{lane_id}: {count}", (panel_x0 + 25, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        y += row_h

    return out


def interactive_rois(frame, existing_rois):
    """Click to define ROI polygons. Right-click to finish each lane."""
    clone  = frame.copy()
    points = []
    rois   = dict(existing_rois)
    idx    = [len(rois)]

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 8, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(clone, points[-2], points[-1], (0, 255, 0), 3)
            cv2.imshow("Define ROIs — R-click to finish lane, Q to done", clone)

        elif event == cv2.EVENT_RBUTTONDOWN and len(points) >= 3:
            lane_id = f"lane_{idx[0]}"
            rois[lane_id] = list(points)
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(clone, [pts], True, (0, 210, 255), 3)
            cx = int(np.mean([p[0] for p in points]))
            cy = int(np.mean([p[1] for p in points]))
            cv2.putText(clone, lane_id, (cx - 60, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 210, 255), 3)
            cv2.imshow("Define ROIs — R-click to finish lane, Q to done", clone)
            print(f'  "{lane_id}": {list(points)},')
            points.clear()
            idx[0] += 1

    cv2.namedWindow("Define ROIs — R-click to finish lane, Q to done")
    cv2.setMouseCallback("Define ROIs — R-click to finish lane, Q to done", mouse)
    cv2.imshow("Define ROIs — R-click to finish lane, Q to done", clone)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("Define ROIs — R-click to finish lane, Q to done")
    return rois


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[DETECTOR] Loading YOLO model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {VIDEO_PATH}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[DETECTOR] {w}×{h} @ {fps:.0f}fps  |  {total} frames  |  {total/fps:.0f}s")
    print("[DETECTOR] Controls: Q=quit  P=pause  R=re-draw ROIs  S=save frame")

    rois            = dict(ROIS)
    history         = defaultdict(lambda: deque(maxlen=STATIONARY_FRAMES))
    track_stationary = {}
    queue_counts    = {lid: 0 for lid in rois}
    last_boxes      = None
    frame_idx       = 0
    paused          = False

    # Display scale so the 4K frame fits on screen
    scale = min(1.0, 1080 / h, 1920 / w)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            frame_idx += 1

        # Run detection every N frames
        if not paused and frame_idx % PROCESS_EVERY_N == 0:
            results      = model.track(frame, classes=VEHICLE_CLASSES,
                                       conf=CONF, persist=True, verbose=False)
            last_boxes   = results[0].boxes
            new_counts   = {lid: 0 for lid in rois}
            track_stationary.clear()

            if last_boxes is not None and last_boxes.id is not None:
                classes = last_boxes.cls.int() if last_boxes.cls is not None else [2] * len(last_boxes)
                for box, tid, cls_id in zip(last_boxes.xyxy, last_boxes.id.int(), classes):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    tid = int(tid)
                    cls_id = int(cls_id)
                    pcu_val = PCU_MAP.get(cls_id, 1.0)

                    hist = history[tid]
                    hist.append((cx, cy))

                    still = False
                    if len(hist) >= STATIONARY_FRAMES:
                        xs = [p[0] for p in hist]
                        ys = [p[1] for p in hist]
                        still = max(max(xs)-min(xs), max(ys)-min(ys)) <= STATIONARY_PX

                    track_stationary[tid] = still
                    if still:
                        for lane_id, poly in rois.items():
                            if point_in_roi(cx, cy, poly):
                                new_counts[lane_id] += pcu_val

            queue_counts = new_counts

        # Build display frame
        display = draw_overlay(frame, rois, queue_counts, last_boxes, track_stationary)

        # Log-scaled obs vector (what PPO would receive)
        obs_vec = [log_scale_queue(queue_counts.get(lid, 0)) for lid in sorted(rois)]
        status  = "PAUSED" if paused else f"frame {frame_idx}/{total}"
        print(f"\r[{status}]  queues={dict(queue_counts)}  obs={[f'{v:.2f}' for v in obs_vec]}   ", end="")

        # Scale down for display
        if scale < 1.0:
            dw = int(w * scale)
            dh = int(h * scale)
            display = cv2.resize(display, (dw, dh))

        cv2.imshow("Vehicle Detector — Queue Extraction", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            fname = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(fname, display)
            print(f"\n[SAVED] {fname}")
        elif key == ord('r'):
            paused = True
            print("\n[ROI] Draw new ROIs — left-click points, right-click to finish lane, Q when done")
            print("      Paste the printed coordinates into ROIS dict to make them permanent.")
            rois = interactive_rois(frame, rois)
            queue_counts = {lid: 0 for lid in rois}

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DETECTOR] Done.")


if __name__ == "__main__":
    main()
