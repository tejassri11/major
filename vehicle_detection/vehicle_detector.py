"""
vehicle_detector.py
====================
Extracts per-lane queue lengths from a live video feed using YOLOv8.
Produces the same halting-vehicle counts that SUMO's TraCI returns,
so the values can feed directly into the PPO/MAPPO observation vector.

Concept flow:
    Camera / video file
        → YOLOv8 vehicle detection  (CPU-safe, no GPU required)
        → ROI (region-of-interest) per lane
        → stationary-vehicle filter (pixel-movement threshold)
        → queue count + wait time per lane
        → log-scaled MAPPO observation → written to CSV

Requirements:
    pip install ultralytics opencv-python numpy

Usage (headless / no GUI):
    detector = VehicleDetector(source="video.mp4", device="cpu")
    detector.define_roi("lane_N1", [(x1,y1), (x2,y2), (x3,y3), (x4,y4)])
    detector.run_headless(
        output_csv="obs_output.csv",
        lane_order=["lane_N1", "lane_N2", ...],   # must match MAPPO lane ordering
        tl_id="6073919354",
    )

Usage (GUI, for calibration):
    detector = VehicleDetector(source=0)
    detector.define_roi("lane_north", [(x1,y1), ...])
    detector.run()   # press Q to quit, R on first frame to draw ROIs
"""

import csv
import time
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# COCO class IDs for vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# How many pixels a vehicle's centre can move and still be considered stationary
STATIONARY_THRESHOLD_PX = 8

# How many consecutive frames a vehicle must be still to count as queued
STATIONARY_MIN_FRAMES = 5

# Match sumo_env.py constants so values feed directly into the PPO obs
OBS_MAX_QUEUE  = 15.0
LOG_OBS_MAX    = float(np.log1p(OBS_MAX_QUEUE))
OBS_MAX_WAIT   = 300.0
LOG_OBS_MAX_WAIT = float(np.log1p(OBS_MAX_WAIT))


class VehicleDetector:
    def __init__(self, source=0, model_path="yolov8n.pt", conf=0.4, device="cpu",
                 infer_width=1280):
        """
        source      : camera index (0) or path to a video file
        model_path  : YOLOv8 weights; 'yolov8n.pt' auto-downloads on first run
        conf        : detection confidence threshold
        device      : 'cpu' (safe on weak GPUs) or 'cuda'
        infer_width : resize frames to this width before YOLO inference.
                      ROI coordinates are always in original-frame pixels —
                      the class handles scaling internally.
                      Set to None to disable resizing.
        """
        self.source      = source
        self.conf        = conf
        self.device      = device
        self.infer_width = infer_width
        self.model       = YOLO(model_path)
        self._scale      = 1.0   # updated each frame

        # lane_id -> list of (x, y) polygon vertices defining the ROI
        self._rois: dict[str, list[tuple[int, int]]] = {}

        # track_id -> deque of (cx, cy) centres across recent frames
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=STATIONARY_MIN_FRAMES))

        # track_id -> consecutive frames spent stationary
        self._stationary_frames: dict[int, int] = defaultdict(int)

        # track_id -> cumulative wait seconds (incremented each frame the vehicle is stationary)
        self._wait_seconds: dict[int, float] = defaultdict(float)

        # Results from last processed frame
        self.queue_counts: dict[str, int] = {}
        self.wait_times:   dict[str, float] = {}   # seconds per lane (sum across halting vehicles)
        self.vehicle_boxes: list = []

    # ──────────────────────────────────────────────────────────────
    # ROI management
    # ──────────────────────────────────────────────────────────────
    def define_roi(self, lane_id: str, polygon: list[tuple[int, int]]):
        """
        Register a lane with its detection zone.
        polygon is a list of (x, y) pixel coordinates forming a closed polygon.
        Tip: use define_rois_interactive() below to draw them with your mouse.

        Example for a northbound approach lane:
            detector.define_roi("lane_N1", [(120,400), (200,400), (220,600), (100,600)])
        """
        self._rois[lane_id] = polygon

    def define_rois_interactive(self, frame: np.ndarray, display_width: int = 900):
        """
        Opens an interactive window so you can click to define ROI polygons.
        Left-click to add points, right-click to finish a polygon, Q to quit.
        Prints the resulting define_roi() calls you can paste into your code.

        Large frames (e.g. 4K phone video) are scaled to display_width for the
        window; clicks are automatically mapped back to original pixel coordinates.
        """
        # Scale for display so window fits on screen
        disp_scale = min(1.0, display_width / frame.shape[1])
        disp_w = int(frame.shape[1] * disp_scale)
        disp_h = int(frame.shape[0] * disp_scale)
        clone    = cv2.resize(frame, (disp_w, disp_h))
        points_disp = []   # in display coords (for drawing)
        points_orig = []   # in original coords (for ROI)
        lane_idx = [0]
        results  = {}

        def _mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points_disp.append((x, y))
                # Map back to original frame coordinates
                ox = int(x / disp_scale)
                oy = int(y / disp_scale)
                points_orig.append((ox, oy))
                cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                if len(points_disp) > 1:
                    cv2.line(clone, points_disp[-2], points_disp[-1], (0, 255, 0), 2)
                cv2.imshow("Define ROIs  [L-click=point  R-click=finish lane  Q=done]", clone)
            elif event == cv2.EVENT_RBUTTONDOWN and len(points_orig) >= 3:
                lane_id = f"lane_{lane_idx[0]}"
                results[lane_id] = list(points_orig)
                cv2.polylines(clone, [np.array(points_disp)], True, (0, 200, 255), 2)
                # Label in display
                cx = int(np.mean([p[0] for p in points_disp]))
                cy = int(np.mean([p[1] for p in points_disp]))
                cv2.putText(clone, lane_id, (cx - 20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.imshow("Define ROIs  [L-click=point  R-click=finish lane  Q=done]", clone)
                print(f'detector.define_roi("{lane_id}", {list(points_orig)})')
                points_disp.clear()
                points_orig.clear()
                lane_idx[0] += 1

        win = "Define ROIs  [L-click=point  R-click=finish lane  Q=done]"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, disp_w, min(disp_h, 900))
        cv2.setMouseCallback(win, _mouse)
        cv2.imshow(win, clone)

        print("\n[ROI TOOL] Left-click to place corners, right-click to close a lane polygon.")
        print("[ROI TOOL] Lanes will be named lane_0, lane_1, ... Copy the printed lines into the script.\n")

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        for lid, poly in results.items():
            self.define_roi(lid, poly)

    # ──────────────────────────────────────────────────────────────
    # Core detection
    # ──────────────────────────────────────────────────────────────
    def _is_in_roi(self, cx: int, cy: int, polygon: list[tuple[int, int]]) -> bool:
        pt  = (float(cx), float(cy))
        pts = np.array(polygon, dtype=np.float32)
        return cv2.pointPolygonTest(pts, pt, False) >= 0

    def _update_tracker(self, track_id: int, cx: int, cy: int) -> bool:
        """
        Returns True if this vehicle has been stationary for STATIONARY_MIN_FRAMES.
        """
        history = self._history[track_id]
        history.append((cx, cy))

        if len(history) < STATIONARY_MIN_FRAMES:
            return False

        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        movement = max(max(xs) - min(xs), max(ys) - min(ys))
        return movement <= STATIONARY_THRESHOLD_PX

    def process_frame(self, frame: np.ndarray, frame_dt: float = 1/30) -> dict[str, int]:
        """
        Run detection on a single frame.
        frame_dt : seconds per frame (used to accumulate per-vehicle wait time).
        Returns {lane_id: halting_count} matching traci.lane.getLastStepHaltingNumber().
        Also populates self.wait_times: {lane_id: total_wait_seconds}.
        """
        # Downscale for faster CPU inference; ROI coords stay in original-pixel space
        if self.infer_width and frame.shape[1] > self.infer_width:
            self._scale = self.infer_width / frame.shape[1]
            infer_h = int(frame.shape[0] * self._scale)
            infer_frame = cv2.resize(frame, (self.infer_width, infer_h))
        else:
            self._scale = 1.0
            infer_frame = frame

        results = self.model.track(
            infer_frame,
            classes=VEHICLE_CLASSES,
            conf=self.conf,
            persist=True,
            verbose=False,
            device=self.device,
        )

        boxes = results[0].boxes
        self.vehicle_boxes = boxes

        counts    = {lane_id: 0   for lane_id in self._rois}
        wait_sums = {lane_id: 0.0 for lane_id in self._rois}

        # Decay wait time for track IDs not seen this frame
        seen_ids = set()

        if boxes is not None and boxes.id is not None:
            for box, track_id in zip(boxes.xyxy, boxes.id.int()):
                # Scale coords back to original-frame space so ROI polygons match
                x1 = int(box[0] / self._scale)
                y1 = int(box[1] / self._scale)
                x2 = int(box[2] / self._scale)
                y2 = int(box[3] / self._scale)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                tid = int(track_id)
                seen_ids.add(tid)

                is_stationary = self._update_tracker(tid, cx, cy)

                if is_stationary:
                    self._stationary_frames[tid] += 1
                    self._wait_seconds[tid] += frame_dt
                else:
                    self._stationary_frames[tid] = 0
                    self._wait_seconds[tid] = 0.0

                if is_stationary:
                    for lane_id, polygon in self._rois.items():
                        if self._is_in_roi(cx, cy, polygon):
                            counts[lane_id]    += 1
                            wait_sums[lane_id] += self._wait_seconds[tid]

        # Clean up tracks that have disappeared
        gone = set(self._wait_seconds.keys()) - seen_ids
        for tid in gone:
            self._wait_seconds.pop(tid, None)
            self._stationary_frames.pop(tid, None)
            self._history.pop(tid, None)

        self.queue_counts = counts
        self.wait_times   = wait_sums
        return counts

    # ──────────────────────────────────────────────────────────────
    # PPO observation helpers
    # ──────────────────────────────────────────────────────────────
    def to_ppo_obs(self, lane_order: list[str]) -> np.ndarray:
        """
        Converts current queue_counts into log-scaled values
        matching the observation format in sumo_env.py.

        lane_order: ordered list of lane IDs matching how the env indexes lanes.
        Returns a 1-D float32 array of length len(lane_order).
        """
        obs = []
        for lid in lane_order:
            h = self.queue_counts.get(lid, 0)
            obs.append(min(float(np.log1p(h) / LOG_OBS_MAX), 1.0))
        return np.array(obs, dtype=np.float32)

    def to_mappo_obs(
        self,
        lane_order: list[str],
        n_lanes: int = 8,
        n_out_lanes: int = 4,
        current_action: int = 0,
        n_actions: int = 3,
        phase_state: str = "GREEN",
    ) -> np.ndarray:
        """
        Build a single-TL local observation matching the MAPPO format in mappo_env.py.

        Structure (matches LOCAL_OBS_DIM = n_lanes*2 + n_out_lanes + 2):
          [queue_obs × n_lanes] [wait_obs × n_lanes] [out_obs × n_out_lanes]
          [action_norm] [phase_flag]

        lane_order  : controlled lane IDs in order (up to n_lanes used)
        n_out_lanes : outgoing lanes — set to 0 if you have no outgoing-lane ROIs
        current_action : current phase index (0-based)
        n_actions   : total number of phases for this TL
        phase_state : "GREEN" or "YELLOW"
        """
        # Queue obs (log-scaled halting count)
        queue_obs = []
        for lid in lane_order[:n_lanes]:
            h = self.queue_counts.get(lid, 0)
            queue_obs.append(min(float(np.log1p(h) / LOG_OBS_MAX), 1.0))
        while len(queue_obs) < n_lanes:
            queue_obs.append(0.0)

        # Wait obs (log-scaled cumulative wait seconds)
        wait_obs = []
        for lid in lane_order[:n_lanes]:
            w = self.wait_times.get(lid, 0.0)
            wait_obs.append(min(float(np.log1p(w) / LOG_OBS_MAX_WAIT), 1.0))
        while len(wait_obs) < n_lanes:
            wait_obs.append(0.0)

        # Outgoing-lane obs — zero-filled if no outgoing ROIs defined
        out_obs = [0.0] * n_out_lanes

        action_norm = current_action / max(n_actions - 1, 1)
        phase_flag  = 0.0 if phase_state == "GREEN" else 1.0

        return np.array(queue_obs + wait_obs + out_obs + [action_norm, phase_flag],
                        dtype=np.float32)

    # ──────────────────────────────────────────────────────────────
    # Headless CSV runner (no GUI, CPU-friendly)
    # ──────────────────────────────────────────────────────────────
    def run_headless(
        self,
        output_csv: str = "mappo_obs.csv",
        lane_order: list[str] | None = None,
        n_lanes: int = 8,
        n_out_lanes: int = 4,
        n_actions: int = 3,
        max_frames: int | None = None,
        print_every: int = 30,
    ):
        """
        Process video/camera without opening any window.
        Writes one CSV row per frame with:
          frame, timestamp_s, {lane}_queue, {lane}_wait_s  (raw)
          obs_0 … obs_N  (flattened MAPPO local obs vector)

        lane_order : ordered list of lane IDs to use as the obs vector.
                     Defaults to sorted(self._rois.keys()).
        max_frames : stop after this many frames (None = run to end of video).
        print_every: print a progress line every N frames.
        """
        if lane_order is None:
            lane_order = sorted(self._rois.keys())

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_dt = 1.0 / fps

        obs_dim  = n_lanes * 2 + n_out_lanes + 2   # matches LOCAL_OBS_DIM

        # Build CSV header
        raw_cols = []
        for lid in lane_order:
            raw_cols += [f"{lid}_queue", f"{lid}_wait_s"]
        obs_cols = [f"obs_{i}" for i in range(obs_dim)]
        header   = ["frame", "timestamp_s"] + raw_cols + obs_cols

        frame_idx = 0
        t0 = time.time()

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.process_frame(frame, frame_dt=frame_dt)

                obs = self.to_mappo_obs(
                    lane_order=lane_order,
                    n_lanes=n_lanes,
                    n_out_lanes=n_out_lanes,
                    n_actions=n_actions,
                )

                timestamp = frame_idx * frame_dt
                raw_vals  = []
                for lid in lane_order:
                    raw_vals.append(self.queue_counts.get(lid, 0))
                    raw_vals.append(round(self.wait_times.get(lid, 0.0), 3))

                writer.writerow([frame_idx, round(timestamp, 3)] + raw_vals + obs.tolist())

                if frame_idx % print_every == 0:
                    elapsed = time.time() - t0
                    print(f"  frame {frame_idx:5d} | t={timestamp:.1f}s | "
                          f"queue={self.queue_counts} | elapsed={elapsed:.1f}s")

                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break

        cap.release()
        print(f"\n[DETECTOR] Done. {frame_idx} frames written to '{output_csv}'.")

    # ──────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────
    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()

        # Draw ROI polygons
        for lane_id, polygon in self._rois.items():
            pts = np.array(polygon, dtype=np.int32)
            cv2.polylines(out, [pts], True, (0, 200, 255), 2)
            cx = int(np.mean([p[0] for p in polygon]))
            cy = int(np.mean([p[1] for p in polygon]))
            count = self.queue_counts.get(lane_id, 0)
            cv2.putText(out, f"{lane_id}: {count}", (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Draw bounding boxes
        if self.vehicle_boxes is not None and self.vehicle_boxes.xyxy is not None:
            for box in self.vehicle_boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(out, (x1, y1), (x2, y2), (50, 200, 50), 2)

        # Queue summary panel
        y_off = 30
        cv2.rectangle(out, (10, 10), (260, 30 + 28 * len(self._rois)), (0, 0, 0), -1)
        for lane_id, count in self.queue_counts.items():
            cv2.putText(out, f"{lane_id:20s} : {count:2d}", (15, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y_off += 28

        return out

    # ──────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────
    def run(self, display_width: int = 900):
        """
        Opens video source, processes frames, and displays annotated output.
        Press Q to quit.  Press R on the first frame to define ROIs interactively.
        Large frames are scaled to display_width for the window.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        print("[DETECTOR] Running — press Q to quit, R to define ROIs interactively.")

        win = "Vehicle Detector — Queue Extraction  [Q=quit  R=define ROIs]"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        first_frame = True
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.process_frame(frame, frame_dt=1.0/fps)
            display = self._draw_overlay(frame)

            # Scale display window to fit screen
            disp_scale = min(1.0, display_width / display.shape[1])
            disp = cv2.resize(display, (
                int(display.shape[1] * disp_scale),
                int(display.shape[0] * disp_scale)
            ))

            print(f"\rQueue: { {k: v for k, v in self.queue_counts.items()} }   ", end="")
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r') and first_frame:
                self.define_rois_interactive(frame, display_width=display_width)

            first_frame = False

        cap.release()
        cv2.destroyAllWindows()
        print("\n[DETECTOR] Stopped.")


# ──────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import os
    # Use the YOLO weights bundled in this folder, regardless of launch directory.
    _HERE = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(description="Vehicle detector → MAPPO observation CSV")
    ap.add_argument("--source",  default="0",           help="Camera index or video file path")
    ap.add_argument("--output",  default="mappo_obs.csv", help="Output CSV file")
    ap.add_argument("--gui",     action="store_true",   help="Show live GUI window (calibration)")
    ap.add_argument("--device",  default="cpu",         help="'cpu' or 'cuda'")
    ap.add_argument("--conf",    type=float, default=0.4)
    ap.add_argument("--frames",  type=int,  default=None, help="Max frames to process")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    detector = VehicleDetector(
        source=src,
        model_path=os.path.join(_HERE, "yolov8n.pt"),   # smallest/fastest YOLO model
        conf=args.conf,
        device=args.device,        # use "cpu" on weak-GPU machines
    )

    # ── Define your ROIs here (pixel coordinates for your camera view) ──────
    # These are placeholders — replace with your actual intersection geometry,
    # or run with --gui and press R on the first frame to draw interactively.
    detector.define_roi("lane_N1", [(100, 400), (200, 400), (210, 600), (90,  600)])
    detector.define_roi("lane_N2", [(210, 400), (310, 400), (320, 600), (200, 600)])
    detector.define_roi("lane_S1", [(350, 100), (450, 100), (460, 300), (340, 300)])
    detector.define_roi("lane_E1", [(500, 250), (620, 250), (630, 380), (490, 380)])

    # Lane order must match the MAPPO controlled-lane ordering for your TL
    LANE_ORDER = ["lane_N1", "lane_N2", "lane_S1", "lane_E1"]

    if args.gui:
        detector.run()
    else:
        print(f"[DETECTOR] Headless mode → writing to '{args.output}'")
        print(f"[DETECTOR] Device: {args.device}  Source: {src}")
        detector.run_headless(
            output_csv=args.output,
            lane_order=LANE_ORDER,
            n_lanes=8,        # matches MAPPO N_LANES
            n_out_lanes=4,    # matches MAPPO N_OUT_LANES
            n_actions=3,      # matches MAPPO N_ACTIONS
            max_frames=args.frames,
        )
