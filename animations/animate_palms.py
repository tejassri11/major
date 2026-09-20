"""
PALMS: Multi-Agent Traffic Signal Control - Manim Animation
============================================================
Renders 4 scenes explaining the system to a non-technical audience.

Install:  pip install manim
Render:   manim -pql animate_palms.py PALMSAll     # quick low-quality preview
          manim -pqh animate_palms.py PALMSAll     # high quality (slow)
          manim -pql animate_palms.py ObsScene     # single scene
"""

from manim import *
import numpy as np

# ── shared palette ──────────────────────────────────────────────────────────
C_BG     = "#0f1117"
C_GREEN  = "#00c896"
C_YELLOW = "#f5c542"
C_RED    = "#e05252"
C_BLUE   = "#4ea8de"
C_GRAY   = "#8899aa"
C_WHITE  = "#e8eaf0"
C_PURPLE = "#b07fff"
C_ORANGE = "#ff9a3c"

# ── Scene 1: System Overview ─────────────────────────────────────────────────
class OverviewScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        title = Text("PALMS", font_size=72, color=C_GREEN, weight=BOLD)
        sub   = Text("Multi-Agent Deep RL · Traffic Signal Control · Palapye",
                     font_size=22, color=C_GRAY)
        VGroup(title, sub).arrange(DOWN, buff=0.3).move_to(ORIGIN)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=UP*0.3))
        self.wait(0.8)
        self.play(VGroup(title, sub).animate.to_edge(UP, buff=0.4))
        self.wait(0.3)

        # ── 3 traffic lights ──────────────────────────────────────────────
        tl_labels = ["TL_1", "TL_2", "TL_3"]
        tl_colors = [C_GREEN, C_YELLOW, C_GREEN]
        tl_positions = [LEFT * 4, ORIGIN, RIGHT * 4]

        traffic_lights = VGroup()
        for label, color, pos in zip(tl_labels, tl_colors, tl_positions):
            box   = RoundedRectangle(width=1.4, height=3.6, corner_radius=0.15,
                                     color=C_GRAY, stroke_width=2, fill_color="#1a1d26",
                                     fill_opacity=1)
            r = Circle(radius=0.35, color=C_RED,    fill_color=C_RED    if color == C_RED    else "#3a1a1a", fill_opacity=1)
            y = Circle(radius=0.35, color=C_YELLOW, fill_color=C_YELLOW if color == C_YELLOW else "#3a2e0a", fill_opacity=1)
            g = Circle(radius=0.35, color=C_GREEN,  fill_color=C_GREEN  if color == C_GREEN  else "#0a2e1a", fill_opacity=1)
            lamps = VGroup(r, y, g).arrange(DOWN, buff=0.18)
            lamps.move_to(box)
            lbl = Text(label, font_size=18, color=C_WHITE).next_to(box, DOWN, buff=0.2)
            tl  = VGroup(box, lamps, lbl).move_to(pos + DOWN * 0.3)
            traffic_lights.add(tl)

        # Connecting road lines
        roads = VGroup(
            Line(tl_positions[0] + RIGHT*0.7 + DOWN*0.3,
                 tl_positions[1] + LEFT*0.7  + DOWN*0.3, color=C_GRAY, stroke_width=2),
            Line(tl_positions[1] + RIGHT*0.7 + DOWN*0.3,
                 tl_positions[2] + LEFT*0.7  + DOWN*0.3, color=C_GRAY, stroke_width=2),
        )

        self.play(LaggedStartMap(FadeIn, traffic_lights, lag_ratio=0.3), run_time=1.5)
        self.play(Create(roads))
        self.wait(0.5)

        # Annotate agent count
        agent_note = Text("3 Agents  ·  3 Actions each  ·  Shared cooperative reward",
                          font_size=20, color=C_BLUE).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(agent_note, shift=UP*0.2))
        self.wait(1.5)

        # Fade out for next scene
        self.play(FadeOut(VGroup(traffic_lights, roads, agent_note, title, sub)))
        self.wait(0.3)


# ── Scene 2: Observation Space ───────────────────────────────────────────────
class ObsScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        # Title
        heading = Text("What Does Each Agent See?", font_size=36, color=C_WHITE, weight=BOLD)
        sub     = Text("22-dimensional local observation vector per traffic light",
                       font_size=20, color=C_GRAY)
        VGroup(heading, sub).arrange(DOWN, buff=0.2).to_edge(UP, buff=0.5)
        self.play(Write(heading), FadeIn(sub, shift=UP*0.2))
        self.wait(0.5)

        # Observation breakdown
        obs_groups = [
            ("Queue lengths",    8,  C_RED,    "[0:8]",   "halting vehicles per incoming lane\nlog₁₊(count) / log₁₊(15)  →  [0, 1]"),
            ("Waiting times",    8,  C_ORANGE, "[8:16]",  "seconds waited per incoming lane\nlog₁₊(secs) / log₁₊(300)  →  [0, 1]"),
            ("Outgoing traffic", 4,  C_BLUE,   "[16:20]", "vehicle count on exit lanes\nlog₁₊(count) / log₁₊(15)  →  [0, 1]"),
            ("Current phase",    1,  C_PURPLE, "[20]",    "action index / (n_phases - 1)\n→  0.0, 0.5, or 1.0"),
            ("Phase state",      1,  C_GREEN,  "[21]",    "0.0 = GREEN  |  1.0 = YELLOW"),
        ]

        # Build the 22-cell vector bar
        cell_w, cell_h = 0.38, 0.55
        all_cells = VGroup()
        color_map = []
        for name, n, color, idx, desc in obs_groups:
            for _ in range(n):
                color_map.append(color)

        for i, color in enumerate(color_map):
            cell = Rectangle(width=cell_w, height=cell_h,
                             fill_color=color, fill_opacity=0.75,
                             stroke_color=C_BG, stroke_width=1.5)
            all_cells.add(cell)

        all_cells.arrange(RIGHT, buff=0.04).move_to(DOWN * 0.2)

        # Index labels below the bar
        dim_label = Text("dim: 22", font_size=16, color=C_GRAY).next_to(all_cells, DOWN, buff=0.15)

        self.play(LaggedStartMap(FadeIn, all_cells, lag_ratio=0.04), run_time=1.4)
        self.play(FadeIn(dim_label))
        self.wait(0.3)

        # Highlight each group and show description
        cursor = 0
        for name, n, color, idx, desc in obs_groups:
            group_cells = all_cells[cursor:cursor + n]
            brace = Brace(group_cells, direction=DOWN, color=color)
            brace_label = Text(f"{idx}  {name}", font_size=14, color=color)\
                          .next_to(brace, DOWN, buff=0.1)

            info_box = VGroup(
                Text(name, font_size=22, color=color, weight=BOLD),
                Text(desc, font_size=16, color=C_WHITE, t2c={"log₁₊": C_YELLOW, "0.0": C_GREEN, "1.0": C_YELLOW}),
            ).arrange(DOWN, buff=0.15, aligned_edge=LEFT).to_edge(DOWN, buff=0.55)

            self.play(
                group_cells.animate.set_fill(opacity=1.0),
                Create(brace), Write(brace_label),
                FadeIn(info_box, shift=UP*0.15),
                run_time=0.8,
            )
            self.wait(1.0)
            self.play(
                group_cells.animate.set_fill(opacity=0.75),
                FadeOut(brace), FadeOut(brace_label), FadeOut(info_box),
                run_time=0.5,
            )
            cursor += n

        # Show global state
        global_note = VGroup(
            Text("Global State (Centralized Critic only):", font_size=20, color=C_GRAY),
            Text("66 dims  =  3 agents × 22", font_size=26, color=C_WHITE, weight=BOLD),
        ).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.6)

        self.play(FadeIn(global_note, shift=UP*0.2))
        self.wait(1.5)
        self.play(FadeOut(VGroup(all_cells, dim_label, heading, sub, global_note)))
        self.wait(0.2)


# ── Scene 3: Reward Function ──────────────────────────────────────────────────
class RewardScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        heading = Text("How Is Reward Computed?", font_size=36, color=C_WHITE, weight=BOLD)
        sub     = Text("Shared cooperative signal — all 3 agents receive the same r",
                       font_size=20, color=C_GRAY)
        VGroup(heading, sub).arrange(DOWN, buff=0.2).to_edge(UP, buff=0.5)
        self.play(Write(heading), FadeIn(sub))
        self.wait(0.4)

        # Example values
        ex_wait    = 480.0    # total waiting seconds across all lanes
        ex_queue   = 12.0     # halting vehicles
        ex_lanes   = 24       # 3 TLs × 8 lanes
        ex_col     = 0
        ex_tel     = 1

        # Formula components
        components = [
            (
                r"r_1 = -\tanh\!\left(\frac{\text{total\_wait}}{n\_lanes \times 60}\right)",
                r"r_1 = -\tanh\!\left(\frac{480}{24 \times 60}\right) = -\tanh(0.33) \approx -0.32",
                C_RED, "Primary: penalise cumulative waiting time"
            ),
            (
                r"r_2 = -0.1 \times \frac{\text{total\_queue}}{n\_lanes \times 15}",
                r"r_2 = -0.1 \times \frac{12}{24 \times 15} = -0.1 \times 0.033 \approx -0.003",
                C_ORANGE, "Secondary: penalise queue build-up"
            ),
            (
                r"r_3 = -0.5 \times \text{collisions}",
                r"r_3 = -0.5 \times 0 = 0.0",
                C_PURPLE, "Safety: collisions are expensive"
            ),
            (
                r"r_4 = -0.3 \times \text{teleports}",
                r"r_4 = -0.3 \times 1 = -0.3",
                C_YELLOW, "Jam proxy: teleport = vehicle stuck too long"
            ),
        ]

        y_pos = 1.6
        running_r = 0.0
        displayed = VGroup()

        for formula, example, color, label in components:
            form_tex = MathTex(formula, font_size=28, color=color).move_to(UP * y_pos + LEFT * 1.5)
            ex_tex   = MathTex(example,  font_size=22, color=C_GRAY).next_to(form_tex, DOWN, buff=0.06)
            lbl_tex  = Text(label, font_size=15, color=C_WHITE).next_to(form_tex, RIGHT, buff=0.4)

            self.play(Write(form_tex), run_time=0.7)
            self.play(FadeIn(ex_tex, shift=UP*0.1), FadeIn(lbl_tex, shift=LEFT*0.1))
            self.wait(0.5)
            displayed.add(form_tex, ex_tex, lbl_tex)
            y_pos -= 1.05

        # Sum line
        r_raw   = -0.32 - 0.003 + 0.0 - 0.3
        r_clip  = np.clip(r_raw, -2.0, 0.5)
        sum_tex = MathTex(
            rf"r = r_1 + r_2 + r_3 + r_4 \approx {r_raw:.3f}",
            font_size=26, color=C_WHITE
        ).to_edge(DOWN, buff=1.1)
        clip_tex = MathTex(
            rf"\text{{clip}}(r,\,-2.0,\,0.5) = {r_clip:.3f}",
            font_size=26, color=C_GREEN
        ).next_to(sum_tex, DOWN, buff=0.2)

        self.play(Write(sum_tex))
        self.play(Write(clip_tex))
        self.wait(2.0)
        self.play(FadeOut(VGroup(displayed, sum_tex, clip_tex, heading, sub)))
        self.wait(0.2)


# ── Scene 4: MAPPO Architecture (CTDE) ───────────────────────────────────────
class MAPPOScene(Scene):
    def construct(self):
        self.camera.background_color = C_BG

        heading = Text("MAPPO: Centralized Training, Decentralized Execution",
                       font_size=28, color=C_WHITE, weight=BOLD)
        heading.to_edge(UP, buff=0.4)
        self.play(Write(heading))
        self.wait(0.4)

        def make_layer(label, width, color, height=0.55):
            box = Rectangle(width=width, height=height,
                            fill_color=color, fill_opacity=0.85,
                            stroke_color=C_WHITE, stroke_width=1.2)
            txt = Text(label, font_size=14, color=C_BG, weight=BOLD).move_to(box)
            return VGroup(box, txt)

        # ── Actor ────────────────────────────────────────────────
        actor_title = Text("Actor  (runs at execution)", font_size=20, color=C_BLUE).move_to(LEFT*3.2 + UP*1.8)

        obs_box   = make_layer("obs  [22]",      2.2, C_BLUE)
        h1a_box   = make_layer("Hidden  [256]",  2.8, "#2a4a6a")
        h2a_box   = make_layer("Hidden  [256]",  2.8, "#2a4a6a")
        act_box   = make_layer("logits  [3]",    2.2, C_GREEN)
        actor_net = VGroup(obs_box, h1a_box, h2a_box, act_box)\
                    .arrange(DOWN, buff=0.28).move_to(LEFT*3.2 + DOWN*0.2)

        actor_arrows = VGroup(
            Arrow(obs_box.get_bottom(), h1a_box.get_top(), buff=0.05, color=C_GRAY, stroke_width=2),
            Arrow(h1a_box.get_bottom(), h2a_box.get_top(), buff=0.05, color=C_GRAY, stroke_width=2),
            Arrow(h2a_box.get_bottom(), act_box.get_top(), buff=0.05, color=C_GRAY, stroke_width=2),
        )

        tanh_labels = VGroup(
            Text("Tanh", font_size=12, color=C_YELLOW).next_to(actor_arrows[0], RIGHT, buff=0.05),
            Text("Tanh", font_size=12, color=C_YELLOW).next_to(actor_arrows[1], RIGHT, buff=0.05),
        )

        act_note = Text("→ sample action (0, 1, or 2)", font_size=13, color=C_GREEN)\
                   .next_to(act_box, DOWN, buff=0.15)

        self.play(FadeIn(actor_title))
        self.play(LaggedStartMap(FadeIn, actor_net, lag_ratio=0.25), run_time=1.0)
        self.play(LaggedStartMap(Create, actor_arrows, lag_ratio=0.2),
                  FadeIn(tanh_labels), FadeIn(act_note))
        self.wait(0.5)

        # ── Critic ────────────────────────────────────────────────
        critic_title = Text("Centralized Critic  (training only)", font_size=20, color=C_RED)\
                       .move_to(RIGHT*3.2 + UP*1.8)

        gs_box    = make_layer("global state  [66]", 2.8, C_RED)
        h1c_box   = make_layer("Hidden  [256]",      2.8, "#5a1a1a")
        h2c_box   = make_layer("Hidden  [256]",      2.8, "#5a1a1a")
        val_box   = make_layer("value  [1]",          2.2, C_ORANGE)
        critic_net = VGroup(gs_box, h1c_box, h2c_box, val_box)\
                     .arrange(DOWN, buff=0.28).move_to(RIGHT*3.2 + DOWN*0.2)

        critic_arrows = VGroup(
            Arrow(gs_box.get_bottom(),  h1c_box.get_top(), buff=0.05, color=C_GRAY, stroke_width=2),
            Arrow(h1c_box.get_bottom(), h2c_box.get_top(), buff=0.05, color=C_GRAY, stroke_width=2),
            Arrow(h2c_box.get_bottom(), val_box.get_top(), buff=0.05, color=C_GRAY, stroke_width=2),
        )

        tanh_labels_c = VGroup(
            Text("Tanh", font_size=12, color=C_YELLOW).next_to(critic_arrows[0], RIGHT, buff=0.05),
            Text("Tanh", font_size=12, color=C_YELLOW).next_to(critic_arrows[1], RIGHT, buff=0.05),
        )

        val_note = Text("→ estimate future reward", font_size=13, color=C_ORANGE)\
                   .next_to(val_box, DOWN, buff=0.15)

        self.play(FadeIn(critic_title))
        self.play(LaggedStartMap(FadeIn, critic_net, lag_ratio=0.25), run_time=1.0)
        self.play(LaggedStartMap(Create, critic_arrows, lag_ratio=0.2),
                  FadeIn(tanh_labels_c), FadeIn(val_note))
        self.wait(0.5)

        # ── Global state = concat of all 3 local obs ─────────────
        concat_note = Text(
            "global state [66]  =  obs_TL1 [22]  ‖  obs_TL2 [22]  ‖  obs_TL3 [22]",
            font_size=15, color=C_GRAY
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(concat_note, shift=UP*0.2))

        # Parameter sharing badge
        badge = VGroup(
            RoundedRectangle(width=4.5, height=0.55, corner_radius=0.12,
                             fill_color="#1a2a1a", fill_opacity=1,
                             stroke_color=C_GREEN, stroke_width=1.5),
            Text("Parameter Sharing: all 3 agents use the same Actor weights",
                 font_size=14, color=C_GREEN),
        )
        badge[1].move_to(badge[0])
        badge.to_edge(DOWN, buff=1.3)
        self.play(FadeIn(badge))
        self.wait(2.5)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.2)


# ── Composite: render all 4 scenes end-to-end ────────────────────────────────
class PALMSAll(Scene):
    def construct(self):
        for SceneCls in [OverviewScene, ObsScene, RewardScene, MAPPOScene]:
            SceneCls.construct(self)
            self.wait(0.5)
