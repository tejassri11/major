from manim import *


class SystemOverview(Scene):
    def construct(self):
        self._title()
        self._loop()
        self._observation()
        self._reward()
        self._phase_transition()

    # ── 1. Title ─────────────────────────────────────────────────────────────
    def _title(self):
        title = Text("DRL Traffic Signal Control", font_size=44, weight=BOLD)
        sub = Text("Single-Agent PPO  ·  Palapye Intersection  ·  SUMO", font_size=22, color=GRAY)
        VGroup(title, sub).arrange(DOWN, buff=0.3).move_to(ORIGIN)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(sub))

    # ── 2. Control loop ───────────────────────────────────────────────────────
    def _loop(self):
        # ── boxes ──
        sumo  = self._box("SUMO\nSimulation", BLUE_D,   LEFT * 4.5)
        traci = self._box("TraCI\nBridge",    YELLOW_D, ORIGIN)
        ppo   = self._box("PPO\nAgent",       GREEN_D,  RIGHT * 4.5)

        self.play(FadeIn(sumo), FadeIn(traci), FadeIn(ppo))
        self.wait(0.5)

        # ── arrows ──
        obs_arrow = self._labeled_arrow(
            traci.get_right(), ppo.get_left(),
            "observation\n(19-D vector)", UP, GREEN_C,
        )
        act_arrow = self._labeled_arrow(
            ppo.get_left(), traci.get_right(),
            "action\n(phase 0-3)", DOWN, ORANGE,
            tip_at_start=True,
        )
        traci_sumo_arrow = self._labeled_arrow(
            traci.get_left(), sumo.get_right(),
            "setPhase()", DOWN, RED_C,
            tip_at_start=True,
        )
        sumo_traci_arrow = self._labeled_arrow(
            sumo.get_right(), traci.get_left(),
            "lane queues\n& wait times", UP, BLUE_C,
        )
        rew_arrow = self._reward_arrow(sumo, ppo)

        self.play(GrowArrow(obs_arrow[0]), Write(obs_arrow[1]))
        self.play(GrowArrow(act_arrow[0]), Write(act_arrow[1]))
        self.play(GrowArrow(sumo_traci_arrow[0]), Write(sumo_traci_arrow[1]))
        self.play(GrowArrow(traci_sumo_arrow[0]), Write(traci_sumo_arrow[1]))
        self.play(GrowArrow(rew_arrow[0]), Write(rew_arrow[1]))
        self.wait(2)

        loop_label = Text("One RL step = 1 sim-second", font_size=20, color=LIGHT_GRAY)
        loop_label.to_edge(DOWN, buff=0.4)
        self.play(Write(loop_label))
        self.wait(2)

        self.play(*[FadeOut(m) for m in self.mobjects])

    # ── 3. Observation breakdown ──────────────────────────────────────────────
    def _observation(self):
        title = Text("Observation Vector  (19 values)", font_size=32, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        rows = [
            ("7 × queue",     "halting vehicles per incoming lane",  BLUE_C),
            ("7 × wait",      "max accumulated wait per lane (s)",   GREEN_C),
            ("4 × phase",     "one-hot: which green phase is active",YELLOW_C),
            ("1 × elapsed",   "time since last phase switch (norm)", ORANGE),
        ]

        entries = VGroup()
        for label, desc, color in rows:
            box  = Rectangle(width=2.2, height=0.65, color=color, fill_opacity=0.2)
            lbl  = Text(label, font_size=20, color=color).move_to(box)
            desc_txt = Text(desc, font_size=17, color=LIGHT_GRAY)
            row = VGroup(VGroup(box, lbl), desc_txt).arrange(RIGHT, buff=0.4)
            entries.add(row)

        entries.arrange(DOWN, buff=0.35).next_to(title, DOWN, buff=0.5)

        for row in entries:
            self.play(FadeIn(row, shift=RIGHT * 0.3), run_time=0.6)
        self.wait(2.5)
        self.play(FadeOut(title), FadeOut(entries))

    # ── 4. Reward function ────────────────────────────────────────────────────
    def _reward(self):
        title = Text("Reward Function", font_size=32, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        formula = MathTex(
            r"r = -\Big(",
            r"0.5 \cdot \bar{q}",
            r"+\; 0.3 \cdot p",
            r"+\; 0.2 \cdot w_{\max}",
            r"\Big)",
            font_size=42,
        )
        formula.set_color_by_tex(r"\bar{q}", BLUE_C)
        formula.set_color_by_tex(r"p",       YELLOW_C)
        formula.set_color_by_tex(r"w",       RED_C)
        formula.next_to(title, DOWN, buff=0.7)
        self.play(Write(formula))

        legend = VGroup(
            self._legend_item(r"\bar{q}", "avg normalised queue length",  BLUE_C),
            self._legend_item(r"p",       "pressure  (incoming − outgoing vehicles)", YELLOW_C),
            self._legend_item(r"w_{\max}","max single-vehicle waiting time",          RED_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(formula, DOWN, buff=0.6)

        for item in legend:
            self.play(FadeIn(item, shift=UP * 0.2), run_time=0.5)
        self.wait(2.5)
        self.play(FadeOut(title), FadeOut(formula), FadeOut(legend))

    # ── 5. Phase transition ───────────────────────────────────────────────────
    def _phase_transition(self):
        title = Text("Safe Phase Transition", font_size=32, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        phases = ["GREEN\n(current)", "YELLOW", "ALL-RED", "GREEN\n(target)"]
        colors = [GREEN,              YELLOW,   RED,        GREEN]
        boxes  = VGroup(*[
            VGroup(
                RoundedRectangle(width=2.0, height=1.0, corner_radius=0.15,
                                 color=c, fill_color=c, fill_opacity=0.25),
                Text(p, font_size=18),
            ).arrange(ORIGIN)
            for p, c in zip(phases, colors)
        ]).arrange(RIGHT, buff=0.5).next_to(title, DOWN, buff=0.8)

        arrows = VGroup(*[
            Arrow(boxes[i].get_right(), boxes[i+1].get_left(), buff=0.1, color=GRAY)
            for i in range(len(boxes)-1)
        ])

        self.play(FadeIn(boxes[0]))
        for i in range(len(arrows)):
            self.play(
                GrowArrow(arrows[i]),
                FadeIn(boxes[i+1]),
                run_time=0.7,
            )
        self.wait(1)

        note = Text(
            "If turning queue ≥ 3:  insert Phase 6 (turns) for 15 steps\nbefore Phase 9 (straight)  →  prevents right-turn spillback",
            font_size=19, color=LIGHT_GRAY, line_spacing=1.4,
        ).to_edge(DOWN, buff=0.6)
        self.play(Write(note))
        self.wait(3)
        self.play(FadeOut(title), FadeOut(boxes), FadeOut(arrows), FadeOut(note))

        end = Text("End", font_size=40, color=GRAY)
        self.play(Write(end))
        self.wait(1)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _box(self, label, color, pos):
        rect = RoundedRectangle(width=2.8, height=1.6, corner_radius=0.2,
                                color=color, fill_color=color, fill_opacity=0.15)
        txt  = Text(label, font_size=22, color=color)
        return VGroup(rect, txt).arrange(ORIGIN).move_to(pos)

    def _labeled_arrow(self, start, end, label, label_dir, color, tip_at_start=False):
        if tip_at_start:
            arr = Arrow(end, start, buff=0.15, color=color)
        else:
            arr = Arrow(start, end, buff=0.15, color=color)
        lbl = Text(label, font_size=16, color=color).next_to(arr, label_dir, buff=0.15)
        return arr, lbl

    def _reward_arrow(self, sumo, ppo):
        path = ArcBetweenPoints(sumo.get_bottom(), ppo.get_bottom(), angle=-TAU / 4)
        arr  = Arrow(sumo.get_bottom(), ppo.get_bottom(), path_arc=-TAU / 4, buff=0.15, color=RED_C)
        lbl  = Text("reward", font_size=16, color=RED_C).next_to(path, DOWN, buff=0.2)
        return arr, lbl

    def _legend_item(self, sym, desc, color):
        math = MathTex(sym, font_size=28, color=color)
        dash = Text(" — ", font_size=22, color=GRAY)
        text = Text(desc, font_size=20, color=LIGHT_GRAY)
        return VGroup(math, dash, text).arrange(RIGHT, buff=0.1)
