![Banner](https://capsule-render.vercel.app/api?type=waving&color=0:141E30,100:243B55&height=200&text=PALMS&fontSize=50&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Multi-Agent+Traffic+Signal+Control+%C2%B7+BIUST&descAlignY=56&descAlign=50)

# Multi-Agent Deep Reinforcement Learning for Adaptive Traffic Signal Control in the Palapye A1 Corridor

> Three traffic lights. One cooperative neural network. MAPPO-trained agents reduce waiting times across a real-world road network in Palapye, Botswana - trained with CTDE: centralised training, decentralised execution.

![Last Commit](https://img.shields.io/github/last-commit/geranimoekia/Multi-Agent-Deep-Reinforcement-Learning-for-Adaptive-Traffic-Signal-Control-Palapye-A1-Corridor?style=for-the-badge&color=0e75b6)
![License](https://img.shields.io/github/license/geranimoekia/Multi-Agent-Deep-Reinforcement-Learning-for-Adaptive-Traffic-Signal-Control-Palapye-A1-Corridor?style=for-the-badge&color=brightgreen)

### Tsotlhe Nayang Seiphepi · Final Year Project · BIUST

Three traffic lights. One cooperative policy. Trained with MAPPO to minimise waiting times across a real-world road network.


## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-111F68?style=for-the-badge&logo=ultralytics&logoColor=white)
![SUMO](https://img.shields.io/badge/SUMO_Simulation-00AFBD?style=for-the-badge&logoColor=white)

| Tool | Purpose |
|---|---|
| **SUMO 1.24+** | Microscopic traffic simulation |
| **TraCI** | Python API for real-time SUMO control |
| **PyTorch** | Neural networks and PPO optimisation |
| **Streamlit + Plotly** | Live training dashboard |
| **NumPy / Gymnasium** | Environment interface and rollout buffer |
| **YOLOv8** | Real-world vehicle detection from video |

---

## What This Is

PALMS trains a multi-agent reinforcement learning system to control the three signalised intersections along the A1 highway corridor through Palapye. Each traffic light is an independent agent that observes its own queue lengths and waiting times, then decides which green phase to hold or switch to. The agents share a single neural network (parameter sharing) and are trained cooperatively - they all benefit when the whole network flows well, not just one junction.

The algorithm is **MAPPO** (Multi-Agent Proximal Policy Optimisation) with a **Centralised Critic** - during training, the critic sees the full network state (all three intersections at once) to produce better value estimates, but at deployment each agent makes decisions using only its own local sensor data.

---

## Architecture

```mermaid
graph LR
    subgraph Training
        GS[Global State 66-dim]
        CC[Centralised Critic]
        V[Value Estimate]
        GS --> CC --> V
    end
    subgraph Deployment
        LO[Local Obs 22-dim per agent]
        AC[Actor shared weights]
        ACT[Action 0 or 1 or 2]
        LO --> AC --> ACT
    end
    Training -. shared Actor weights .-> Deployment
```

PALMS uses **CTDE (Centralised Training, Decentralised Execution)**. During training the critic sees the full network state across all three junctions to produce accurate value estimates. At deployment only the Actor runs, using each agent's own local sensors.

---

### Actor Network (Policy)

All three traffic lights share a single Actor instance. Each agent feeds its own 22-number local observation through the network independently and receives a phase decision.

**Input - 22 values per agent**

| Index | Feature | Description |
|-------|---------|-------------|
| 0 - 7 | Queue length per incoming lane | Halting vehicles, log-scaled |
| 8 - 15 | Waiting time per incoming lane | Cumulative wait in seconds, log-scaled |
| 16 - 19 | Vehicle count on outgoing lanes | Downstream congestion, log-scaled |
| 20 | Current green phase | Normalised 0 to 1 |
| 21 | Phase state flag | 0 = green, 1 = yellow |

**Layer structure**

```
Input (22)  →  Linear(256)  →  Tanh  →  Linear(256)  →  Tanh  →  Linear(3)
```

**Output** - 3 raw logits, one per green phase. A Categorical distribution converts them to probabilities. During training the action is sampled for exploration; during evaluation the highest-probability phase is always chosen.

---

### Centralised Critic (Value Network)

Used only during training and discarded at deployment. The critic receives the full global state - all three agents' observations concatenated - and returns a single scalar value V(s) used to compute advantages for the PPO update.

**Input - 66 values (3 agents x 22)**

```
[ Agent 1 obs (22) | Agent 2 obs (22) | Agent 3 obs (22) ]  =  66-dim global state
```

By seeing all queues, waiting times, and phases across every junction at once, the critic can accurately judge joint situations - for example, whether clearing one junction is causing a knock-on queue at another.

**Layer structure**

```
Input (66)  →  Linear(256)  →  Tanh  →  Linear(256)  →  Tanh  →  Linear(1)
```

---

### Parameter Sharing

One Actor network is shared across all three traffic lights rather than training three separate networks. This works because every intersection has the same lane structure and observation format, so the same decision rules generalise across junction positions.

| Benefit | Detail |
|---------|--------|
| Fewer parameters | 1x network instead of 3x - faster convergence |
| More gradient signal | Each environment step produces 3 training samples |
| Implicit coordination | Agents learn a policy that works well across all positions in the network |

---

### Training Data Flow

```
4 parallel SUMO environments collect experience simultaneously
              |
              v
  GAE - Generalised Advantage Estimation
  estimates how much better/worse each action was than expected
  (lambda = 0.95, gamma = 0.99)
              |
              v
  PPO Update - clip ratio epsilon = 0.2
  updates Actor and Critic weights without taking destructive steps
              |
              v
  Evaluate on held-out episodes
  save best_actor.pth if mean reward improves
```

---

## Project Structure

```
PALMS-Multi-Agent-Deep-Reinforcement-Learning-Traffic-Signal-Control-Palapye/
│
├── src/                          # All Python source (run scripts from the repo root)
│   ├── mappo_env.py              # Multi-agent SUMO environment (reset / step / reward)
│   ├── mappo_networks.py         # Actor + CentralizedCritic neural networks
│   ├── train_mappo.py            # Main training loop (rollout → GAE → PPO update)
│   │
│   ├── sumo_env.py               # Single-agent PPO environment (baseline)
│   ├── train_ppo.py              # Single-agent PPO training (baseline)
│   ├── dashboard.py              # Streamlit live monitor
│   │
│   ├── traffic_injector.py       # Dynamic vehicle injection by scenario
│   ├── traffic_scenario.py       # Demand scenario profiles (rush hour, holiday, etc.)
│   ├── green_wave.py             # Rule-based green-wave offset pre-computation
│   ├── tl_programs.py            # SUMO traffic light phase program loader
│   │
│   ├── evaluate_mappo.py         # Evaluate MAPPO vs fixed-time / baselines
│   ├── evaluate_network.py       # Network-wide evaluation across junctions
│   └── watch_mappo.py, demo_ctde.py, ...   # Live viewers & demos
│
├── Performance Analysis/         # Controller comparison scripts (run from repo root)
│   ├── mappo_vs_ppo_vs_fixed_network.py        # All 3 controllers, full network (bar charts)
│   ├── mappo_vs_ppo_vs_fixed_junction.py       # All 3 controllers at junction TL_A (bar charts)
│   ├── mappo_vs_ppo_vs_fixed_junction_live.py  # Same, per-step line graphs
│   ├── mappo_vs_fixed_network_live.py          # MAPPO vs Fixed, network time-series
│   └── ppo_vs_mappo_live_gui.py                # Live dual SUMO-GUI: SA-PPO vs MAPPO
│
├── network/                      # SUMO simulation files
│   ├── triple.sumocfg            # Simulation config entry point
│   ├── network_tripled.net.xml
│   ├── triple_routes_flows.rou.xml
│   ├── tls.add.xml               # Custom traffic light phase programs
│   └── vtypes.add.xml            # Vehicle type definitions
│
├── vehicle_detection/            # YOLOv8 vehicle detection (run from repo root)
│   ├── detect_traffic.py         # Runs YOLOv8 over traffic.mp4, counts/queues vehicles
│   ├── vehicle_detector.py       # Reusable detector class → MAPPO observation CSV
│   ├── traffic.mp4               # Source footage (gitignored - large)
│   ├── first_frame.jpg           # Calibration still (first frame of traffic.mp4)
│   └── yolov8n.pt, yolov8s.pt    # YOLO weights (gitignored - large)
│
├── tools/                        # Utilities (plotting, TLS editor, presentation gen)
├── animations/                   # Manim animations of the system
├── docs/                         # Project report (LaTeX) - see "Project Report" below
│   ├── main.tex, references.bib
│   ├── figures/                  # All report figures
│   └── presentations/            # Slide decks (.pptx)
│
├── mappo_models/                 # Saved checkpoints - best_/final_ tracked; numbered ones gitignored
├── mappo_logs/                   # CSV training log (gitignored)
├── output/                       # Generated evaluation plots/CSVs (gitignored)
├── SPPO_model/                   # Archive of the original single-agent PPO version
├── requirements.txt
└── README.md
```

> **Note on running scripts:** the Python modules use flat imports, so always run
> them **from the repository root** with the `src/` prefix, e.g.
> `python src/train_mappo.py`. This keeps both module imports and the relative
> data paths (`network/`, `mappo_models/`) resolving correctly.

---

## Setup

**Requirements:** Python 3.10+, SUMO 1.24+

```bash
# Create virtual environment
python -m venv rl_env
rl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set SUMO_HOME (Windows - adjust path if needed)
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
```

---

## Training

```bash
python src/train_mappo.py
```

- **4 parallel SUMO environments** - diverse experience, faster data collection
- **1.5 million timesteps** - converges in ~3–4 hours on CPU
- **Curriculum learning** - starts on easy scenarios, adds rush hour and holiday as training progresses
- Saves `mappo_models/best_actor.pth` whenever evaluation reward improves
- Logs every update to `mappo_logs/mappo_train.csv` - open in Excel or plot with pandas

### Curriculum Schedule

| Episode count | Scenarios in pool |
|---|---|
| 0 – 29 | `low`, `normal` |
| 30 – 79 | `normal`, `rush_hour_am`, `rush_hour_pm`, `holiday` |
| 80+ | All scenarios (full random) |

### Training Output (example)

```
──────────────────────────────────────────────────────────────
  Progress  [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  20.0%
  Steps        300,000 / 1,500,000   ETA 02h 41m 03s
──────────────────────────────────────────────────────────────
  Metric                  Value
  ──────                  ─────
  Update                     73
  Mean Reward            -0.412
  Actor Loss             0.0183
  Critic Loss            0.2741
  Entropy                 1.073
  FPS                       104
──────────────────────────────────────────────────────────────
```

---

## Dashboard

```bash
python -m streamlit run src/dashboard.py
```

Live Streamlit interface showing:
- Real-time queue lengths and throughput per intersection
- Origin–destination flow heatmap
- Scenario selector and green wave toggle

---

## Evaluation

Run these from the **repository root** after training completes (or using the pre-trained checkpoints in `mappo_models/`).

### Single-intersection evaluation - MAPPO vs Fixed vs Random

```bash
python src/evaluate_mappo.py
```

Runs three controllers across all six traffic scenarios and records per-step metrics at every junction:

| Controller | Description |
|---|---|
| `mappo` | Trained policy (`mappo_models/best_actor.pth`), deterministic greedy |
| `fixed` | Fixed-time cycling - phases 0 → 1 → 2 → 0 every N steps |
| `random` | Uniform random phase selection (sanity baseline) |

**Output** (written to repo root and `output/`):
- `eval_summary.csv` - mean waiting time, queue length, throughput, stop ratio per controller per scenario
- `output/eval_comparison.png` - grouped bar chart across all scenarios

---

### Network-wide evaluation - MAPPO vs Fixed across all three junctions

```bash
python src/evaluate_network.py
```

Collects per-junction and network-wide metrics across all six demand scenarios.

**Output** (`output/`):
- `output/network_eval_summary.csv` - per-controller, per-scenario numbers for TL_A / TL_B / TL_C + network aggregate
- `output/network_eval_wait.png`, `network_eval_queue.png`, `network_eval_throughput.png`, `network_eval_pressure.png` - network-wide bar charts
- `output/network_eval_junctions_*.png` - per-junction breakdown

---

## Performance Analysis

All scripts in `Performance Analysis/` compare controllers head-to-head. Run them from the **repository root**:

### Bar-chart comparisons (static, all scenarios averaged)

```bash
# All 3 controllers - network-wide metrics
python "Performance Analysis/mappo_vs_ppo_vs_fixed_network.py"

# All 3 controllers - junction TL_A only
python "Performance Analysis/mappo_vs_ppo_vs_fixed_junction.py"
```

Compares **MAPPO**, **SA-PPO** (single-agent baseline from `SPPO_model/`), and **Fixed-time** on five metrics: waiting time, queue length, throughput, stop ratio, pressure.

**Output** (`output/`):
- `output/compare_summary.csv`, `output/compare_overall.png` - network-wide grouped bar charts
- `output/compare_tla_all_metrics.png` - TL_A junction bar chart

---

### Time-series comparisons (per-step line graphs)

```bash
# Line graphs at junction TL_A - MAPPO vs SA-PPO vs Fixed
python "Performance Analysis/mappo_vs_ppo_vs_fixed_junction_live.py"

# Line graphs across the full network - MAPPO vs Fixed
python "Performance Analysis/mappo_vs_fixed_network_live.py"
```

Records metrics at every simulation step and plots them as growing line charts (one line per controller).

**Output** (`output/`):
- `output/compare_tla_live.png` - per-step TL_A comparison
- `output/compare_network_waiting_time.png`, `compare_network_queue_length.png`, `compare_network_throughput.png`, `compare_network_stop_ratio.png`, `compare_network_pressure.png`

---

### Live dual-GUI comparison

```bash
# Opens two SUMO-GUI windows simultaneously: SA-PPO (left) vs MAPPO (right)
python "Performance Analysis/ppo_vs_mappo_live_gui.py"
```

Launches both simulations side by side and plots five metrics in real time as both step. Requires a display. Close the matplotlib window to stop both simulations cleanly.

---

### Metrics reference

| Metric | Definition |
|---|---|
| **Waiting time** | Mean seconds a vehicle spends stationary per incoming lane |
| **Queue length** | Mean halting vehicles per incoming lane |
| **Throughput** | Total vehicles that cleared the junction per episode |
| **Stop ratio** | Fraction of incoming vehicles stopped each step (mean) |
| **Pressure** | Mean (incoming vehicles − outgoing vehicles) per step |

---

## RL Design

| Component | Detail |
|---|---|
| **Algorithm** | MAPPO - Multi-Agent PPO with Centralised Critic |
| **Agents** | 3 (one per traffic light), parameter sharing |
| **Observation** | 22-dim local per agent; 66-dim global for critic |
| **Action** | Discrete(3) per agent - select green phase 0, 1, or 2 |
| **Reward** | Shared: `−tanh(avg_wait / 60s) − 0.1 × queue_ratio − collision/teleport penalties` |
| **Episode length** | 500 RL steps × 3 sim-seconds = 25 simulated minutes |
| **Min green hold** | 5 steps (15 sim-seconds) before a phase switch is allowed |
| **Advantage** | GAE (λ=0.95, γ=0.99) |
| **PPO clip** | ε = 0.2 |
| **Parallel envs** | 4 |

---

## Traffic Scenarios

| Scenario | Description | Direction bias |
|---|---|---|
| `low` | Off-peak, sparse traffic | Mixed |
| `normal` | Weekday mixed flow | Mixed |
| `rush_hour_am` | Heavy inbound morning commute | Inbound |
| `rush_hour_pm` | Heavy outbound evening commute | Outbound |
| `holiday` | Transit traffic through town | Transit |
| `incident` | Two entry edges blocked | Mixed |

---

## Results (single-agent PPO baseline → MAPPO)

The single-agent baseline (archived in [`SPPO_model/`](SPPO_model/), Stable Baselines3) treated all three traffic lights as one combined action space. MAPPO separates them into cooperating agents with a shared policy and a centralised critic, improving coordination at the cost of a more complex training setup.

Training logs are saved to `mappo_logs/mappo_train.csv`. Plot mean reward over timesteps to track convergence.

---

## Project Report

The full final year project report (LaTeX source) is in [`docs/`](docs/).

| File | Description |
|------|-------------|
| [`docs/main.tex`](docs/main.tex) | Main report document |
| [`docs/Simulation Report.tex`](docs/Simulation%20Report.tex) | Simulation analysis chapter |
| [`docs/references.bib`](docs/references.bib) | Bibliography |
| [`docs/figures/`](docs/figures/) | All report figures (resolved via `\graphicspath`) |
| [`docs/presentations/`](docs/presentations/) | Slide decks (.pptx) |

To compile locally: open `docs/main.tex` in [Overleaf](https://overleaf.com) (File → Upload Project → zip) or compile with `pdflatex` + `bibtex`.

---
