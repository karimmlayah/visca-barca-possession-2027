# VISCA BARCA — More Than A Hack 2027: Ball Possession Challenge

Official competition solution by **Team Visca Barca** for **More Than A Hack 2027 (Phase 1: Ball Possession)**, organized by [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com), and the [Mobile World Capital Barcelona](https://www.mwcbarcelona.com) at the Talent Arena.

---

## 1. Executive Summary & Problem Statement

In professional football analytics, ball possession is conventionally approximated via per-frame nearest-player proximity. While such naive heuristics yield plausible per-frame accuracies, they suffer from severe temporal fragmentation: passes in flight, contested duels, and temporary occlusions trigger spurious turnover cascades and artificial stoppages.

### The Phase 1 Task
Given raw automatic tracking, homography, and tactical event tagging from a full LaLiga match (**FC Barcelona vs. Athletic Club**, Matchday 2, 2024/25; 199,319 frames @ 25 fps, ~106 minutes of match play), the objective is to predict one discrete state for every single frame $t \in [0, 199318]$:
- `H`: Home team (FC Barcelona) in controlled possession (ball alive)
- `A`: Away team (Athletic Club) in controlled possession (ball alive)
- `D`: Dead ball (ball out of play, preparatory stoppages, whistled restarts)
- `X`: Ungraded frames (pre-match, halftime break, post-whistle)

Submissions are evaluated on **`spellF1`**—the mean F1 score across temporal Intersection-over-Union (IoU) thresholds $\tau \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$ over contiguous possession intervals—with per-frame **`macroF1`** as the primary tie-breaker.

---

## 2. System Architecture & Methodology

Our system replaces local heuristic smoothing with a principled, physically grounded **Hidden Markov Model (HMM) with dynamic programming Viterbi decoding** and **Semi-Markov duration filtering**.

```
                        ┌───────────────────────────────────────────────┐
                        │      Raw Automatic Tracking Data (ATD)        │
                        │    (22 Players + Ball Kinematics @ 25 fps)     │
                        └───────────────────────┬───────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
  ┌─────────────────────────────┐                               ┌─────────────────────────────┐
  │ Kinematic Control Potentials│                               │ Multi-Signal Dead-Ball (D)  │
  │ • Metric Euclidean Distances│                               │ • Pitch Boundary Excursions │
  │ • Velocity Attenuation      │                               │ • Boundary Disappearances   │
  │ • Control Envelope (R=2.5m) │                               │ • Smart Tagging Set-Pieces  │
  └──────────────┬──────────────┘                               │ • Collective Deceleration   │
                 │                                              └──────────────┬──────────────┘
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │       Multi-State Emission Potentials         │
                        │      Injected with Smart Tagging Priors       │
                        └───────────────────────┬───────────────────────┘
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │        Global HMM Viterbi Sequence            │
                        │   • Asymmetric Turnover Penalty (-15.0)       │
                        │   • Stoppage & Restart Transitions (-6.5/-5.0)│
                        │   • Missing-Ball Neutrality (Inertia)         │
                        └───────────────────────┬───────────────────────┘
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │        Semi-Markov Duration Filter            │
                        │   • Min Possession: 25 frames (1.0s)          │
                        │   • Min Stoppage: 35 frames (1.4s)            │
                        └───────────────────────┬───────────────────────┘
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │   Frame-Accurate Validated Submission CSV     │
                        │          (Exactly 199,319 Rows)               │
                        └───────────────────────────────────────────────┘
```

### A. Kinematic Control Envelope & Velocity Attenuation
For each frame $t$ with detected ball coordinates $(x_b, y_b)$, Euclidean distances to each team's nearest player in calibrated pitch space ($105.3\text{m} \times 68.0\text{m}$) are computed: $d_H(t) = \min_{i \in H} d_i(t)$ and $d_A(t) = \min_{j \in A} d_j(t)$.
- **Control Envelope:** Active control requires proximity within $R_{ctrl} = 2.5\text{ m}$.
- **Velocity Attenuation:** Fast-moving balls ($v_{ball} > 10\text{ m/s}$) correspond to passes, clearances, or shots in flight. Proximity evidence during high ball velocity is scaled down, preventing defending players from being falsely credited with possession while a pass travels past them.

### B. Possession Inertia & Asymmetric Turnover Penalty
Under association football rules, possession remains with the team that had it during loose balls, contested challenges, and deflections until the defending team establishes clear control. In our transition log-probability matrix:
- Direct turnovers ($H \to A$ or $A \to H$) carry a substantial penalty ($\Delta \ln P = -15.0$).
- This mathematically requires opposing players to demonstrate sustained control over consecutive frames before a possession change is triggered.

### C. Robust Missing-Ball Handling (41% of Graded Frames)
The tracking system loses the ball on 41% of graded frames due to aerial trajectories, scrums, and camera occlusions. Rather than falsely treating missing frames as stoppages (which fragments real possessions), our HMM treats missing observations as **information-neutral** (emission score = 0). Under neutral evidence, transition penalties naturally preserve the current possession spell across tracking dropouts.

### D. Multi-Modal Dead-Ball Detection (`D`)
Accurately detecting dead-ball intervals without human whistles requires fusing four physical and tactical indicators:
1. **Pitch Boundary Excursions:** Ball coordinates outside calibrated boundaries ($x < -0.5\text{m}, x > 105.8\text{m}, y < -0.5\text{m}, y > 68.5\text{m}$) inject definitive $+25.0$ dead-ball evidence.
2. **Boundary Disappearance Lead-ins:** 25.9% of ball dropouts occur within 1.5m of touchlines or goal lines. When the ball vanishes adjacent to boundaries, an out-of-play dead-ball prior is propagated across subsequent frames.
3. **SportsCode Smart Tagging Set-Pieces:** SportsCode XML marks 59 set pieces (corners, goal kicks, free kicks, throw-ins). Dead-ball priors ($+3.5$) are applied across the preparatory stoppage phase preceding each restart.
4. **Collective Kinetic Deceleration:** Mean player velocity drops sharply ($< 1.3\text{ m/s}$) during whistled stoppages, injuries, and substitutions, signaling dead-ball states.

### E. Global Viterbi Optimization & Semi-Markov Duration Filtering
The optimal state sequence is decoded globally via dynamic programming in $O(|\mathcal{S}|^2 T)$ time (< 3 seconds). A secondary Semi-Markov filter eliminates physiologically impossible micro-spells (possession spells $< 1.0\text{s}$ / 25 frames, stoppages $< 1.4\text{s}$ / 35 frames), ensuring high temporal IoU.

---

## 3. Public Evaluation Benchmark Results

The pipeline was evaluated using the official scoring harness (`evaluation/run_evaluation.py`) against the public 5-minute ground truth (7,500 graded frames, 17 true possessions, mean duration 14.2s).

> **Important Note:** In strict compliance with `RULES.md`, these public labels were used **exclusively as a diagnostic sanity check** and never for training or hyperparameter tuning. The official competition ranking is determined on the private full-match ground truth (158,857 graded frames).

| Submission / Algorithm | macroF1 | spellF1 | dead F1 | alive/dead F1 | transF1 (@25) | Latency | Possessions (True: 17) |
|---|---|---|---|---|---|---|---|
| `baseline_raw` (window 0) | 0.5972 | 0.0378 | 0.4197 | 0.8920 | 0.1410 | -3.1 f | 131 (2.0s) |
| `baseline_smooth25` | 0.6200 | 0.2868 | 0.3857 | 0.8974 | 0.2034 | +8.2 f | 36 (7.6s) |
| **`baseline_smooth50` (Official Floor)** | **0.6207** | **0.3256** | **0.3225** | **0.8990** | **0.0816** | **+4.0 f** | **26 (10.8s)** |
| `baseline_smooth125`* | 0.5812 | 0.4242* | 0.1464 | 0.8991 | 0.1667 | -7.7 f | 16 (18.5s) |
| **Visca Barca (Our Solution)** | **0.7173** | **0.3619** | **0.5848** | **0.9312** | **0.2800** | **+0.9 f** | **25 (10.7s)** |

*\*Note: As explained in `EVALUATION.md`, `baseline_smooth125` severely over-smooths across the full match (dropping to 0.2242 full-match spellF1, below `smooth50`'s 0.2433 floor).*

### Key Performance Highlights
- **`macroF1`:** Improved from 0.6207 to **0.7173** (+15.6% relative gain, #1 rank overall), driven by dead-ball precision (92.8%).
- **`spellF1`:** Improved from the 0.3256 baseline floor to **0.3619** (+11.1% gain).
- **Transition Detection F1 (@25):** Increased from 0.0816 to **0.2800** (+243% gain, 7 exact matched transitions vs 2).
- **Temporal Latency:** Near-zero bias at **+0.9 frames** (~36 ms), verifying sub-frame temporal alignment.

---

## 4. Installation & Requirements

The solution is implemented in pure Python with minimal scientific dependencies:

```bash
# 1. Clone repository
git clone https://github.com/karimmlayah/visca-barca-possession-2027.git
cd visca-barca-possession-2027

# 2. Create and activate a clean virtual environment
python -m venv venv
venv\Scripts\activate   # Windows PowerShell / CMD
# source venv/bin/activate # Linux / macOS

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 5. Dataset Directory Structure

Download the official competition data bundle and unpack it into `data/tactical/` (or point `MTAH_DATA_DIR` to your data path):

```
data/
└── tactical/
    ├── BAR-ATH_tactical.mp4               # Reference video (199,319 frames)
    ├── BAR-ATH_tactical_atd.txt           # Raw tracking data
    ├── BAR-ATH_tactical_atd.xml           # Tracking metadata (FIFA/EPTS)
    ├── BAR-ATH_tactical_homography.csv    # 3x3 homography matrices
    └── BAR-ATH_tactical_smart_tagging.xml # SportsCode tactical annotations
```

---

## 6. Reproducing the Predictions

To regenerate the exact competition submission CSV:

```bash
python run_solution.py
```

This single command:
1. Streams and extracts kinematic features from raw ATD (~8–10 seconds).
2. Parses Smart Tagging tactical context and set-piece stoppage priors.
3. Decodes the optimal state sequence via global Viterbi HMM.
4. Validates the submission format and writes the output file to `submission/Visca Barca.csv`.

To evaluate against the public ground truth sanity check:
```bash
python run_solution.py --evaluate
```

---

## 7. Competition Rules Compliance (`RULES.md`)

- **Strictly Algorithmic:** All predictions are produced automatically by deterministic code. Zero frames were manually labeled or adjusted.
- **Zero Public GT Tuning:** The 5-minute public ground truth was strictly held out as a post-hoc diagnostic tool and never used for training or tuning.
- **No External Data on This Match:** No human-produced event data (Opta, StatsBomb, Wyscout) or external tracking on this match was used.
- **Data Protection:** No match videos, tracking coordinates, or private intermediates are included in this repository. All match data remains gated on the user's local machine.

---

## 8. Limitations & Future Extensions

- **Player Identity Anonymity:** Tracking slots lack persistent player IDs; integrating OCR jersey number recognition from tactical video could model passing networks.
- **Multi-Camera Geometry:** Broadcast video is unaligned to tactical frames and homography is 34.2% invalid; pitch feature keypoint matching could leverage broadcast close-ups to resolve goalmouth occlusions.
