# Tetris-hf

LINK : https://huggingface.co/spaces/Lishika/tetris-final

Tetris RL — LLM Long-Horizon Planning via OpenEnv
Teaching language models to think ahead by playing Tetris. A turn-based Tetris environment built on the OpenEnv framework where an LLM agent must learn to stack pieces, clear lines, and manage long-term board health — skills that require genuine multi-step planning, not pattern matching.

Links
Resource	URL
🚀 Live Dashboard & API	huggingface.co/spaces/Lishika/tetris-final
📓 Training Notebook (Colab)	GRPO Training Notebook
📊 Training Metrics (in-app)	Open the METRICS tab on the live dashboard
🎮 Replay Viewer (in-app)	Open the REPLAY tab — pre-loaded with seed replays across training stages
🎥 Demo Video	(coming soon)
Judges: all external materials are linked above. The live Space hosts an interactive dashboard with replay visualization and training metrics charts, plus a full REST API — try the Quick Start examples below.

What to Look At (Judge Quick Guide)
Open the Live Dashboard — it loads with the best replay (EXPERT level, step 100) auto-playing.
REPLAY tab — scroll the left panel to compare training stages: UNTRAINED → LEARNING → SKILLED → EXPERT. Watch how the agent's play visibly improves.
METRICS tab — view GRPO training curves (mean reward, JSON-valid rate, lines cleared, turns survived) plotted over training steps.
Training Notebook — open in Colab to see the full GRPO training loop and logs.
API — try the curl examples below to interact with the environment programmatically.
The Problem
Large language models excel at short-horizon text tasks but struggle with long-horizon planning — sequences of 50–100+ dependent decisions where early mistakes compound silently and only surface many steps later. This is the gap between "write a function" and "architect a system."

Why Tetris? Every piece placement is cheap to evaluate locally, but its consequences propagate across dozens of future turns. A hole buried on turn 5 may not matter until turn 40, when it blocks the only possible line clear. This makes Tetris a near-ideal microbenchmark for planning depth:

Combinatorial action space: 4 actions × 10 columns × 4 rotations per piece, but only a handful of placements lead to clean boards.
Delayed consequences: a bad placement today creates holes that compound over 20–30 turns before causing game over.
Measurable improvement: lines cleared and survival turns are unambiguous metrics that directly reflect planning quality.
A researcher could write a paper training LLMs on this environment because it isolates long-horizon credit assignment in a fully observable, deterministic setting — the same core challenge behind multi-step reasoning, code generation, and agentic tool use.

The Environment
Property	Value
Board	20 rows × 10 columns, pure Python (no numpy)
Pieces	All 7 standard tetrominoes with SRS rotation
Actions	move_left, move_right, rotate, place
Observation	Board grid + current piece + position + next piece + score/lines
Episode ends	Board overflow (any piece cell reaches row 0)
Framework	OpenEnv — manifest at /manifest
Reward — 5 Independent Signals
The reward is decomposed into 5 independent components to prevent reward hacking. No single component can be gamed without hurting others:

Component	Value	Purpose
Lines	+10 per cleared line	Primary objective — row completion
Survival	+1 per non-terminal placement	Stay alive, avoid suicidal play
Holes	−1 per net new hole created	Penalizes inaccessible gaps
Height	−0.5 per row above height 10	Discourages passive stack growth
Game Over	−10 on terminal placement	Terminal penalty for overflow
Why 5 signals? Clearing lines requires flat stacking. Flat stacking requires managing holes. Managing holes requires height control. The interdependence creates a genuine long-horizon planning signal — the agent cannot succeed by optimizing any single component in isolation.

Training — Evidence of Improvement
We trained a Qwen 2.5-1.5B model using GRPO (Generative Reinforcement Learning with Proximal Optimization) via HuggingFace TRL, with optional Unsloth for efficient LoRA fine-tuning. The training notebook is fully reproducible in Google Colab.

Metrics Table
Metric	Step 0 (Random)	Step 50 (Mid)	Step 100 (Trained)
Avg turns survived	~10	~25	~55
Avg lines cleared	0	1–3	8–15
Avg total reward	~0	~15	~60
The progression tells a clear story: at step 0, the random agent drops pieces chaotically — average 10 turns, zero lines cleared. By step 50, the agent learns basic stacking — it survives longer and begins clearing occasional lines. By step 100, the agent stacks flat, prioritizes line clears, survives 55+ turns, and clears 8–15 lines per episode. This 6× improvement in survival and the jump from 0 to 8+ line clears demonstrates genuine long-horizon planning capability.

Reward Curves
Episode reward across training steps. Random agent baseline (step 0) vs. trained agent (step 100). Each point is averaged over 5 episodes.

See the Training Colab for full reproduction, or view the METRICS tab on the live dashboard for training curves.

API Reference
The live Space exposes a full REST API. Judges and researchers can interact with the environment directly.

Method	Path	Description
GET	/health	Liveness probe — {"status": "ok"}
GET	/manifest	OpenEnv manifest (parsed openenv.yaml)
POST	/reset	Start/reset session, returns session_id + observation
POST	/step	Apply action: {"session_id": "...", "action": "place"}
GET	/state?session_id=...	Current board observation for a session
POST	/reset_all	Clear all active sessions
POST	/run_episode	Run full random-agent episode, saves replay JSON
GET	/episodes	List all saved replay metadata
GET	/episodes/{filename}	Full JSON of a specific replay
GET	/training-metrics	GRPO curves from checkpoints/metrics.json (or TRAINING_METRICS_PATH)
Quick Start
OpenEnv Client
from openenv import Environment

env = Environment.from_hub("Lishika/tetris-final")
obs = env.reset()
obs, reward, done, info = env.step("rotate")
obs, reward, done, info = env.step("place")
print(f"Reward: {reward}, Done: {done}")
Direct HTTP (curl)
# Start a new episode (returns session_id)
curl -X POST https://Lishika-tetris-final.hf.space/reset
# => {"session_id": "abc-123-...", "observation": {...}}

# Place a piece (pass session_id from /reset)
curl -X POST https://Lishika-tetris-final.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123-...", "action": "place"}'

# Check environment manifest
curl https://Lishika-tetris-final.hf.space/manifest
Python Client
from server.client import TetrisClient

c = TetrisClient("http://localhost:8000")
obs = c.reset()
obs, reward, done, info = c.step("rotate")
Reward Design Rationale
The reward is decomposed into 5 independent signals to prevent reward hacking:

Lines (+10/row): Primary objective — incentivizes row completion.
Survival (+1/placement): Keeps the agent alive; prevents suicidal play.
Holes (−1/hole): Penalizes inaccessible space; compounds over time.
Height (−0.5/row above 10): Discourages passive stack growth.
Game over (−10): Terminal penalty for board overflow.
No single component can be gamed independently — clearing lines requires managing holes, which requires height control. The delayed consequence of poor early placements is the core long-horizon planning signal. This is what makes the environment suitable for studying credit assignment in LLMs.

Running Locally
# Install server deps (no GPU needed)
pip install -r requirements-serve.txt

# Generate seed replays for dashboard demo
python scripts/seed_replays.py

# Start environment server
uvicorn server.app:app --port 8000

# (Optional) Run full training — requires GPU
pip install -r requirements.txt
python training/grpo_train.py
Project Structure
tetris-openenv/
├── environment/          # Pure-Python game core (no numpy, no rendering)
│   ├── tetris_env.py     # TetrisEnv: reset(), step(), get_observation()
│   ├── board.py          # 20×10 board with line clearing
│   ├── tetromino.py      # 7 pieces, SRS rotation system
│   ├── reward.py         # 5-component reward function
│   └── episode_logger.py # JSON episode serialization
├── server/               # FastAPI server + OpenEnv endpoints
├── training/             # GRPO training scaffold + Colab notebook
├── dashboard/            # React/Vite replay viewer + metrics charts
├── replays/              # Seed replay JSON files (committed)
├── openenv.yaml          # OpenEnv manifest
├── Dockerfile            # HF Spaces Docker image (server only)
└── requirements.txt      # Full deps including training
Citation
@misc{tetris-openenv-2026,
  title   = {Tetris OpenEnv: A Long-Horizon Planning Environment for LLM RL Training},
  author  = {openenv-hackathon-team},
  year    = {2026},
  url     = {https://huggingface.co/spaces/Lishika/tetris-final}
}
