# PPO for MuJoCo locomotion — with a reward ablation

A compact, from-scratch PPO implementation for continuous-control locomotion
(`Walker2d-v5` by default), plus a controlled study of how the **control-cost
term in the reward** changes the learned gait.

The point of the repo is not to beat a benchmark — PPO on Walker2d is a solved
problem — but to have a clean, reproducible setup for asking *how the reward
shapes the policy*, which is the part that transfers to real robots.

```
ppo.py            single-file PPO (policy, GAE, update loop, CSV logging)
run_ablation.sh   sweeps ctrl_cost_weight x seeds
plot.py           learning curves, seed bands, final-performance table
eval.py           deterministic evaluation + mp4 of the gait
```

## Quickstart

```bash
pip install -r requirements.txt

python ppo.py --env-id Walker2d-v5 --total-timesteps 1000000 --seed 1
python plot.py --metric episodic_return
python eval.py runs/Walker2d-v5__ctrl0.001__s1/checkpoint.pt --episodes 10 --video videos/
```

Recording video on a headless machine needs a MuJoCo offscreen backend:
`MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU, needs `libosmesa6`).

## Implementation

Standard PPO with the details that actually matter for MuJoCo:

- Gaussian policy, state-independent `log_std`, orthogonal init (`std=0.01` on
  the policy head so the initial policy is near-deterministic)
- separate 64-64 tanh MLPs for actor and critic
- GAE(λ=0.95), γ=0.99, advantage normalisation per minibatch
- clipped surrogate objective (ε=0.2), clipped value loss, gradient clipping
- running observation normalisation and return-scaled reward normalisation;
  the normalisation statistics are saved with the checkpoint, because a policy
  evaluated without them is a different policy
- linear LR annealing, KL-based early stop of an update (`target_kl=0.03`)
- 4 parallel envs × 512 steps = 2048-step batch, 32 minibatches, 10 epochs

Episode statistics are recorded *before* reward normalisation, so the logged
returns are raw environment returns.

## The ablation

`Walker2d-v5` reward = `forward_reward + healthy_reward − ctrl_cost`, where
`ctrl_cost = ctrl_cost_weight · Σaᵢ²`. The sweep:

| config         | `ctrl_cost_weight` |
|----------------|--------------------|
| `ctrl0`        | 0.0                |
| `ctrl_default` | 0.001 (Gymnasium default) |
| `ctrl_high`    | 0.01               |

```bash
bash run_ablation.sh          # 3 configs × 2 seeds × 1M steps
```

**Reading the result correctly.** Changing a reward weight changes the
objective, so the returns of the three configs are not on a common scale — a
higher return under `ctrl0` does not mean a better policy. Every run therefore
also logs two quantities that do not depend on the reward weights:

- `x_distance` — how far the walker actually travelled
- `action_energy` — mean `Σaᵢ²` per step, i.e. how much actuation it spent

The interesting question is the trade-off curve between them: how much distance
a heavier control penalty costs, and how much actuation it saves.

## Results

Fill in after running the sweep (last 10% of episodes, mean ± spread over seeds):

| config         | return | x distance | action energy |
|----------------|--------|------------|---------------|
| `ctrl0`        |        |            |               |
| `ctrl_default` |        |            |               |
| `ctrl_high`    |        |            |               |

`videos/` holds one episode per config for a visual comparison of the gaits.

## Reproducibility

- Python 3.12, `torch` 2.14, `gymnasium` 1.3, `mujoco` 3.13
- CPU-only run measured at ≈630 env-steps/s with 4 parallel envs, so ~26 min
  per 1M-step run and ~2.5 h for the full 6-run sweep
- every run writes `runs/<name>/progress.csv` (per-episode records) and
  `checkpoint.pt` (weights + observation statistics + the exact config)
- seeds fixed for Python, NumPy, Torch and the env / action spaces

## Limitations and next steps

- Walker2d is 2D and contact-simplified; nothing here addresses sim-to-real
  transfer, which is where locomotion actually gets hard
- no domain randomisation, no observation delay or actuator model
- the natural extension, and the reason I built this: learn a latent dynamics
  model with physical constraints and use it for model-based rollouts, then
  compare sample efficiency against this model-free baseline
