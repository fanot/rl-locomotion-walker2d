"""Evaluate a trained checkpoint and optionally record a video of the gait.

    python eval.py runs/ctrl_default_s1/checkpoint.pt --episodes 10
    python eval.py runs/ctrl_default_s1/checkpoint.pt --video videos/

Headless machines need a MuJoCo offscreen backend for --video:
    MUJOCO_GL=egl    python eval.py ... --video videos/     # GPU
    MUJOCO_GL=osmesa python eval.py ... --video videos/     # CPU, needs libosmesa6
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch

from ppo import Agent, REWARD_KWARGS


def build_env(ckpt_args: dict, render: bool, video_dir: str | None):
    keys = REWARD_KWARGS.get(ckpt_args["env_id"].split("-")[0], ())
    kwargs = {k: ckpt_args[k] for k in keys}
    env = gym.make(ckpt_args["env_id"], render_mode="rgb_array" if render else None, **kwargs)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda i: True,
                                       name_prefix="gait")
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--video", default=None, help="directory for mp4 output")
    p.add_argument("--stochastic", action="store_true", help="sample instead of using the mean")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    env = build_env(ckpt["args"], render=args.video is not None, video_dir=args.video)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    agent = Agent(obs_dim, act_dim)
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    mean, var = ckpt["obs_mean"], ckpt["obs_var"]
    normalise = lambda o: np.clip((o - mean) / np.sqrt(var + 1e-8), -10.0, 10.0)

    returns, lengths, distances, energies = [], [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, total, steps, energy = False, 0.0, 0, 0.0
        x_final = 0.0
        while not done:
            with torch.no_grad():
                x = torch.as_tensor(normalise(obs), dtype=torch.float32).unsqueeze(0)
                if args.stochastic:
                    action = agent.act(x)[0]
                else:
                    action = agent.act(x, deterministic=True)
            a = np.clip(action.numpy()[0], env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(a)
            total += float(reward)
            energy += float(np.square(a).sum())
            x_final = float(info.get("x_position", x_final))
            steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)
        distances.append(x_final)
        energies.append(energy / max(steps, 1))
        print(f"episode {ep:2d}  return {total:8.1f}  length {steps:4d}  x {x_final:6.2f}")

    env.close()
    print("\n--- summary over", args.episodes, "episodes ---")
    print(f"return        {np.mean(returns):8.1f} ± {np.std(returns):.1f}")
    print(f"length        {np.mean(lengths):8.1f} ± {np.std(lengths):.1f}")
    print(f"x distance    {np.mean(distances):8.2f} ± {np.std(distances):.2f}")
    print(f"action energy {np.mean(energies):8.3f} ± {np.std(energies):.3f}   (mean sum a^2 per step)")
    if args.video:
        print(f"video written to {args.video}/")


if __name__ == "__main__":
    main()
