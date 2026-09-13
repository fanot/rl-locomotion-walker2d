"""PPO for MuJoCo locomotion (Walker2d / Hopper / Ant / HalfCheetah).

Single-file implementation:
  * Gaussian policy with state-independent log-std, orthogonal init
  * GAE(lambda), clipped surrogate objective, clipped value loss, entropy bonus
  * observation / reward normalisation, action clipping, linear LR annealing
  * per-episode logging of the raw reward *components* (forward / ctrl / healthy)
    and of the distance travelled, so a reward ablation can be judged on a
    metric that does not itself change when the reward weights change.

Usage:
    python ppo.py --env-id Walker2d-v5 --total-timesteps 1000000 --seed 1
    python ppo.py --ctrl-cost-weight 0.0 --run-name ctrl0_s1
"""

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass, asdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class Args:
    env_id: str = "Walker2d-v5"
    seed: int = 1
    total_timesteps: int = 1_000_000
    num_envs: int = 4
    num_steps: int = 512             # per env, per rollout -> batch = 2048
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03          # early-stop an update if KL blows up

    # reward-shaping knobs -- this is what the ablation sweeps
    ctrl_cost_weight: float = 0.001
    forward_reward_weight: float = 1.0
    healthy_reward: float = 1.0

    run_name: str = ""
    out_dir: str = "runs"
    device: str = "cpu"


def parse_args() -> Args:
    defaults = asdict(Args())
    p = argparse.ArgumentParser()
    for key, value in defaults.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            p.add_argument(flag, type=lambda s: str(s).lower() in ("1", "true", "yes"), default=value)
        else:
            p.add_argument(flag, type=type(value), default=value)
    return Args(**vars(p.parse_args()))


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
REWARD_KWARGS = {
    "Walker2d": ("ctrl_cost_weight", "forward_reward_weight", "healthy_reward"),
    "Hopper": ("ctrl_cost_weight", "forward_reward_weight", "healthy_reward"),
    "Ant": ("ctrl_cost_weight", "forward_reward_weight", "healthy_reward"),
    "HalfCheetah": ("ctrl_cost_weight", "forward_reward_weight"),
}


def env_kwargs_for(args: Args) -> dict:
    keys = REWARD_KWARGS.get(args.env_id.split("-")[0], ())
    return {k: getattr(args, k) for k in keys}


def make_env(args: Args, idx: int):
    def thunk():
        env = gym.make(args.env_id, **env_kwargs_for(args))
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda o: np.clip(o, -10.0, 10.0), env.observation_space
        )
        env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
        env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10.0, 10.0)))
        env.reset(seed=args.seed + idx)
        env.action_space.seed(args.seed + idx)
        return env

    return thunk


# --------------------------------------------------------------------------- #
# agent
# --------------------------------------------------------------------------- #
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def act(self, x, action=None, deterministic=False):
        mean = self.actor_mean(x)
        if deterministic:
            return mean
        std = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.value(x)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    run_name = args.run_name or f"{args.env_id}__ctrl{args.ctrl_cost_weight}__s{args.seed}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    envs = gym.vector.SyncVectorEnv([make_env(args, i) for i in range(args.num_envs)])
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = max(args.total_timesteps // batch_size, 1)

    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    act_buf = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    logp_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rew_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    val_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    csv_file = open(os.path.join(run_dir, "progress.csv"), "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["global_step", "episodic_return", "episodic_length", "forward_reward",
                     "ctrl_cost", "healthy_reward", "x_distance", "action_energy"])

    # per-env accumulators for the raw reward components, reset at episode end
    comp = {k: np.zeros(args.num_envs) for k in ("fwd", "ctrl", "healthy", "energy")}

    global_step = 0
    start = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(np.asarray(next_obs), dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    recent_returns: list[float] = []

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (update - 1.0) / num_updates) * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logp, _, value = agent.act(next_obs)
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value

            np_action = action.cpu().numpy()
            obs_np, reward, terminated, truncated, infos = envs.step(np_action)
            done = np.logical_or(terminated, truncated)

            rew_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(np.asarray(obs_np), dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done.astype(np.float32), device=device)

            # ---- raw component accounting (independent of reward weights) ----
            comp["energy"] += np.square(np.clip(np_action, -1.0, 1.0)).sum(axis=1)
            for key, name in (("reward_forward", "fwd"), ("reward_ctrl", "ctrl"),
                              ("reward_survive", "healthy")):
                if key in infos:
                    vals = np.nan_to_num(np.asarray(infos[key], dtype=np.float64))
                    mask = np.asarray(infos.get("_" + key, np.ones(args.num_envs, dtype=bool)))
                    comp[name] += np.where(mask, vals, 0.0)

            if "episode" in infos:
                ep_mask = np.asarray(infos["_episode"])
                ep_r = np.asarray(infos["episode"]["r"], dtype=np.float64)
                ep_l = np.asarray(infos["episode"]["l"], dtype=np.float64)
                x_pos = np.asarray(infos.get("x_position", np.zeros(args.num_envs)), dtype=np.float64)
                for i in np.nonzero(ep_mask)[0]:
                    writer.writerow([global_step, f"{ep_r[i]:.3f}", int(ep_l[i]),
                                     f"{comp['fwd'][i]:.3f}", f"{comp['ctrl'][i]:.4f}",
                                     f"{comp['healthy'][i]:.3f}", f"{x_pos[i]:.3f}",
                                     f"{comp['energy'][i] / max(ep_l[i], 1.0):.4f}"])
                    recent_returns.append(float(ep_r[i]))
                    for k in comp:
                        comp[k][i] = 0.0
                csv_file.flush()

        # ------------------------------ GAE -------------------------------- #
        with torch.no_grad():
            next_value = agent.value(next_obs)
            advantages = torch.zeros_like(rew_buf)
            lastgaelam = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nextvalues * nextnonterminal - val_buf[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + val_buf

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        inds = np.arange(batch_size)
        approx_kl = torch.tensor(0.0)
        for _ in range(args.update_epochs):
            np.random.shuffle(inds)
            for start_i in range(0, batch_size, minibatch_size):
                mb = inds[start_i:start_i + minibatch_size]
                _, newlogp, entropy, newvalue = agent.act(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                if args.clip_vloss:
                    v_unclipped = (newvalue - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + torch.clamp(newvalue - b_val[mb],
                                                        -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_ret[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_ret[mb]) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl.item() > args.target_kl:
                break

        if update % 5 == 0 or update == num_updates:
            avg = float(np.mean(recent_returns[-20:])) if recent_returns else float("nan")
            sps = int(global_step / (time.time() - start))
            print(f"update {update}/{num_updates}  step {global_step}  "
                  f"return(20ep) {avg:8.1f}  kl {approx_kl.item():.4f}  {sps} steps/s", flush=True)

    # save policy + observation normalisation stats (both are needed at eval time)
    norm_env = envs.envs[0]
    while not isinstance(norm_env, gym.wrappers.NormalizeObservation):
        norm_env = norm_env.env
    torch.save({"model": agent.state_dict(),
                "obs_mean": norm_env.obs_rms.mean,
                "obs_var": norm_env.obs_rms.var,
                "args": asdict(args)},
               os.path.join(run_dir, "checkpoint.pt"))
    csv_file.close()
    envs.close()
    print(f"saved -> {run_dir}/checkpoint.pt")


if __name__ == "__main__":
    main()
