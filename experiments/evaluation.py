#!/usr/bin/env python
import os
import json
import math
import argparse
import random
from dataclasses import asdict
from collections import Counter, defaultdict

import numpy as np
import torch

from stable_baselines3 import PPO

from envs.harassment_env import HarassmentInterventionEnv, EnvConfig
from models.llm_wrappers import llm_harasser, llm_victim, llm_intervener
from models.harassment_koala import harassment_score
from models.text_encoder import text_encoder
from personas.personas import get_persona


# ---------------------------
# Utilities
# ---------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def severity_bin(s: float, safe_t: float, fail_t: float) -> str:
    if s <= safe_t:
        return "low"
    elif s < fail_t:
        return "medium"
    else:
        return "high"


def mean_ci95(values):
    """Normal-approx CI; fine for N>=30. For small N, increase episodes or use bootstrap."""
    vals = np.asarray(values, dtype=np.float64)
    m = float(np.mean(vals)) if len(vals) else float("nan")
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    se = sd / math.sqrt(len(vals)) if len(vals) > 0 else float("nan")
    ci = 1.96 * se if len(vals) > 1 else 0.0
    return m, ci


# ---------------------------
# Policy interfaces
# ---------------------------

class PolicyInterface:
    def act(self, obs, deterministic: bool = True) -> int:
        raise NotImplementedError


class PPOPolicy(PolicyInterface):
    def __init__(self, model: PPO):
        self.model = model

    def act(self, obs, deterministic: bool = True) -> int:
        a, _ = self.model.predict(obs, deterministic=deterministic)
        return int(a)


class AlwaysZeroPolicy(PolicyInterface):
    def act(self, obs, deterministic: bool = True) -> int:
        return 0


class RandomPolicy(PolicyInterface):
    def __init__(self, n_actions: int, rng_seed: int):
        self.n_actions = n_actions
        self.rng = np.random.RandomState(rng_seed)

    def act(self, obs, deterministic: bool = True) -> int:
        return int(self.rng.randint(self.n_actions))


# ---------------------------
# Evaluation core
# ---------------------------

def run_eval(
    policy: PolicyInterface,
    env_config: EnvConfig,
    episodes: int,
    base_seed: int,
    deterministic: bool,
):
    """
    Runs evaluation episodes and returns:
      - per-episode metric arrays
      - action counters (global and by severity bin)
      - raw trajectories (optional; here we keep minimal info)
    """
    # Build a fresh env per episode to keep seeds clean and avoid state carryover.
    safe_t = env_config.safe_threshold
    fail_t = env_config.fail_threshold

    action_counts = Counter()
    action_counts_by_bin = defaultdict(Counter)

    # Episode-level metrics
    deltas = []
    finals = []
    maxes = []
    aucs = []
    steps_above_fail = []
    steps_above_safe = []
    lengths = []

    for ep in range(episodes):
        ep_seed = base_seed + ep
        set_all_seeds(ep_seed)

        env = HarassmentInterventionEnv(
            llm_harasser=llm_harasser,
            llm_victim=llm_victim,
            llm_intervener=llm_intervener,
            get_persona=get_persona,
            harassment_scorer=harassment_score,
            text_encoder=text_encoder,
            config=env_config,
            seed=ep_seed,
        )

        obs, info = env.reset(seed=ep_seed)

        scores = []
        done = False
        truncated = False

        while not (done or truncated):
            # Use current score for binning (pre-action)
            cur_score = float(info.get("harassment_score", env.harassment_score))
            b = severity_bin(cur_score, safe_t, fail_t)

            a = policy.act(obs, deterministic=deterministic)
            action_counts[a] += 1
            action_counts_by_bin[b][a] += 1

            obs, r, done, truncated, info = env.step(a)
            scores.append(float(info.get("harassment_score", env.harassment_score)))

        if len(scores) == 0:
            # Should not happen, but guard anyway.
            continue

        s0 = scores[0]
        sf = scores[-1]
        mx = max(scores)
        auc = float(np.sum(scores))
        above_fail = int(np.sum(np.asarray(scores) >= fail_t))
        above_safe = int(np.sum(np.asarray(scores) >= safe_t))

        deltas.append(s0 - sf)
        finals.append(sf)
        maxes.append(mx)
        aucs.append(auc)
        steps_above_fail.append(above_fail)
        steps_above_safe.append(above_safe)
        lengths.append(len(scores))

        env.close()

    metrics = {
        "delta_severity": deltas,
        "final_severity": finals,
        "max_severity": maxes,
        "auc": aucs,
        "steps_above_fail": steps_above_fail,
        "steps_above_safe": steps_above_safe,
        "episode_len": lengths,
    }

    return metrics, action_counts, action_counts_by_bin


def summarize_metrics(metrics: dict):
    out = {}
    for k, vals in metrics.items():
        m, ci = mean_ci95(vals)
        out[k] = {"mean": m, "ci95": ci, "n": len(vals)}
    return out


# ---------------------------
# Main: evaluate checkpoints 1k..5k (or any list)
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, default="./experiments/checkpoints",
                    help="Directory containing SB3 checkpoint zips.")
    ap.add_argument("--checkpoints", type=str, default="1000,2000,3000,4000,5000",
                    help="Comma-separated checkpoint step counts (expects ppo_harassment_{N}_steps.zip).")
    ap.add_argument("--episodes", type=int, default=50, help="Evaluation episodes per checkpoint per policy.")
    ap.add_argument("--base_seed", type=int, default=1234, help="Base seed for eval episodes.")
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic actions for PPO eval.")
    ap.add_argument("--out_dir", type=str, default="./experiments/eval_reports",
                    help="Where to write CSV/JSON reports.")
    # Env config overrides (must match training config for fair comparisons)
    ap.add_argument("--max_steps_per_episode", type=int, default=8)
    ap.add_argument("--history_window", type=int, default=-1)
    ap.add_argument("--safe_threshold", type=float, default=0.004)
    ap.add_argument("--fail_threshold", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    history_window = None if args.history_window < 0 else args.history_window

    env_config = EnvConfig(
        max_steps=args.max_steps_per_episode,
        history_window=history_window,
        safe_threshold=args.safe_threshold,
        fail_threshold=args.fail_threshold,
        # keep other reward params from your EnvConfig defaults
    )

    ckpt_steps = [int(x.strip()) for x in args.checkpoints.split(",") if x.strip()]
    rows = []

    # Policies to compare
    baseline_policies = {
        "always_0": AlwaysZeroPolicy(),
        "random": RandomPolicy(n_actions=4, rng_seed=args.base_seed + 999),
    }

    for n in ckpt_steps:
        ckpt_name = f"ppo_harassment_{n}_steps.zip"
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Missing checkpoint: {ckpt_path} (skipping)")
            continue

        # Load PPO checkpoint
        model = PPO.load(ckpt_path, device="cpu")  # eval on cpu is fine; env is the bottleneck
        ppo_policy = PPOPolicy(model)

        eval_policies = {"ppo": ppo_policy, **baseline_policies}

        # Evaluate each policy
        for pol_name, pol in eval_policies.items():
            metrics, act_counts, act_counts_by_bin = run_eval(
                policy=pol,
                env_config=env_config,
                episodes=args.episodes,
                base_seed=args.base_seed,
                deterministic=args.deterministic if pol_name == "ppo" else True,
            )
            summ = summarize_metrics(metrics)

            # Save action histograms (JSON) per checkpoint+policy
            histo_path = os.path.join(args.out_dir, f"actions_{pol_name}_{n}.json")
            with open(histo_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "checkpoint_steps": n,
                        "policy": pol_name,
                        "global_action_counts": dict(act_counts),
                        "action_counts_by_bin": {k: dict(v) for k, v in act_counts_by_bin.items()},
                        "env_config": asdict(env_config),
                        "episodes": args.episodes,
                        "base_seed": args.base_seed,
                    },
                    f,
                    indent=2,
                )

            # Add one summary row per policy per checkpoint
            row = {
                "checkpoint_steps": n,
                "policy": pol_name,
                "episodes": args.episodes,
                "base_seed": args.base_seed,
                "safe_threshold": env_config.safe_threshold,
                "fail_threshold": env_config.fail_threshold,
            }
            for k, v in summ.items():
                row[f"{k}_mean"] = v["mean"]
                row[f"{k}_ci95"] = v["ci95"]
                row[f"{k}_n"] = v["n"]
            # include global action proportions
            total_actions = sum(act_counts.values()) if len(act_counts) else 0
            for a in range(4):
                row[f"action_{a}_pct"] = (act_counts.get(a, 0) / total_actions) if total_actions > 0 else 0.0
            rows.append(row)

            print(f"[OK] ckpt={n} policy={pol_name} "
                  f"Δ={row['delta_severity_mean']:.4f} AUC={row['auc_mean']:.4f} "
                  f"failSteps={row['steps_above_fail_mean']:.2f} "
                  f"acts={dict(act_counts)}")

    # Write CSV summary
    import csv
    csv_path = os.path.join(args.out_dir, "checkpoint_eval_summary.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Also write a compact JSON for convenience
    json_path = os.path.join(args.out_dir, "checkpoint_eval_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nWrote:\n  {csv_path}\n  {json_path}\n  (and per-checkpoint action JSONs in {args.out_dir})")


if __name__ == "__main__":
    main()
