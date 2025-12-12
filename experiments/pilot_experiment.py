#!/usr/bin/env python
"""
Main RL experiment driver for harassment intervention.

- Uses HarassmentInterventionEnv
- Harasser, victim, intervener = Mistral-7B
- Harassment scorer = KoalaAI/Text-Moderation
- Text encoder = sentence-transformers/all-MiniLM-L6-v2
- Intervener policy = PPO (Stable-Baselines3)
"""

import os
import json
import time
import argparse
import random

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from envs.harassment_env import HarassmentInterventionEnv, EnvConfig
from models.llm_wrappers import llm_harasser, llm_victim, llm_intervener
from models.harassment_koala import harassment_score
from models.text_encoder import text_encoder
from personas.personas import get_persona



# ---------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------

def make_single_env(env_config: EnvConfig, seed: int):
    """
    Returns a function that creates a single HarassmentInterventionEnv
    with given config and seed. Suitable for DummyVecEnv/SubprocVecEnv.
    """
    def _init():
        env = HarassmentInterventionEnv(
            llm_harasser=llm_harasser,
            llm_victim=llm_victim,
            llm_intervener=llm_intervener,
            get_persona=get_persona,
            harassment_scorer=harassment_score,
            text_encoder=text_encoder,
            config=env_config,
            seed=seed,
        )
        return env
    return _init


def make_vec_env(
    env_config: EnvConfig,
    num_envs: int,
    base_seed: int,
    use_subproc: bool = False,
):
    """
    Create a vectorized environment (DummyVecEnv or SubprocVecEnv).
    """
    env_fns = [
        make_single_env(env_config, seed=base_seed + i)
        for i in range(num_envs)
    ]
    if use_subproc and num_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)

# ---------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------

class StopOnFileCallback(BaseCallback):
    """
    Stop training early if a given file exists.
    This lets you create the file while the job is running
    to request a clean stop.
    """
    def __init__(self, stop_file: str, verbose: int = 0):
        super().__init__(verbose)
        self.stop_file = stop_file

    def _on_step(self) -> bool:
        if os.path.exists(self.stop_file):
            if self.verbose > 0:
                print(f"[StopOnFileCallback] Stop file found at {self.stop_file}, stopping training.")
            # Returning False stops training
            return False
        return True



# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def train_ppo(
    total_timesteps: int,
    num_envs: int,
    max_steps_per_episode: int,
    history_window: int,
    learning_rate: float,
    batch_size: int,
    n_steps: int,
    gamma: float,
    ent_coef: float,
    tensorboard_log: str,
    output_dir: str,
    use_subproc: bool,
    seed: int,
    checkpoint_freq: int,
    checkpoint_dir: str,
    stop_file: str,
):
    """
    Train PPO on the harassment intervention environment.
    """

    # Env config
    env_config = EnvConfig(
        max_steps=max_steps_per_episode,
        history_window=history_window,
    )

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Vec env
    vec_env = make_vec_env(
        env_config=env_config,
        num_envs=num_envs,
        base_seed=seed,
        use_subproc=use_subproc,
    )

    # PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        seed=seed,
    )

    # --- Callbacks: checkpointing + stop-on-file ------------------------
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save_freq is in environment steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, checkpoint_freq // max(1, num_envs)),
        save_path=checkpoint_dir,
        name_prefix="ppo_harassment",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    stop_callback = StopOnFileCallback(stop_file=stop_file, verbose=1)

    callback_list = [checkpoint_callback, stop_callback]

    
    # Train
    model.learn(total_timesteps=total_timesteps, callback=callback_list)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "ppo_harassment_intervener.zip")
    model.save(model_path)

    return model_path
    
def evaluate_policy(
    model_path: str,
    num_episodes: int,
    max_steps_per_episode: int,
    history_window: int,
    output_dir: str,
    seed: int,
):
    """
    Run evaluation episodes with a trained PPO policy and save FULL trajectories.

    For each episode, we log:
      - harasser_traits
      - victim_traits
      - harassment_goal
      - steps: list of {
            step,
            action,
            reward,
            harassment_score,
            max_harassment,
            history,
            intervention_log
        }
    """

    from stable_baselines3 import PPO  # local import to avoid circular issues

    # Reload model
    model = PPO.load(model_path)

    # Single env for evaluation
    env_config = EnvConfig(
        max_steps=max_steps_per_episode,
        history_window=history_window,
    )
    eval_env = HarassmentInterventionEnv(
        llm_harasser=llm_harasser,
        llm_victim=llm_victim,
        llm_intervener=llm_intervener,
        get_persona=get_persona,
        harassment_scorer=harassment_score,
        text_encoder=text_encoder,
        config=env_config,
        seed=seed,
    )

    trajectories = []

    for ep in range(num_episodes):
        obs, info = eval_env.reset()

        episode_record = {
            "episode_index": ep,
            "harasser_traits": info.get("harasser_traits", {}),
            "victim_traits": info.get("victim_traits", {}),
            "harassment_goal": info.get("harassment_goal", None),
            "steps": [],  # filled below
        }

        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(int(action))

            step_record = {
                "step": info.get("step", eval_env.step_count),
                "action": int(action),
                "reward": float(reward),
                "harassment_score": float(info.get("harassment_score", eval_env.harassment_score)),
                "max_harassment": float(info.get("max_harassment", eval_env.max_harassment)),
                "history": info.get("history", list(eval_env.history)),
                "intervention_log": info.get("intervention_log", []),
            }

            # (Optional) also log observation vector if you want:
            # step_record["obs_vector"] = obs.tolist()

            episode_record["steps"].append(step_record)

        trajectories.append(episode_record)

    # Save as JSONL
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "eval_trajectories.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj) + "\n")

    return jsonl_path



# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RL experiment for harassment intervention with multi-agent LLM simulation."
    )

    # Training hyperparams
    parser.add_argument("--total_timesteps", type=int, default=200_000,
                        help="Total PPO timesteps.")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments.")
    parser.add_argument("--max_steps_per_episode", type=int, default=8,
                        help="Maximum steps per episode in the env.")
    parser.add_argument("--history_window", type=int, default=-1,
                        help="History window for encoding. -1 => full history.")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="PPO learning rate.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="PPO batch size.")
    parser.add_argument("--n_steps", type=int, default=64,
                        help="Number of steps to run per PPO update.")
    parser.add_argument("--gamma", type=float, default=0.90,
                        help="Discount factor.")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient.")

    # Checkpointing / control
    parser.add_argument("--checkpoint_freq", type=int, default=1000,
                        help="Save a PPO checkpoint every N environment steps.")
    parser.add_argument("--checkpoint_dir", type=str, default="./experiments/checkpoints1",
                        help="Directory for intermediate PPO checkpoints.")
    parser.add_argument("--stop_file", type=str, default="./experiments/STOP_TRAINING",
                        help="Path to a file that, if exists, will cause training to stop cleanly.")

    # Eval params
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of episodes to evaluate after training.")

    # Logging / IO
    parser.add_argument("--tensorboard_log", type=str, default="./tb_logs_harassment",
                        help="TensorBoard log dir.")
    parser.add_argument("--output_dir", type=str, default="./experiments/harassment_rl_outputs",
                        help="Directory to save model and evaluation trajectories.")
    parser.add_argument("--use_subproc_vecenv", action="store_true",
                        help="Use SubprocVecEnv instead of DummyVecEnv when num_envs > 1.")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of CPU threads for Torch.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle history_window=-1 => None
    history_window = None if args.history_window < 0 else args.history_window

    # Basic runtime configuration
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(args.num_threads)
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)

    # Log args for reproducibility
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # --------------------------- TRAINING ---------------------------
    start_time = time.time()
    model_path = train_ppo(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        max_steps_per_episode=args.max_steps_per_episode,
        history_window=history_window,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        tensorboard_log=args.tensorboard_log,
        output_dir=args.output_dir,
        use_subproc=args.use_subproc_vecenv,
        seed=args.seed,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        stop_file=args.stop_file,
    )
    train_time = time.time() - start_time

    print(f"Training finished. Model saved at: {model_path}")
    print(f"Training wall-clock time: {train_time:.2f} seconds")

    # --------------------------- EVALUATION ---------------------------
    eval_jsonl = evaluate_policy(
        model_path=model_path,
        num_episodes=args.eval_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        history_window=history_window,
        output_dir=args.output_dir,
        seed=args.seed + 1,
    )

    print(f"Evaluation trajectories saved at: {eval_jsonl}")


if __name__ == "__main__":
    main()
