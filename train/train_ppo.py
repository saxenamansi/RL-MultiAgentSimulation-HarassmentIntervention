import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.harassment_env import HarassmentInterventionEnv, EnvConfig
from models.mistral_wrappers import llm_harasser, llm_victim, llm_intervener
from models.harassment_koala import harassment_score
from models.text_encoder import text_encoder
from personas.personas import get_persona


def make_env():
    config = EnvConfig(max_steps=8, history_window=None)
    env = HarassmentInterventionEnv(
        llm_harasser=llm_harasser,
        llm_victim=llm_victim,
        llm_intervener=llm_intervener,
        get_persona=get_persona,
        harassment_scorer=harassment_score,
        text_encoder=text_encoder,
        config=config,
        seed=None,
    )
    return env


if __name__ == "__main__":
    # Performance / determinism settings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)

    # Vectorized env (1 worker for now; Mistral will be the bottleneck)
    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./tb_logs_harassment/",
    )

    # You can start smaller (e.g., 10_000) for smoke testing
    model.learn(total_timesteps=200_000)
    model.save("ppo_harassment_intervener")

    # quick eval
    eval_env = make_env()
    obs, info = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(int(action))
        print(
            f"Step={eval_env.step_count}, "
            f"action={int(action)}, "
            f"reward={reward:.3f}, "
            f"score={eval_env.harassment_score:.2f}"
        )
    eval_env.render()
