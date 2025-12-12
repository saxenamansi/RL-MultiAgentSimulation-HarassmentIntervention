import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional
import random
import re

from goals.harassment_goals import HARASSMENT_GOALS

@dataclass
class EnvConfig:
    max_steps: int = 8
    history_window: Optional[int] = None
    alpha: float = 1.0    # weight for improvements
    beta: float = 0.5     # weight for being above safe band
    safe_threshold: float = 0.004
    fail_threshold: float = 0.02
    terminal_safe_bonus: float = 1.0
    terminal_fail_penalty: float = -2.0

    false_negative_penalty: float = 1.5   # P1: high harassment, no intervention
    false_positive_penalty: float = 0.5   # P2: low harassment, but intervention

    action_costs: Dict[int, float] = field(default_factory=lambda: {
        0: 0.0,  # do nothing
        1: 0.0,  # mild nudge
        2: 0.0,  # strong confrontation
        3: 0.0,  # escalate/report
    })

class HarassmentInterventionEnv(gym.Env):
    """
    Custom environment for harassment intervention with 3 LLM roles:
    - harasser
    - victim
    - intervener (RL agent chooses discrete action, LLM generates text)
    
    RL controls the intervener; harasser/victim are scripted by LLMs.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        llm_harasser: Callable[[str], str],
        llm_victim: Callable[[str], str],
        llm_intervener: Callable[[str], str],
        get_persona: Callable[[], Tuple[Dict[str, float], Dict[str, float]]],
        harassment_scorer: Callable[[List[str]], float],
        text_encoder: Callable[[str], np.ndarray],
        config: Optional[EnvConfig] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.llm_harasser = llm_harasser
        self.llm_victim = llm_victim
        self.llm_intervener = llm_intervener

        self.get_persona = get_persona
        self.harassment_scorer = harassment_scorer
        self.text_encoder = text_encoder

        self.config = config or EnvConfig()

        # Randomness
        self._rng = np.random.RandomState(seed)
        random.seed(seed)

        # Infer embedding dimension from encoder
        dummy_vec = self.text_encoder("dummy text for dimension")
        if not isinstance(dummy_vec, np.ndarray):
            dummy_vec = np.asarray(dummy_vec, dtype=np.float32)
        self.emb_dim = dummy_vec.shape[-1]

        # Observation:
        # [dialogue_embedding (emb_dim),
        #  harasser_traits (8),
        #  victim_traits (8),
        #  harassment_score (1),
        #  step_index (1)]
        obs_dim = self.emb_dim + 8 + 8 + 1 + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Actions: 0..3, as defined in config.action_costs
        self.action_space = spaces.Discrete(4)

        # Internal state
        self.history: List[str] = []
        self.h_traits: Dict[str, float] = {}
        self.v_traits: Dict[str, float] = {}
        self.h_traits_vec: np.ndarray = np.zeros(8, dtype=np.float32)
        self.v_traits_vec: np.ndarray = np.zeros(8, dtype=np.float32)
        self.harassment_score: float = 0.0
        self.max_harassment: float = 0.0
        self.step_count: int = 0

        self.harassment_goal_key: Optional[str] = None
        self.harassment_goal_text: str = ""

        self.last_intervention_for_harasser: Optional[str] = None
        self.last_intervention_for_victim: Optional[str] = None
        self.intervention_log: List[dict] = []


    # ---- Helper methods -------------------------------------------------

    def _traits_to_vec(self, traits: Dict[str, float]) -> np.ndarray:
        """
        Map your trait dict into a fixed 8-d vector.
        For now, we assume traits is an ordered dict-like or has stable ordering.
        Adapt as needed to match your actual trait schema.
        """
        # If already 8-dim numeric vector, just cast.
        if isinstance(traits, (list, tuple, np.ndarray)) and len(traits) == 8:
            return np.asarray(traits, dtype=np.float32)

        # Otherwise sort by key for deterministic mapping.
        items = sorted(traits.items(), key=lambda kv: kv[0])
        vals = [float(v) for _, v in items[:8]]

        return np.asarray(vals[:8], dtype=np.float32)

    def _bucket_trait(self, v: float) -> str:
        if v < 0.2:
            return "very low"
        elif v < 0.4:
            return "low"
        elif v < 0.6:
            return "moderate"
        elif v < 0.8:
            return "high"
        else:
            return "very high"


    def _build_persona_text(self, traits: Dict[str, float], role: str) -> str:
        """
        Convert traits into a short textual persona description for prompting the LLM.
        We keep both numeric values (0–1) and a coarse verbal label.
        """
    
        # Canonical order for Big Five + Dark Triad
        order = ["O", "C", "E", "A", "N", "M", "P", "R"]
        long_names = {
            "O": "openness",
            "C": "conscientiousness",
            "E": "extraversion",
            "A": "agreeableness",
            "N": "neuroticism",
            "M": "machiavellianism",
            "P": "psychopathy",
            "R": "narcissism",
        }
    
        parts = []
        for k in order:
            if k in traits:
                v = float(traits[k])
                bucket = self._bucket_trait(v)
                parts.append(f"{bucket} {long_names[k]} ({k}={v:.2f})")
    
        persona_desc = "; ".join(parts)
    
        return (
            f"You are the {role} in this conversation. "
            f"Your personality profile is: {persona_desc}. "
            "Behave consistently with this profile in your messages."
        )

    def _sample_harassment_goal(self):
        """
        Sample one harassment goal type for this episode.
        """
        keys = list(HARASSMENT_GOALS.keys())
        # self._rng is a numpy RandomState
        idx = self._rng.randint(len(keys))
        key = keys[idx]
        return key, HARASSMENT_GOALS[key]

    def _clean_reply(self, raw: str, role_label: str) -> str:
        """
        Remove duplicated role prefixes like 'Harasser:' or 'Victim:' and surrounding quotes.
        """
        text = raw.strip().strip('"').strip("'")

        # Remove repeated role labels at the start
        pattern = re.compile(rf"^{role_label}\s*:\s*", re.IGNORECASE)
        changed = True
        while changed:
            new_text = pattern.sub("", text).lstrip()
            changed = (new_text != text)
            text = new_text

        return text

    def _history_slice(self) -> List[str]:
        """
        Return the subset of history used for prompts/encoding.
        If history_window is None, return full history.
        """
        if self.config.history_window is None:
            return self.history
        return self.history[-self.config.history_window:]

    def _build_harasser_prompt(self) -> str:
        persona_txt = self._build_persona_text(self.h_traits, "harasser")
        goal_txt = self.harassment_goal_text
        history_txt = "\n".join(self._history_slice())

        system_notice = ""
        if self.last_intervention_for_harasser:
            system_notice = (
                "\n\n[Private platform notice to you (not visible to anyone else): "
                f"\"{self.last_intervention_for_harasser}\"]\n"
                "Do NOT reply to this notice directly. "
            )

        return (
            f"{persona_txt}\n"
            f"{goal_txt}\n"
            "Here is the conversation so far (a casual private chat):\n"
            f"{history_txt}"
            f"{system_notice}\n\n"
            "Write the harasser's next message as a short, informal chat message "
            "for a social media or private messaging app. Use 1–3 sentences. "
            "Do NOT include speaker labels like 'Harasser:'."
        )

    def _build_victim_prompt(self) -> str:
        persona_txt = self._build_persona_text(self.v_traits, "victim")
        history_txt = "\n".join(self._history_slice())

        system_notice = ""
        if self.last_intervention_for_victim:
            system_notice = (
                "\n\n[Private platform notice to you (not visible to anyone else): "
                f"\"{self.last_intervention_for_victim}\"]\n"
                "Do NOT reply to this notice directly. "
            )

        return (
            f"{persona_txt}\n"
            "You are experiencing potential harassment in an online private chat.\n"
            "Here is the conversation so far:\n"
            f"{history_txt}"
            f"{system_notice}\n\n"
            "Write the victim's next message as a short, informal chat message "
            "for a social media or private messaging app. Use 1–3 sentences. "
            "and do NOT include speaker labels like 'Victim:' and do NOT mention any platform notices."
        )

    def _build_intervener_prompt(self, action: int) -> str:
        """
        Build an internal prompt for the intervener LLM.
        The generated text is a platform-style notice to an individual agent.
        """
        history_txt = "\n".join(self._history_slice())
        base = (
            "You are generating short, neutral platform notices to mitigate an online sexual harassment issue that has developed between two individuals. Take all measures to strictly intervene and stop the harassment. Do NOTHING else. Here is the conversation so far between a harasser and a victim:\n"
            f"{history_txt}\n\n"
        )

        harassment_str = (
            f"Current harassment severity score (safe zone < 0.004): {self.harassment_score:.2f}.\n"
        )

        if action == 0:
            raise ValueError("Action 0 is 'do nothing' and should not call _build_intervener_prompt.")

        elif action == 1:
            directive = (
                "Write a very short notice (1–2 sentences) to the HARASSER reminding them that their "
                "recent message may violate community guidelines and that they should stop any unwanted behavior. "
            )
            
        elif action == 2:
            directive = (
                "Write a very short, firm notice (1–2 sentences) to the HARASSER stating that their behavior "
                "violates harassment policies and they must stop immediately. "
            )
        elif action == 3:
            directive = (
                "Write a short notice (1–2 sentences) to the HARASSER stating that their behavior has been reported to moderators "
                "and that their account has been temporarily suspended until further notice from the moderator. "
            )
        else:
            raise ValueError(f"Unknown action id {action}")

        return base + harassment_str + directive

    def _encode_state(self) -> np.ndarray:
        # Encode last K messages into a single vector
        history_txt = "\n".join(self._history_slice())
        emb = self.text_encoder(history_txt if history_txt else "no conversation yet")
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape[-1] != self.emb_dim:
            raise ValueError("Encoder dimension changed during runtime.")

        step_norm = np.array([self.step_count / max(1, self.config.max_steps - 1)], dtype=np.float32)
        hs = np.array([self.harassment_score], dtype=np.float32)

        obs = np.concatenate(
            [emb, self.h_traits_vec, self.v_traits_vec, hs, step_norm],
            axis=0,
        ).astype(np.float32)

        return obs

    # ---- Gym API --------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
            random.seed(seed)

        self.history = []
        self.step_count = 0
        self.harassment_score = 0.0
        self.max_harassment = 0.0

        # Sample personas
        self.h_traits, self.v_traits = self.get_persona()
        self.h_traits_vec = self._traits_to_vec(self.h_traits)
        self.v_traits_vec = self._traits_to_vec(self.v_traits)

        # sample harassment goal type for this episode
        self.harassment_goal_key, self.harassment_goal_text = self._sample_harassment_goal()

        # Add a generic scenario seed
        scenario_seed = (
            "System: You are observing an online text-based private chat between two individuals discussing general, everyday topics. "
        )
        self.history.append(scenario_seed)

        # Initial harassment score based on the seed
        self.harassment_score = float(self.harassment_scorer(self.history))
        self.max_harassment = self.harassment_score

        # NEW: force an initial harasser + victim exchange BEFORE RL
        harasser_prompt = self._build_harasser_prompt()
        harasser_msg = self.llm_harasser(harasser_prompt).strip()
        if harasser_msg:
            self._clean_reply(harasser_msg, "Harasser")
            self.history.append(f"Harasser: {harasser_msg}")
        
        victim_prompt = self._build_victim_prompt()
        victim_msg = self.llm_victim(victim_prompt).strip()
        if victim_msg:
            victim_msg = self._clean_reply(victim_msg, "Victim")
            self.history.append(f"Victim: {victim_msg}")
        
        # Update harassment score after initial exchange
        self.harassment_score = float(self.harassment_scorer(self.history))
        self.max_harassment = self.harassment_score

        obs = self._encode_state()
        info = {
            "step": self.step_count,
            "history": list(self.history),
            "harasser_traits": self.h_traits,
            "victim_traits": self.v_traits,
            "harassment_goal": self.harassment_goal_key,
            "harassment_score": self.harassment_score,
            "max_harassment": self.max_harassment,
            "intervention_log": list(self.intervention_log),   
        }
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        s_t = self.harassment_score
        self.step_count += 1

        # 1) Intervener message (if action != 0) – SIDE CHANNEL ONLY
        if action != 0:
            prompt = self._build_intervener_prompt(action)
            intervener_msg = self.llm_intervener(prompt).strip()

            if intervener_msg:
                if action in (1, 2, 3):   # nudges / confrontation / report
                    self.last_intervention_for_harasser = intervener_msg

                # Log for analysis, but DO NOT add to chat history
                self.intervention_log.append({
                    "step": self.step_count,
                    "action": int(action),
                    "text": intervener_msg,
                })

        # 2) Harasser message
        harasser_prompt = self._build_harasser_prompt()
        harasser_msg = self.llm_harasser(harasser_prompt).strip()
        if harasser_msg:
            harasser_msg = self._clean_reply(harasser_msg, "Harasser")
            self.history.append(f"Harasser: {harasser_msg}")

        # 3) Victim message
        victim_prompt = self._build_victim_prompt()
        victim_msg = self.llm_victim(victim_prompt).strip()
        if victim_msg:
            victim_msg = self._clean_reply(victim_msg, "Victim")
            self.history.append(f"Victim: {victim_msg}")
            
        # 4) Update harassment score
        s_tp1 = float(self.harassment_scorer(self.history))
        self.harassment_score = s_tp1
        self.max_harassment = max(self.max_harassment, s_tp1)

        # 5) Reward computation

        # Δs = s_t - s_{t+1} -> positive if harassment goes down
        delta_s = s_t - s_tp1

        fail_t = max(self.config.fail_threshold, 1e-6)
        safe_t = self.config.safe_threshold
        band = max(fail_t - safe_t, 1e-6)

        # Reward reductions in harassment, normalized
        delta_term = (delta_s / fail_t)  # >0 if harassment decreases

        # Penalize being above the safe band (0 penalty below safe_threshold)
        overshoot = max(0.0, s_tp1 - safe_t)
        level_term = (overshoot / band)

        # Base shaping reward
        r = self.config.alpha * delta_term - self.config.beta * level_term

        # P1: high harassment, but NO intervention (false negative)
        if s_tp1 >= self.config.fail_threshold and action == 0:
            r -= self.config.false_negative_penalty

        # P2: low harassment, but intervention (false positive)
        if s_tp1 <= self.config.safe_threshold and action != 0:
            r -= self.config.false_positive_penalty

        # 6) Termination conditions
        done = False
        truncated = False

        # Hard safety cut-off
        if s_tp1 >= 1.0:  # still a guardrail, probably never triggered with current scorer
            done = True

        # Max steps
        if self.step_count >= self.config.max_steps:
            done = True

        # Terminal bonuses/penalties based on the maximum harassment level seen
        if done:
            if self.max_harassment <= self.config.safe_threshold and self.step_count > 1:
                # Only give safe bonus if you actually let some interaction happen
                r += self.config.terminal_safe_bonus
            elif self.max_harassment >= self.config.fail_threshold:
                r += self.config.terminal_fail_penalty

        obs = self._encode_state()
        info = {
            "history": list(self.history),
            "harasser_traits": self.h_traits,
            "victim_traits": self.v_traits,
            "harassment_score": self.harassment_score,
            "max_harassment": self.max_harassment,
            "intervention_log": list(self.intervention_log),
        }

        return obs, r, done, truncated, info

    def render(self):
        print("\n--- Conversation ---")
        for msg in self.history:
            print(msg)
        print(f"\nHarassment score: {self.harassment_score:.2f} (max: {self.max_harassment:.2f})")

    def close(self):
        pass
