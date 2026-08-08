# In algorithms/qnas/configs.py

from dataclasses import dataclass, field
from typing import List, Union, Any

@dataclass
class NetworkRulesConfig:
    """Configuration for NAS architectural rules and constraints."""
    terminal_op_name: str = "no_op"
    pool_op_name: Union[str, List[str]] = "pool"
    min_active_len: int = 5
    truncate_after_noop: bool = True
    avoid_consecutive_pool: bool = True
    enforce_noop_in_update: bool = True
    noop_max_prob: float = 0.90
    noop_ramp_cap: bool = True

@dataclass
class EliteUpdateConfig:
    """Configuration for the quantum elite update strategy."""
    elite_mode: str = "global_k"
    k_elites: int = 5
    pool_factor: int = 2
    ema_beta: float = 0.7
    rank_weighting: bool = True

@dataclass
class MOEAConfig:
    """Configuration for MOEA/D-specific parameters."""
    moead_q_low: float = 0.30
    moead_q_high: float = 0.90
    topP_mult: int = 5
    ref_dir_method: str = "das-dennis"