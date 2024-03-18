

import torch as th
from typing import Any, Callable, Dict, List, NamedTuple, Optional, SupportsFloat, Tuple, Union

class RolloutBufferSamplesFVRL(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    scores: th.Tensor

class ReplayBufferSamplesFVRL(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    scores: th.Tensor