from typing import Any, Dict, Generator, List, Optional, Union
import numpy as np
from stable_baselines3.common.type_aliases import RolloutBufferSamples
import torch as th
from gymnasium import spaces

# 轨迹replaybuffer，用于每次收集完使用完清空
from stable_baselines3.common.buffers import RolloutBuffer
# repalybuffer，不清空，但是很大
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from .type_aliases import RolloutBufferSamplesFVRL, ReplayBufferSamplesFVRL

class RolloutBuffer_FVRL(RolloutBuffer):
    """
    在他们的基础上，加上一个打分器奖励的内容
    """

    def reset(self) -> None:
        """
        添加一个score_reward的字段，用于额外的存储
        """
        self.scores = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32)
        return super().reset()
    
    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        在RolloutBuffer的基础上进行修改
        把打分器的内容，加到奖励上
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            ### 修改之后，advantage = reward + scores + gamma * next_values -  values
            ### return = advantage + values = reward + scores + gamma * next_values
            ### 为什么不用gae，因为我还没想好这个打分器和gae怎么适配
            delta = self.rewards[step] + self.scores[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
    
    def add(self,
            obs: np.ndarray, 
            action: np.ndarray, 
            reward: np.ndarray, 
            episode_start: np.ndarray, 
            value: th.Tensor, 
            log_prob: th.Tensor,
            scores: np.ndarray) -> None:
        
        self.scores[self.pos] = np.array(scores).copy()
        
        return super().add(obs, action, reward, episode_start, value, log_prob)
    
    def get(self, batch_size: Optional[int]=None) -> Generator[RolloutBufferSamplesFVRL, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "scores",  # 添加自己的分数内容
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamplesFVRL:  # type: ignore[signature-mismatch] #FIXME
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.scores[batch_inds].flatten(),
        )
        return RolloutBufferSamplesFVRL(*tuple(map(self.to_torch, data)))


class ReplayBuffer_FVRL(ReplayBuffer):
    """
    在他们的基础上，加上一个打分器奖励的内容
    这个是回放缓冲区的内容
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage, handle_timeout_termination=handle_timeout_termination)
        # 添加上分数的内容
        self.scores = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        scores: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        添加分数的经验
        """
        self.scores[self.pos] = np.array(scores).copy()
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamplesFVRL:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super().sample(batch_size, env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamplesFVRL:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.scores[batch_inds, env_indices],
        )
        return ReplayBufferSamplesFVRL(*tuple(map(self.to_torch, data)))