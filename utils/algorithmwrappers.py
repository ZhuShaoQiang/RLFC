"""
在这里对现有的算法进行改造，改造成我想用的算法
"""

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

from stable_baselines3.common.buffers import DictRolloutBuffer, ReplayBuffer, RolloutBuffer, DictReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import VecEnv
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TrainFreq
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.utils import obs_as_tensor, explained_variance, get_schedule_fn
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFrequencyUnit
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.type_aliases import MaybeCallback

from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, Callable
from copy import deepcopy
import numpy as np
import time
import sys

from .bufferwrappers import RolloutBuffer_FVRL, ReplayBuffer_FVRL

class FVRL():
    """一个FVRL的父类，包装内容，让其他的东西直接能使用
    """
    def __init__(self, scorer: th.nn.Module, 
                 scorer_func: Callable[[float], float], 
                 params: dict, zf = None) -> None:
        self.scorer = scorer
        self.scorer.eval()
        self.scorer_func = scorer_func
        self._zf = zf
        self.params = params
        self.obs_his = []
    
    def _process_obs(self, x, update=False) -> th.nn.Module:
        """
        处理obs
        对obs归一化并转为tensor
        如果zf没有，就直接返回x
        """
        if self._zf == None:
            x = x
        else:
            x = self._zf(x, update=update)  # 归一化
        if not isinstance(x, th.Tensor):
            x = th.from_numpy(x).to(self.params["device"]).to(th.float32)
        return x

    def _score2reward(self, score) -> float:
        """
        通过scorer_func计算当前打分真正的得分，这个得分是对于当前的行动的一个立即奖励
        得到的分数是一个float，这个分数不低于0
        """
        return self.scorer_func(score)

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer_FVRL,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        scores: np.ndarray,
    ) -> None:
        """
        重新往buffer里存储的内容
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(self._last_original_obs, next_obs, buffer_action, reward_, dones, scores, infos)
        # replay_buffer.add(self._last_original_obs,next_obs,buffer_action,reward_,dones,scores,infos)

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
        

class FVRL_PPO(PPO):
    """
    在PPO的基础上，加上我自己的想法
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        scorer: th.nn.Module,  # 打分器
        scorer_func: Callable[[float], float],  # 打分器到分数的一个函数
        zf,                                     # 关于状态的归一化
        params: dict,  # 参数 
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        tensorboard_log=None,
        gae_lambda=0.95,
    ):
        super().__init__(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            gae_lambda=gae_lambda,
        )
        self.scorer = scorer
        self.scorer.eval()
        self.scorer_func = scorer_func
        self._zf = zf
        self.params = params
        self.obs_his = []

    def _process_obs(self, x, update=False) -> th.nn.Module:
        """
        处理obs
        对obs归一化并转为tensor
        """
        x = self._zf(x, update=update)  # 归一化
        if not isinstance(x, th.Tensor):
            x = th.from_numpy(x).to(self.params["device"]).to(th.float32)
        return x
    
    def collect_rollouts(self, env: VecEnv, 
                         callback: BaseCallback, 
                         rollout_buffer: RolloutBuffer, 
                         n_rollout_steps: int) -> bool:
        """
        在原本的收集回放缓冲区的时候，添加打分器的内容
        1. 打分器需要记录k-1个历史状态，用第k个状态打分
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            ### 自己添加的关于打分器的内容
            # 如果里面没有内容，就填入足够数量的内容
            if len(self.obs_his) < self.params["time_len"]:  # 正常来讲，长度应该是time_len
                for _ in range(self.params["time_len"]):
                    self.obs_his.append(
                        self._process_obs(self._last_obs)
                    )
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            ### TODO: 加上打分器计算打分
            # 拼接新的状态
            self.obs_his = self.obs_his[-(self.params["time_len"]-1):] + [self._process_obs(new_obs)]
            tmp = th.cat(self.obs_his, dim=1).to(self.params["device"])
            score = self.scorer(tmp).item()
            reward_from_scorer = self._score2reward(score=score)
            ###

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

                    ### 每次环境结束重新计算
                    self.obs_his = []

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                reward_from_scorer,  # 添加打分器打出的分数
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    def _setup_model(self) -> None:
        # base的初始化
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        # 判定使用哪个buffer
        if isinstance(self.observation_space, spaces.Dict):
            buffer_cls = DictRolloutBuffer
        else:
            buffer_cls = RolloutBuffer_FVRL

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        # pytype:enable=not-instantiable
        self.policy = self.policy.to(self.device)

        ### PPO的初始化
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
    
    def _score2reward(self, score) -> float:
        """
        通过scorer_func计算当前打分真正的得分，这个得分是对于当前的行动的一个立即奖励
        得到的分数是一个float，这个分数不低于0
        """
        return self.scorer_func(score)
    
    def train(self) -> None:
        """
        在原本的PPO的基础上，加上训练奖励的部分
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        # 确定这里使用的是FVRL版本的rolloutbuffer
        assert isinstance(self.rollout_buffer, RolloutBuffer_FVRL)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # 拿到一批数据
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                ### 拿到之前的分数
                score = rollout_data.scores

                # Normalize advantaoe
                advantages = rollout_data.advantages + score
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

class FVRL_PPO_2(PPO):
    """
    在PPO的基础上，加上我自己的想法
    这个地方是actor和critic都加上奖励的内容
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        scorer: th.nn.Module,  # 打分器
        scorer_func: Callable[[float], float],  # 打分器到分数的一个函数
        zf,                                     # 关于状态的归一化
        params: dict,  # 参数 
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        tensorboard_log=None,
        gae_lambda=0.95,
    ):
        super().__init__(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            gae_lambda=gae_lambda,
        )
        self.scorer = scorer
        self.scorer.eval()
        self.scorer_func = scorer_func
        self._zf = zf
        self.params = params
        self.obs_his = []

    def _process_obs(self, x, update=False) -> th.nn.Module:
        """
        处理obs
        对obs归一化并转为tensor
        """
        x = self._zf(x, update=update)  # 归一化
        if not isinstance(x, th.Tensor):
            x = th.from_numpy(x).to(self.params["device"]).to(th.float32)
        return x
    
    def collect_rollouts(self, env: VecEnv, 
                         callback: BaseCallback, 
                         rollout_buffer: RolloutBuffer, 
                         n_rollout_steps: int) -> bool:
        """
        在原本的收集回放缓冲区的时候，添加打分器的内容
        1. 打分器需要记录k-1个历史状态，用第k个状态打分
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            ### 自己添加的关于打分器的内容
            # 如果里面没有内容，就填入足够数量的内容
            if len(self.obs_his) < self.params["time_len"]:  # 正常来讲，长度应该是time_len
                for _ in range(self.params["time_len"]):
                    self.obs_his.append(
                        self._process_obs(self._last_obs)
                    )
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            ### TODO: 加上打分器计算打分
            # 拼接新的状态
            self.obs_his = self.obs_his[-(self.params["time_len"]-1):] + [self._process_obs(new_obs)]
            tmp = th.cat(self.obs_his, dim=1).to(self.params["device"])
            score = self.scorer(tmp).item()
            reward_from_scorer = self._score2reward(score=score)
            ###

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

                    ### 每次环境结束重新计算
                    self.obs_his = []

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                reward_from_scorer,  # 添加打分器打出的分数
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    def _setup_model(self) -> None:
        # base的初始化
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        # 判定使用哪个buffer
        if isinstance(self.observation_space, spaces.Dict):
            buffer_cls = DictRolloutBuffer
        else:
            buffer_cls = RolloutBuffer_FVRL

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        # pytype:enable=not-instantiable
        self.policy = self.policy.to(self.device)

        ### PPO的初始化
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
    
    def _score2reward(self, score) -> float:
        """
        通过scorer_func计算当前打分真正的得分，这个得分是对于当前的行动的一个立即奖励
        得到的分数是一个float，这个分数不低于0
        """
        return self.scorer_func(score)
    
    def train(self) -> None:
        """
        在原本的PPO的基础上，加上训练奖励的部分
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        # 确定这里使用的是FVRL版本的rolloutbuffer
        assert isinstance(self.rollout_buffer, RolloutBuffer_FVRL)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # 拿到一批数据
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                ### 拿到之前的分数
                score = rollout_data.scores

                # Normalize advantaoe
                advantages = rollout_data.advantages + score
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                ### 添加入score的内容
                value_loss = F.mse_loss(rollout_data.returns + score, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

class FVRL_PPO_3(PPO):
    """
    FVRL_PPO_2把打分器的分数加错地方了，我在这里进行改正
    应该加到reward上，2加在了advantage上，看看怎么效果好用哪个
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        scorer: th.nn.Module,  # 打分器
        scorer_func: Callable[[float], float],  # 打分器到分数的一个函数
        zf,                                     # 关于状态的归一化
        params: dict,  # 参数 
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        tensorboard_log=None,
        gae_lambda=0.95,
    ):
        super().__init__(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            gae_lambda=gae_lambda,
        )
        self.scorer = scorer
        self.scorer.eval()
        self.scorer_func = scorer_func
        self._zf = zf
        self.params = params
        self.obs_his = []

    def _process_obs(self, x, update=False) -> th.nn.Module:
        """
        处理obs
        对obs归一化并转为tensor
        """
        x = self._zf(x, update=update)  # 归一化
        if not isinstance(x, th.Tensor):
            x = th.from_numpy(x).to(self.params["device"]).to(th.float32)
        return x
    
    def collect_rollouts(self, env: VecEnv, 
                         callback: BaseCallback, 
                         rollout_buffer: RolloutBuffer, 
                         n_rollout_steps: int) -> bool:
        """
        在原本的收集回放缓冲区的时候，添加打分器的内容
        1. 打分器需要记录k-1个历史状态，用第k个状态打分
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            ### 自己添加的关于打分器的内容
            # 如果里面没有内容，就填入足够数量的内容
            if len(self.obs_his) < self.params["time_len"]:  # 正常来讲，长度应该是time_len
                for _ in range(self.params["time_len"]):
                    self.obs_his.append(
                        self._process_obs(self._last_obs)
                    )
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            ### TODO: 加上打分器计算打分
            # 拼接新的状态
            self.obs_his = self.obs_his[-(self.params["time_len"]-1):] + [self._process_obs(new_obs)]
            tmp = th.cat(self.obs_his, dim=1).to(self.params["device"])
            score = self.scorer(tmp).item()
            reward_from_scorer = self._score2reward(score=score)
            ###

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

                    ### 每次环境结束重新计算
                    self.obs_his = []

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                reward_from_scorer,  # 添加打分器打出的分数
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    def _setup_model(self) -> None:
        # base的初始化
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        # 判定使用哪个buffer
        if isinstance(self.observation_space, spaces.Dict):
            buffer_cls = DictRolloutBuffer
        else:
            buffer_cls = RolloutBuffer_FVRL

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        # pytype:enable=not-instantiable
        self.policy = self.policy.to(self.device)

        ### PPO的初始化
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
    
    def _score2reward(self, score) -> float:
        """
        通过scorer_func计算当前打分真正的得分，这个得分是对于当前的行动的一个立即奖励
        得到的分数是一个float，这个分数不低于0
        """
        return self.scorer_func(score)
    
class FVRL_DQN(DQN):
    """
    根据原本的dqn改写的使用fvrl的dqn
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        scorer: th.nn.Module,  # 打分器
        scorer_func: Callable[[float], float],  # 打分器到分数的一个函数
        params: dict,  # 参数 
        policy_kwargs: Optional[Dict[str, Any]] = None,
        zf=None,                                     # 关于状态的归一化
        verbose: int = 0,
        tensorboard_log=None,
    ):
        super().__init__(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
        )
        self.scorer = scorer
        self.scorer.eval()
        self.scorer_func = scorer_func
        self._zf = zf
        self.params = params
        self.obs_his = []

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer_FVRL

        if self.replay_buffer is None:
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,  # pytype:disable=wrong-keyword-args
            )

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()
        ### 上面是offpolicy的，下面是dqn的
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """添加我们自己的分数的系统
        target_q = rewards + scores

        Args:
            gradient_steps (int): _description_
            batch_size (int, optional): _description_. Defaults to 100.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                # target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                ### 这里改成+score
                target_q_values = replay_data.rewards + replay_data.scores + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer_FVRL,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        scores: np.ndarray,
    ) -> None:
        """
        重新往buffer里存储的内容
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            scores,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
    
    def _process_obs(self, x, update=False) -> th.nn.Module:
        """
        处理obs
        对obs归一化并转为tensor
        """
        if not self._zf is None:
            x = self._zf(x, update=update)  # 归一化
        if not isinstance(x, th.Tensor):
            if x.shape == (1,):
                ## 默认是这个形状cw，转成one_hot
                x = np.eye(48)[x]
            x = th.from_numpy(x).to(self.params["device"]).to(th.float32)
        return x

    def _score2reward(self, score) -> float:
        """
        通过scorer_func计算当前打分真正的得分，这个得分是对于当前的行动的一个立即奖励
        得到的分数是一个float，这个分数不低于0
        """
        return self.scorer_func(score)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq,
        replay_buffer: ReplayBuffer_FVRL,
        action_noise = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        添加计算分数的内容
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            ### 计算得到的分数，然后存入回放缓冲区
            if len(self.obs_his) < self.params["time_len"]:  # 正常来讲，长度应该是time_len
                for _ in range(self.params["time_len"]):
                    self.obs_his.append(
                        self._process_obs(self._last_obs)
                    )
            self.obs_his = self.obs_his[-(self.params["time_len"]-1):] + [self._process_obs(new_obs)]
            tmp = th.cat(self.obs_his, dim=1).to(self.params["device"])
            score = self.scorer(tmp).item()
            reward_from_scorer = np.array(
                [self._score2reward(score=score)],
                dtype=np.float32,
            )
            ### done

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            ### 添加打分器的内容
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos, reward_from_scorer)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)


class FVRL_SAC(FVRL, SAC):
    """自己改写的SAC，与其他一样，打的分数加在reward上
    """

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        scorer: th.nn.Module,  # 打分器
        scorer_func: Callable[[float], float], # 打分器的函数
        params: dict, # 定义的参数
        zf = None, # 关于状态的归一化
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
    ):
        super().__init__(
            scorer,
            scorer_func,
            params,
            zf,
        )
        super(FVRL, self).__init__(  # 这里是原sac执行的init函数
            policy,
            env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
        )

    def _setup_model(self) -> None:
        """重写sac的setupmodel
        """
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        print(self.replay_buffer_class)
        print(self.replay_buffer)
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                print("使用dictreplaybufuer")
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer_FVRL
                # 这里修改使用自己写的FVRL版本的replaybuffer

        if self.replay_buffer is None:
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            ### TODO: 用不到HerReplayBuffer，所以目前不使用，先不修改
            if issubclass(self.replay_buffer_class, HerReplayBuffer):
                assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
                replay_buffer_kwargs["env"] = self.env
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,  # pytype:disable=wrong-keyword-args
            )

        # 设置策略网络
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        # 把train_freq的变量转换为一个同一个类型
        self._convert_train_freq()

        ### 上面的是最上层的_steup_model的内容，下面的是sac的内容

        # 把policy.actor重命名为actor，等等
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC_FVRL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, 
                         train_freq: TrainFreq, replay_buffer: ReplayBuffer, 
                         action_noise: ActionNoise = None, 
                         learning_starts: int = 0,
                         log_interval: int = None) -> RolloutReturn:
        """
        重写SAC的collect_rollouts，其他都不变，但是添加多一个打分器的内容
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):

            ### 训练之前检查obs_his里面的数量够不够
            if len(self.obs_his) < self.params["time_len"]:  # 正常来讲，长度应该是time_len
                for _ in range(self.params["time_len"]):
                    self.obs_his.append(
                        self._process_obs(self._last_obs)
                    )

            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            ### 计算打分器给出的分数
            self.obs_his = self.obs_his[-(self.params["time_len"]-1):] + [self._process_obs(new_obs)]
            tmp = th.cat(self.obs_his, dim=1).to(self.params["device"])
            score = self.scorer(tmp).item()
            reward_from_scorer = self._score2reward(score=score)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            ### 添加打分器的打分
            self._store_transition(
                replay_buffer, 
                buffer_actions, 
                new_obs, 
                rewards, 
                dones, 
                infos,
                reward_from_scorer)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        重写train,适配fvrl
        对于评论家
        sac: target_q_value = rewards + (1-done)*next_q_value
        sac_fvrl: target_q_value = rewards + scores + (1-done)*next_q_value

        对于演员：
        sac: actor_loss = (ent_coef * log_prob - cur_q_vlaues).mean()
        sac_fvrl: actor_loss = (ent_coef * log_prob - (cur_q_vlaues+scores)).mean() 

        本质都是对于一个比较好的动作调整梯度，向这个方向更新更多的距离
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                ### 修改这里
                target_q_values = replay_data.scores +  replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            ### 修改这里
            # actor_loss = (ent_coef * log_prob - (min_qf_pi+replay_data.scores)).mean()
            ### 这里不需要修改，因为critic中隐含这个min_qf_pi中含有score了
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    
