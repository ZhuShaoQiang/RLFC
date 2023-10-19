# -*- coding: utf-8 -*-

"""
存放算法的文件，如PPO
此处的PPO算法我重新写，不按照之前的写了，之前的每收集2048步训练一次不妥当，在这个环境下应该每次死亡后，训练一次
"""
import torch
import numpy as np
from gym import Env

from . import models
from .network import BaseNetwork
from .models import BaseModel, DQN, BaseAC, BaseQ
from .replaybuffer import ReplayBuffer
from .logger import Logger

from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """
    本文件所有的算法都会继承这个类，作为基础算法
    """
    def __init__(self, env: Env, params: dict, policy: BaseModel) -> None:
        """
        env: 环境
        params: 一个字典，是config文件中的配置文件信息中的字典
        """
        super().__init__()
        self.env = env
        self.params = params
        self.policy = policy.to(params["device"])
        self.rb = ReplayBuffer(params["buffer_size"])

        if "LOGS_PATH" in self.params and self.params["LOGS_PATH"] != "":
            self.logger = Logger(params["LOGS_PATH"])
        else:
            self.logger = None

        # 设定策略的优化器
        self.policy_optimizer = self.params["optimizer"](
            self.policy.parameters(),
            lr=self.params["lr"]
        )
        # 判断是否需要衰减探索率
        if self.params["epsilon_decay"]:
            self.params["epsilon"] = self.params["init_epsilon"]

        self.episode_reward = 0
        self.episodes_reward = []
        self.last_n_win = []  # 记录最近10场胜利情况，胜利为1，失败为0
    
    @torch.no_grad()
    def collect_rollouts(self, episode_num):
        """
        收集经验，在这里每次收集一轮，而不是固定步数（因为这个任务太简单了）
        这里写的是一个最基础的版本
        """
        # 设定为评估模式
        self.policy.eval()
        self.episode_reward = 0

        # 重置环境
        state, info = self.env.reset()  # [1, state_dim]
        state = state.to(self.params["device"])
        while True:
            # 通过网络，得到当前状态的各个动作的价值
            # 计算动作，由于这个网络产生的是argmax产生得一个值，所以ori_action和action_to_take应该是一样的
            ori_action, action_to_take = self.predict(state)
            # action_values = self.policy.forward(x=state).squeeze()
            # 采样动作
            # action_to_take = self.policy.sample_action(action_values, self.params["epsilon"])
            # 得到当前动作的预估价值
            # action_to_take_value = action_values[action_to_take]

            # 在环境中采取这个步骤
            state_next, reward, win, die, info = self.env.step(action_to_take)
            done = win or die

            ### 记录内容，动作应该记录产生的，而不是采取的
            self.cache(state, ori_action, reward, state_next, done, info)
            self.episode_reward += reward
            
            if done:
                self.episodes_reward.append(self.episode_reward)

                # 记录胜利记录
                if win:
                    self.last_n_win.append(1)
                    self.win_tims += 1
                else:
                    self.last_n_win.append(0)
                self.last_n_win = self.last_n_win[-self.params["last_n_avg_rewards"]:]
                
                ### 每回合记录日志
                self.log(episode_num)
                # self.logger.record_num("episode reward", self.episode_reward, episode_num)
                # self.logger.record_num("last n avg reward", self.last_n_avg_reward, episode_num)
                # self.logger.record_num("last n win ratio", np.mean(self.last_n_win), episode_num)
                print(f"近n轮平均回报:{self.last_n_avg_reward}, 本轮回报：{self.episode_reward}; ", end="")
                return
            # fi done

            state = state_next.to(self.params["device"])

    @abstractmethod
    def predict(self, x, deterministic=False):
        """
        计算应该采取的动作，只计算动作
        x: 输入的状态
        deterministic: 是否确定性采取动作
        """
        raise f"在Base Algorithm中未实现这个方法，这个应该在视线中实现"
    @abstractmethod
    def log(self, episode_num):
        """
        记录日志，记录什么由具体实现决定
        episode_num: 多少个episode
        """
        raise f"在Base Algorithm中未实现这个方法，这个应该在视线中实现"
    
    @abstractmethod
    def compute_advantage_and_return(self):
        """
        根据收集到的经验，计算优势和回报
        """
        raise f"在Base Algorithm中未实现这个方法 compute_advantage_and_return，请在继承类中重写"
        
    @abstractmethod
    def train(self):
        """
        训练网络
        这个需要具体实现
        """
        raise f"BaseAlgorithm中未实现此算法，需要具体实现"

    @abstractmethod
    def learn(self):
        """
        学习
        total_timestep: 总共训练的时间步数量
        """
        raise f"BaseAlgorithm中未实现此算法"
    
    @property
    def last_n_avg_reward(self):
        """
        计算最近100局的平均奖励，后面改成n
        """
        if len(self.episodes_reward) <= 0:
            return 0
        return np.mean(self.episodes_reward[-self.params["last_n_avg_rewards"]:])
    

    def cache(self, *args):
        """
        存储经验，这个经验的个数都是自定义的
        """
        # 这里还能对经验进行处理，这里就不处理了，不需要处理
        self.rb.put(experience=list(args))

    def reset_rb(self):
        """
        重置回放缓冲区
        其实不需要重置，因为超过最大大小的都会被弹出
        """
        self.rb = ReplayBuffer(self.params["buffer_size"])

class VanillaPPO(BaseAlgorithm):
    """
    普通的PPO的算法
    按照每次收集固定长度内容的方法
    """
    def __init__(self, env: Env, params: dict, policy: BaseAC) -> None:
        super().__init__(env, params, policy)
        print(self.policy.state_dict)
        self.optimizer = self.params["optimizer"](
            self.policy.parameters(), lr=self.params["lr"]
        )
        self.critic_loss_fn = torch.nn.MSELoss()
        self._last_obs = None
        self.win_tims = 0
        self.total_episodes = 0
        self.next_state_value = 0
        self.episode_reward = 0
    
    @torch.no_grad()
    def collect_rollouts(self):
        """
        收集一轮经验
        一般环境都是1000最大步数
        """
        self.policy.eval()

        if self._last_obs == None:  # 就是第一次执行这个环境重置
            self._last_obs, _ = self.env.reset()

        for _ in range(0, self.params["buffer_size"]):
            self._last_obs = self._last_obs.to(self.params["device"])
            action_mean, value = self.policy.forward(x=self._last_obs)  # 得到action_mean,value
            # 得到动作和action_log_prob
            action, action_log_prob = self.policy.sample_action(action_mean)

            action_to_take = np.clip(
                action.squeeze(0).cpu(), 
                self.env.action_space.low,
                self.env.action_space.high).numpy()

            state_next, reward, win, die, _ = self.env.step(action=action_to_take)
            self.episode_reward += reward
            done = win or die

            ### 记录内容
            self.cache(self._last_obs, action, reward, value, done, action_log_prob, 0, 0)
            
            if done:
                # 记录各种内容
                self.total_episodes += 1
                self.episodes_reward.append(self.episode_reward)
                # print(f"近n轮平均回报:{self.last_n_avg_reward}, 本轮回报：{self.episode_reward}")
                self.log(self.total_episodes)

                # 记录完成，清理环境和各种内容
                self.episode_reward = 0
                if win:
                    self.win_tims += 1
                
                self._last_obs, _ = self.env.reset()
            else:
                self._last_obs = state_next
        
        # 拿到2048个数据之后，还要拿到第2049个状态的估计价值
        self._last_obs = self._last_obs.to(self.params["device"])
        self.next_state_value = self.policy.get_state_value_only(self._last_obs)
        print(f"近n轮平均回报：{self.last_n_avg_reward}", flush=True)

    def compute_advantage_and_return(self):
        """
        本地方修改自StableBaseline的PPO算法，但是删除了gae部分
        因为要和PPO比较
        根据收集到的经验，计算优势和回报
        0        1      2       3      4      5               6     7
        state, action, reward, value, done, action_log_prob,  0,    0)
        状态    动作    环境奖励  预估奖励  完成   动作概率         优势  return
        这次的收集环境经验是固定数量收集，不是一回合一回合的收集
        self.next_state_value: 需要用到第2049个状态的奖励，就是参数
        """
        last_gae_lam = 0  # 使用gae的算法
        for idx in reversed(range(self.params["buffer_size"])):
            is_not_done = 1 - self.rb.buffer[idx][4]  # 对done取反，则done了是0，未done是1
            if self.params["buffer_size"] - 1 == idx:
                # 如果是回放缓冲区的最后一个，则要查看第2049个状态的价值和当前是否结束
                idx_plus_1_reward = self.next_state_value
            else:
                # 否则就直接用下一个状态的估计来叠加计算
                # TODO: 改成下一个状态的环境的立即奖励是否更好?
                idx_plus_1_reward = self.rb.buffer[idx+1][3]
            
            # 计算return = 当前环境立即奖励 + gamma * 下一步的估计奖励 * 当前未结束
            # TODO: 为什么不是乘以下一步的环境的立即奖励
            self.rb.buffer[idx][7] = self.rb.buffer[idx][2] + self.params["gamma"] * idx_plus_1_reward * is_not_done

            # 计算advantage=当前实际回报 - 当前估计回报
            # self.rb.buffer[idx][6] = self.rb.buffer[idx][7] - self.rb.buffer[idx][3]

            # 下面是用gae的算法计算advantage
            delta = self.rb.buffer[idx][7] - self.rb.buffer[idx][3]
            if self.params["use_gae"]:
                last_gae_lam = delta + self.params["gamma"] * self.params["gae_lambda"] * is_not_done * last_gae_lam
            else:
                last_gae_lam = delta
            
            ### 计算advantage
            self.rb.buffer[idx][6] = last_gae_lam
    
    def train(self):
        """
        根据收集到的经验，训练
        0        1      2       3      4      5               6
        state, action, return, value, done, action_log_prob, advantage
        状态    动作    环境回报  预估奖励  完成   动作概率         优势
        """
        self.policy.train()
        for i in range(self.params["train_num_epoch"]):
            experiences = self.rb.get(self.params["batch_size"])

            states = torch.stack([e[0] for e in experiences]).squeeze(1)
            old_actions = torch.stack([e[1] for e in experiences]).squeeze()
            old_action_log_probs = torch.tensor([e[5] for e in experiences], device=self.params["device"]).squeeze()
            advantages = torch.tensor([e[6] for e in experiences], device=self.params["device"], dtype=torch.float).squeeze()
            returns = torch.tensor([e[7] for e in experiences], device=self.params["device"], dtype=torch.float).squeeze()

            del experiences

            # 拿到当前网络下，以前的动作的新的log_prob
            new_values, new_action_log_probs, new_entropy = self.policy.evaluate_action(states, old_actions)
            new_values = new_values.squeeze()

            # TODO: 这个是否需要正则化？正则化advantage
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(
                new_action_log_probs - old_action_log_probs
            )

            ### actor的损失
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1-self.params["clip_range"], 1+self.params["clip_range"])
            policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

            ### critic的损失
            values_pred = new_values
            values_loss = self.critic_loss_fn(returns, values_pred)

            ### entropy的损失
            entropy_loss = - torch.mean(new_entropy)

            ### loss计算并反向传播更新
            loss = policy_loss + self.params["entropy_coef"]*entropy_loss + self.params["value_loss_coef"]*values_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.params["max_grad_norm"]
            )  # 防止梯度爆炸
            self.optimizer.step()
    
    def learn(self):
        """
        学习，对象调用的内容
        """
        for i in range(0, self.params["train_total_episodes"]):
            print(f"正在训练第{i}轮：", end="", flush=True)
            # 0. 清空经验
            self.reset_rb()
            # 1. 收集经验
            self.collect_rollouts()
            # 2. 计算优势和回报
            self.compute_advantage_and_return()
            # 3. 训练
            self.train()
            print(f"当前总胜率:{self.win_tims/(i+1)}")
            if i % self.params["save_every"] == 0:
                self.save(f"{self.params['SAVE_PATH']}epoch{i}_{self.last_n_avg_reward}.pth")
        print(f"总胜率:{self.win_tims/self.params['train_total_episodes']}, 近n场胜率: {np.mean(self.last_n_win)}")
    
    def log(self, episode_num):
        """
        记录第几个episode的日志
        episode_num: 第几个episode
        """
        if self.logger != None:
            self.logger.record_num("episode reward", self.episode_reward, episode_num)
            self.logger.record_num("last n avg reward", self.last_n_avg_reward, episode_num)
            self.logger.record_num("last n win ratio", np.mean(self.last_n_win), episode_num)

    def predict(self, x, deterministic=False):
        pass

    def save(self, path):
        """
        保存模型
        path: 完成的文件路径，包括文件的后缀名字
        """
        tmp = {
            "policy": self.policy.state_dict(),
        }
        torch.save(tmp, path)
    
    def load(self, path):
        """
        加载模型
        """
        tmp = torch.load(path)
        self.policy.load_state_dict(tmp["policy"])

class PPO_RLFC(BaseAlgorithm):
    """
    DQN的PPO的算法
    这个是我们的IDEA，从经验中学习
    """
    def __init__(self, env: Env, params: dict, policy: BaseModel, scorer: BaseModel) -> None:
        """
        env: 强化学习的环境
        params: 参数
        policy: 策略网络
        scorer: 打分器
        """
        super().__init__(env, params, policy)
        self.scorer = scorer.to(self.params["device"]).requires_grad_(False)
        self.optimizer = self.params["optimizer"](
            self.policy.parameters(), lr=self.params["lr"]
        )
        self.loss_fn = torch.nn.MSELoss()
        self.win_tims = 0
        self.last_n_win = []  # 记录最近10场胜利情况，胜利为1，失败为0
        # 判断是否需要衰减探索率
        if self.params["epsilon_decay"]:
            self.params["epsilon"] = self.params["init_epsilon"]
    
    @torch.no_grad()
    def collect_rollouts(self, episode_num):
        """
        收集一轮经验
        episode_num：第几次收集经验
        """
        self.policy.eval()
        self.episode_reward = 0

        state, info = self.env.reset()  # [1, state_dim]
        s0 = state.to(self.params["device"])  # 记录上一个状态
        s1 = s0  # 记录当前状态
        while True:
            state = state.to(self.params["device"])

            # 通过网络，得到当前状态的各个动作的价值
            action_values = self.policy.forward(x=state).squeeze()
            # 采样动作
            action_to_take = self.policy.sample_action(action_values, self.params["epsilon"])
            # 得到当前动作的预估价值
            # action_to_take_value = action_values[action_to_take]

            state_next, reward, win, die, _ = self.env.step(action=action_to_take)

            ### 计算分数
            state_next = state_next.to(self.params["device"])
            s2 = state_next
            # state_next就是s2
            # 根据前一个状态，现在的状态，下一个状态，得到与专家经验相符的分数
            tmp = torch.cat([s0, s1, s2], dim=1)
            score = self.scorer.forward(tmp).item()*self.params["scorer_eps"]
            
            done = win or die

            ### 记录内容
            # 保存的分数应该多一个score，作为他的额外奖励
            self.cache(state, action_to_take, reward+score, state_next)
            self.episode_reward += reward  # 但是画图用的折线应该用原奖励
            
            if done:
                self.episodes_reward.append(self.episode_reward)
                print(f"近n轮平均回报:{self.last_n_avg_reward}, 本轮回报：{self.episode_reward}; ", end="")
                if win:
                    self.last_n_win.append(1)
                    self.win_tims += 1
                else:
                    self.last_n_win.append(0)
                
                ### 每回合记录日志
                self.last_n_win = self.last_n_win[-self.params["last_n_avg_rewards"]:]
                self.logger.record_num("episode reward", self.episode_reward, episode_num)
                self.logger.record_num("last n avg reward", self.last_n_avg_reward, episode_num)
                self.logger.record_num("last n win ratio", np.mean(self.last_n_win), episode_num)
                return
            # fi done

            # 向后推进一个时间步
            s0 = s1
            s1 = state_next
            state = state_next.to(self.params["device"])

    def compute_advantage_and_return(self):
        """
        根据收集到的经验，计算回报
        0        1              2       3
        state action_to_take reward, state_next)
        状态    动作          环境奖励   下一个状态
        现在这个算法是收集一整轮然后计算，所以可以直接倒着计算环境奖励的累计
        """
        for idx in reversed(range(1, self.rb.size)):
            # 下面这个步骤计算完之后，reward就变成return了
            self.rb.buffer[idx-1][2] = self.rb.buffer[idx-1][2] + self.params["gamma"] * self.rb.buffer[idx][2]
    
    def train(self):
        """
        根据收集到的经验，训练
        0        1              2       3
        state action_to_take reward, state_next)
        状态    动作          环境奖励   下一个状态
        """
        self.policy.train()
        for _ in range(self.params["train_num_epoch"]):
            experiences = self.rb.get(self.params["batch_size"])

            states = torch.stack([e[0] for e in experiences]).squeeze(1)
            old_actions = torch.tensor([e[1] for e in experiences], device=self.params["device"]).squeeze()
            returns = torch.tensor([e[2] for e in experiences], device=self.params["device"], dtype=torch.float)
            # next_states = torch.stack([e[3] for e in experiences]).squeeze(1)

            # 拿到当前网络下，这些状态会执行什么动作
            action_values = self.policy.forward(states)
            # 拿到之前执行的动作的当前的value
            pred_values = action_values[range(len(experiences)), old_actions]

            ### 计算损失
            loss = self.loss_fn(returns, pred_values)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.params["max_grad_norm"]
            )  # 防止梯度爆炸
            self.optimizer.step()
    
    def learn(self):
        """
        学习，对象调用的内容
        """
        for i in range(0, self.params["train_total_episodes"]):
            print(f"正在训练第{i}轮：", end="", flush=True)
            # 0. 清空经验
            self.reset_rb()
            # 1. 收集经验
            self.collect_rollouts(i)
            # 2. 计算优势和回报
            self.compute_advantage_and_return()
            # 3. 训练
            self.train()
            ### 训练一轮衰减一次探索率
            if self.params["epsilon_decay"]:
                self.params["epsilon"] = self.params["epsilon"] * self.params["decay_ratio"]
                self.params["epsilon"] = max(self.params["epsilon"], self.params["min_epsilon"])

            print(f"当前总胜率:{self.win_tims/(i+1)}, 最近n场胜率：{np.mean(self.last_n_win)}")
        print(f"总胜率:{self.win_tims/self.params['train_total_episodes']}")


class DQN(BaseAlgorithm):
    """
    DQN的算法，不同于另一个程序中的DQN策略
    """
    def __init__(self, env: Env, params: dict, policy: BaseModel) -> None:
        """
        env: 环境
        params: 一个字典，是config文件中的配置文件信息中的字典
        policy: 我们的策略网络，可能是AC，可能是Q网络，但是这里是Q网络
        """
        super().__init__(env, params, policy)
        self.loss_fn = torch.nn.MSELoss()
        self.win_tims = 0
        self.run_episode = 0  # 运行了多少个回合了
        self.run_steps = 0  # 运行了多少步了
        self._last_obs = None  # 记录上一个obs
    
    def compute_advantage_and_return(self):
        """
        根据收集到的经验，计算优势和回报
        这个使用的是原本的collect rollouts
        0       1          2        3           4    5
        state, ori_action, reward, state_next, done, info
        """
        pass

    @torch.no_grad()
    def collect_rollouts(self):
        """
        这里重新写收集。因为这里不设置最大步数，所以应该手动设置每x步训练一次
        params里有一个参数：train_freq，这个是收集步骤数
        """
        # 设定为评估模式
        self.policy.eval()

        if self._last_obs == None:
            # 重置环境
            self._last_obs, info = self.env.reset()  # [1, state_dim]
            self._last_obs = self._last_obs.to(self.params["device"])
        while True:
            self.run_steps += 1
            # 通过网络，得到当前状态的各个动作的价值
            # 计算动作，由于这个网络产生的是argmax产生得一个值，所以ori_action和action_to_take应该是一样的
            ori_action, action_to_take = self.predict(self._last_obs)
            # action_values = self.policy.forward(x=state).squeeze()
            # 采样动作
            # action_to_take = self.policy.sample_action(action_values, self.params["epsilon"])
            # 得到当前动作的预估价值
            # action_to_take_value = action_values[action_to_take]

            # 在环境中采取这个步骤
            state_next, reward, win, die, info = self.env.step(action_to_take)
            done = win or die

            ### 记录内容，动作应该记录产生的，而不是采取的
            self.cache(self._last_obs, ori_action, reward, state_next, done, info)
            self.episode_reward += reward
            
            if done:  # 每次环境结束时才记录日志
                self.run_episode += 1
                self.episodes_reward.append(self.episode_reward)

                # 记录胜利记录
                if win:
                    self.last_n_win.append(1)
                    self.win_tims += 1
                else:
                    self.last_n_win.append(0)
                self.last_n_win = self.last_n_win[-self.params["last_n_avg_rewards"]:]
                
                ### 每回合记录日志
                self.log(self.run_episode)
                # self.logger.record_num("episode reward", self.episode_reward, episode_num)
                # self.logger.record_num("last n avg reward", self.last_n_avg_reward, episode_num)
                # self.logger.record_num("last n win ratio", np.mean(self.last_n_win), episode_num)
                print(f"近n轮平均回报:{self.last_n_avg_reward}, 本轮回报：{self.episode_reward};")

                self._last_obs, info = self.env.reset()  # [1, state_dim]
                self._last_obs = self._last_obs.to(self.params["device"])
                self.episode_reward = 0
                # return
            else:  # 如果结束了，记录一次运行的日志，并重置环境，如果没结束，只是拿到下一个时刻的状态
                self._last_obs = state_next.to(self.params["device"])
            # fi done

            if self.run_steps >= self.params["learning_starts"]:
                if self.run_steps % self.params["train_freq"] == 0:
                    # 说明不应该收集了，break
                    break
            if self.params["total_timesteps"] <= self.run_steps:  # 结束了
                return "done"
        
    def train(self):
        """
        根据收集到的经验，计算优势和回报
        这个使用的是原本的collect rollouts
        0       1          2        3           4    5
        state, ori_action, reward, state_next, done, info
        """
        self.policy.train()

        for _ in range(self.params["train_num_epochs"]):
            experiences = self.rb.get(self.params["batch_size"])

            states = torch.stack([e[0] for e in experiences]).squeeze(1).to(self.params["device"])
            states_next = torch.stack([e[3] for e in experiences]).squeeze(1).to(self.params["device"])
            old_actions = torch.tensor([e[1] for e in experiences], device=self.params["device"]).squeeze()
            rewards = torch.tensor([e[2] for e in experiences], device=self.params["device"], dtype=torch.float)
            dones = torch.tensor([e[4] for e in experiences], device=self.params["device"], dtype=torch.float)

            # 需要先计算一下，下一个状态下最大的价值下标，即认为下一步是贪心选择的动作
            with torch.no_grad():
                next_values, next_values_max_idx = self.policy(states_next).max(dim=1)  # 找到最大的动作对应的价值
                # 计算目标的价值
                target_q_values = rewards + (1-dones)*self.params["gamma"]*next_values

            # 拿到当前网络下，这些状态的预估价值
            action_values = self.policy.forward(states)
            # 拿到之前执行的动作的当前的value，好像gather就能实现下面的内容，到时候试一试
            # pred_values = torch.gather(action_values, dim=1, index=old_actions.long())
            pred_values = action_values[range(len(experiences)), old_actions]

            ### 计算损失
            loss = self.loss_fn(target_q_values, pred_values)

            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.params["max_grad_norm"]
            )  # 防止梯度爆炸
            self.policy_optimizer.step()
    
    def predict(self, x, deterministic=False):
        """
        通过自己的网络预测下一步的动作
        """
        # 得到网络的输出
        dist = self.policy.forward(x).squeeze()

        # 从输出中采样动作
        action_to_take = self.policy.sample_action(dist, deterministic)
        return action_to_take, action_to_take
    
    def log(self, episode_num):
        """
        """
        if self.logger != None:
            self.logger.record_num("episode reward", self.episode_reward, episode_num)
            self.logger.record_num("last n avg reward", self.last_n_avg_reward, episode_num)
            self.logger.record_num("last n win ratio", np.mean(self.last_n_win), episode_num)

    def learn(self):
        """
        学习
        total_timestep: 总共训练的时间步数量
        """
        # for i in range(0, self.params["train_total_episodes"]):
        while True:
            # print(f"正在训练第{i}轮：", end="", flush=True)
            # 0. 清空经验
            # TODO: 普通的DQN不清空经验
            self.reset_rb()
            # 1. 收集经验
            tmp = self.collect_rollouts()

            # 2. 计算优势和回报
            self.compute_advantage_and_return()
            # 3. 训练
            self.train()
            print(f"当前总步数：{self.run_steps}, 当前总胜率:{self.win_tims/(self.run_episode+1)}, 最近n场胜率：{np.mean(self.last_n_win)}, 当前探索率：{self.params['epsilon']}")

            # 设定探索率衰减
            if self.params["epsilon_decay"]:
                self.params["epsilon"] = self.params["epsilon"] * self.params["decay_ratio"]
                self.params["epsilon"] = max(self.params["epsilon"], self.params["min_epsilon"])
            if tmp == "done":
                break

        print(f"当前总步数：{self.run_steps}, 总胜率:{self.win_tims/self.run_episode}")

    def reset_rb(self):
        """
        在DQN中不需重置，设定了30w的上限
        """
        pass
    

class DQN_RLFC(BaseAlgorithm):
    """
    DQN的算法，加上我的idea
    """
    def __init__(self, env: Env, params: dict, policy: BaseModel, scorer: BaseModel) -> None:
        """
        env: 环境
        params: 一个字典，是config文件中的配置文件信息中的字典
        policy: 我们的策略网络，可能是AC，可能是Q网络，但是这里是Q网络
        """
        super().__init__(env, params, policy)
        self.scorer = scorer.to(self.params["device"]).requires_grad_(False)
        self.loss_fn = torch.nn.MSELoss()
        self.win_tims = 0
        self.run_episode = 0  # 运行了多少个回合了
        self.run_steps = 0  # 运行了多少步了
        self._last_obs = None  # 记录上一个obs
        self.s1 = None
        self.s2 = None
        self.s3 = None
    
    def compute_advantage_and_return(self):
        """
        根据收集到的经验，计算优势和回报
        这个使用的是原本的collect rollouts
        0       1          2        3           4    5
        state, ori_action, reward, state_next, done, info
        """
        pass

    @torch.no_grad()
    def collect_rollouts(self):
        """
        这里重新写收集。因为这里不设置最大步数，所以应该手动设置每x步训练一次
        params里有一个参数：train_freq，这个是收集步骤数
        """
        # 设定为评估模式
        self.policy.eval()

        if self._last_obs == None:
            # 重置环境
            self._last_obs, info = self.env.reset()  # [1, state_dim]
            self._last_obs = self._last_obs.to(self.params["device"])
            self.s1 = torch.zeros_like(self._last_obs)
            self.s2 = self._last_obs  # 最初的时候，s1为全0
        while True:
            self.run_steps += 1
            # 通过网络，得到当前状态的各个动作的价值
            # 计算动作，由于这个网络产生的是argmax产生得一个值，所以ori_action和action_to_take应该是一样的
            ori_action, action_to_take = self.predict(self._last_obs)
            # action_values = self.policy.forward(x=state).squeeze()
            # 采样动作
            # action_to_take = self.policy.sample_action(action_values, self.params["epsilon"])
            # 得到当前动作的预估价值
            # action_to_take_value = action_values[action_to_take]

            # 在环境中采取这个步骤
            state_next, reward, win, die, info = self.env.step(action_to_take)
            self.s3 = state_next.to(self.params["device"])
            done = win or die
            
            ### 计算打分器中这个多少分
            tmp = torch.cat([self.s1, self.s2, self.s3], dim=1)
            score = self.scorer.forward(tmp)*self.params["scorer_eps"]

            ### 记录内容，动作应该记录产生的，而不是采取的
            # 存储的奖励也是通过我们的score正则过的奖励
            # 不需要对score单独储存了，因为我们储存的reward已经包含了score了
            self.cache(self._last_obs, ori_action, reward+score, state_next, done, info)
            self.episode_reward += reward
            
            if done:  # 每次环境结束时才记录日志
                self.run_episode += 1
                self.episodes_reward.append(self.episode_reward)

                # 记录胜利记录
                if win:
                    self.last_n_win.append(1)
                    self.win_tims += 1
                else:
                    self.last_n_win.append(0)
                self.last_n_win = self.last_n_win[-self.params["last_n_avg_rewards"]:]
                
                ### 每回合记录日志
                self.log(self.run_episode)
                # self.logger.record_num("episode reward", self.episode_reward, episode_num)
                # self.logger.record_num("last n avg reward", self.last_n_avg_reward, episode_num)
                # self.logger.record_num("last n win ratio", np.mean(self.last_n_win), episode_num)
                print(f"近n轮平均回报:{self.last_n_avg_reward}, 本轮回报：{self.episode_reward};")

                self._last_obs, info = self.env.reset()  # [1, state_dim]
                self._last_obs = self._last_obs.to(self.params["device"])
                self.episode_reward = 0
                # return
            else:  # 如果结束了，记录一次运行的日志，并重置环境，如果没结束，只是拿到下一个时刻的状态
                self._last_obs = state_next.to(self.params["device"])
            # fi done

            if self.run_steps >= self.params["learning_starts"]:
                if self.run_steps % self.params["train_freq"] == 0:
                    # 说明不应该收集了，break
                    break
            if self.params["total_timesteps"] <= self.run_steps:  # 结束了
                return "done"
        
    def train(self):
        """
        根据收集到的经验，计算优势和回报
        这个使用的是原本的collect rollouts
        0       1          2               3           4    5
        state, ori_action, reward+score, state_next, done, info
        """
        self.policy.train()

        for _ in range(self.params["train_num_epochs"]):
            experiences = self.rb.get(self.params["batch_size"])

            states = torch.stack([e[0] for e in experiences]).squeeze(1).to(self.params["device"])
            states_next = torch.stack([e[3] for e in experiences]).squeeze(1).to(self.params["device"])
            old_actions = torch.tensor([e[1] for e in experiences], device=self.params["device"]).squeeze()
            rewards = torch.tensor([e[2] for e in experiences], device=self.params["device"], dtype=torch.float)
            dones = torch.tensor([e[4] for e in experiences], device=self.params["device"], dtype=torch.float)

            # 需要先计算一下，下一个状态下最大的价值下标，即认为下一步是贪心选择的动作
            with torch.no_grad():
                next_values, _ = self.policy(states_next).max(dim=1)  # 找到最大的动作对应的价值
                # 计算目标的价值
                target_q_values = rewards + (1-dones)*self.params["gamma"]*next_values

            # 拿到当前网络下，这些状态的预估价值
            action_values = self.policy.forward(states)
            # 拿到之前执行的动作的当前的value，好像gather就能实现下面的内容，到时候试一试
            # pred_values = torch.gather(action_values, dim=1, index=old_actions.long())
            pred_values = action_values[range(len(experiences)), old_actions]

            ### 计算损失
            loss = self.loss_fn(target_q_values, pred_values)

            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.params["max_grad_norm"]
            )  # 防止梯度爆炸
            self.policy_optimizer.step()
    
    def predict(self, x, deterministic=False):
        """
        通过自己的网络预测下一步的动作
        """
        # 得到网络的输出
        dist = self.policy.forward(x).squeeze()

        # 从输出中采样动作
        action_to_take = self.policy.sample_action(dist, deterministic)
        return action_to_take, action_to_take
    
    def log(self, episode_num):
        """
        """
        if self.logger != None:
            self.logger.record_num("episode reward", self.episode_reward, episode_num)
            self.logger.record_num("last n avg reward", self.last_n_avg_reward, episode_num)
            self.logger.record_num("last n win ratio", np.mean(self.last_n_win), episode_num)

    def learn(self):
        """
        学习
        total_timestep: 总共训练的时间步数量
        """
        # for i in range(0, self.params["train_total_episodes"]):
        while True:
            # print(f"正在训练第{i}轮：", end="", flush=True)
            # 0. 清空经验
            # TODO: 普通的DQN不清空经验
            self.reset_rb()
            # 1. 收集经验
            tmp = self.collect_rollouts()

            # 2. 计算优势和回报
            self.compute_advantage_and_return()
            # 3. 训练
            self.train()
            print(f"当前总步数：{self.run_steps}, 当前总胜率:{self.win_tims/(self.run_episode+1)}, 最近n场胜率：{np.mean(self.last_n_win)}, 当前探索率：{self.params['epsilon']}")

            # 设定探索率衰减
            if self.params["epsilon_decay"]:
                self.params["epsilon"] = self.params["epsilon"] * self.params["decay_ratio"]
                self.params["epsilon"] = max(self.params["epsilon"], self.params["min_epsilon"])
            if tmp == "done":
                break

        print(f"当前总步数：{self.run_steps}, 总胜率:{self.win_tims/self.run_episode}")

    def reset_rb(self):
        """
        在DQN中不需重置，设定了30w的上限
        """
        pass
    def save(self, path):
        """
        保存模型
        path: 完成的文件路径，包括文件的后缀名字
        """
        tmp = {
            "policy": self.policy.state_dict(),
        }
        torch.save(tmp, path)
    
    def load(self, path):
        """
        加载模型
        """
        tmp = torch.load(path)
        self.policy.load_state_dict(tmp["policy"])
