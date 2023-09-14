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
from .models import BaseModel, DQN
from .replaybuffer import ReplayBuffer
from .logger import Logger

from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """
    本文件所有的算法都会继承这个类，作为基础算法
    """
    def __init__(self, env: Env, params: dict) -> None:
        """
        env: 环境
        params: 一个字典，是config文件中的配置文件信息中的字典
        """
        super().__init__()
        self.env = env
        self.params = params
        self.rb = ReplayBuffer(params["buffer_size"])

        if "LOGS_PATH" in self.params and self.params["LOGS_PATH"] != "":
            self.logger = Logger(params["LOGS_PATH"])
        else:
            self.logger = None

        self.episode_reward = 0
        self.episodes_reward = []
    
    @abstractmethod
    def collect_rollouts(self):
        """
        收集经验，在这里每次收集一轮，而不是固定步数（因为这个任务太简单了）
        """
        raise f"在Base Algorithm中未实现这个方法 collect_rollouts，请在继承类中重写"
    
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
    def learn(self, total_timestep):
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
    
class VanillaPPO_dqn(BaseAlgorithm):
    """
    DQN的PPO的算法
    """
    def __init__(self, env: Env, params: dict, policy: DQN) -> None:
        super().__init__(env, params)
        self.policy = policy.to(self.params["device"])
        self.optimizer = self.params["optimizer"](
            self.policy.parameters(), lr=self.params["lr"]
        )
        self.loss_fn = torch.nn.MSELoss()
        self.win_tims = 0
        self.last_n_win = []  # 记录最近10场胜利情况，胜利为1，失败为0
    
    @torch.no_grad()
    def collect_rollouts(self, episode_num):
        """
        收集一轮经验
        episode_num：第几次收集经验
        """
        self.policy.eval()
        self.episode_reward = 0

        state, info = self.env.reset()  # [1, state_dim]
        while True:
            state = state.to(self.params["device"])

            # 通过网络，得到当前状态的各个动作的价值
            action_values = self.policy.forward(x=state).squeeze()
            # 采样动作
            action_to_take = self.policy.sample_action(action_values, self.params["epsilon"])
            # 得到当前动作的预估价值
            # action_to_take_value = action_values[action_to_take]

            state_next, reward, win, die, _ = self.env.step(action=action_to_take)
            done = win or die

            ### 记录内容
            self.cache(state, action_to_take, reward, state_next)
            self.episode_reward += reward
            
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
            print(f"当前总胜率:{self.win_tims/(i+1)}, 最近n场胜率：{np.mean(self.last_n_win)}")
        print(f"总胜率:{self.win_tims/self.params['train_total_episodes']}")

class VanillaPPO(BaseAlgorithm):
    """
    普通的PPO的算法
    """
    def __init__(self, env: Env, params: dict, policy: BaseNetwork) -> None:
        super().__init__(env, params)
        self.policy = policy.to(self.params["device"])
        self.optimizer = self.params["optimizer"](
            self.policy.parameters(), lr=self.params["lr"]
        )
        self.critic_loss_fn = torch.nn.MSELoss()
        self.win_tims = 0
    
    @torch.no_grad()
    def collect_rollouts(self):
        """
        收集一轮经验
        """
        self.policy.eval()
        self.episode_reward = 0

        state, info = self.env.reset()  # [1, state_dim]
        while True:
            state = state.to(self.params["device"])

            action, value, action_log_prob = self.policy.forward(x=state)

            action_to_take = action.cpu().item()  # 离散动作就是这样的
            state_next, reward, win, die, _ = self.env.step(action=action_to_take)
            done = win or die

            ### 记录内容
            self.cache(state, action, reward, value, done, action_log_prob, 0)
            self.episode_reward += reward
            
            if done:
                self.episodes_reward.append(self.episode_reward)
                print(f"近n轮平均回报:{self.last_n_avg_reward}, 本轮回报：{self.episode_reward}")
                if win:
                    self.win_tims += 1
                return

            state = state_next.to(self.params["device"])

    def compute_advantage_and_return(self):
        """
        根据收集到的经验，计算优势和回报
        0        1      2       3      4      5               6
        state, action, reward, value, done, action_log_prob,   0
        状态    动作    环境奖励  预估奖励  完成   动作概率         优势
        现在这个算法是收集一整轮然后计算，所以可以直接倒着计算环境奖励的累计
        """
        for idx in reversed(range(1, self.rb.size)):
            # 若总长度是10,那么下标就是0 - 9, range(1, 10) >> 1, 2, ..., 9
            # reverse之后，就是 9, 8, ..., 1
            # 可以计算每个地方的环境累计奖励是   reward[idx-1] = reward[idx-1] + gamma*reward[idx]
            # 若gamma=0.9
            # -0.1, -0.1, -0.1, -0.1, 10 >>> -0.1, -0.1, -0.1, 8.9, 10 >> ...
            # 下面这个步骤计算完之后，reward就变成return了
            self.rb.buffer[idx-1][2] = self.rb.buffer[idx-1][2] + self.params["gamma"] * self.rb.buffer[idx][2]

            # 下面计算advantage，如果从环境得到的立即回报 - 网络预估回报，说明这个动作值得执行，那么就可以执行
            self.rb.buffer[idx][6] = self.rb.buffer[idx][2] - self.rb.buffer[idx][3]
    
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
            returns = torch.tensor([e[2] for e in experiences], device=self.params["device"], dtype=torch.float).squeeze()
            advantages = torch.tensor([e[6] for e in experiences], device=self.params["device"], dtype=torch.float).squeeze()

            del experiences

            # 拿到当前网络下，以前的动作的新的log_prob
            new_values, new_action_log_probs, new_entropy = self.policy.evaluate_actions(states, old_actions)
            new_values = new_values.squeeze()

            # TODO: 这个是否需要正则化？正则化advantage
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

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
        print(f"总胜率:{self.win_tims/self.params['train_total_episodes']}")

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
        super().__init__(env, params)
        self.policy = policy.to(self.params["device"])
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