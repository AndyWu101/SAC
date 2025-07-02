import torch
import torch.nn.functional as F


from config import args
from model import Actor , Critic
from replay_buffer import ReplayBuffer


class SAC:

    def __init__(self, action_dim):

        self.alpha = torch.tensor(args.alpha, dtype=torch.float32, device=args.device)
        self.gamma = torch.tensor(args.gamma, dtype=torch.float32, device=args.device)
        self.tau = torch.tensor(args.tau, dtype=torch.float32, device=args.device)

        if args.adaptive_alpha == True:
            self.target_entropy = torch.tensor(-action_dim, dtype=torch.float32, device=args.device)
            self.log_alpha = torch.tensor(0, requires_grad=True, dtype=torch.float32, device=args.device)
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.critic_learning_rate)
            self.alpha = self.log_alpha.detach().exp()


    def train(
            self,
            actor: Actor,
            critic: Critic,
            critic_target: Critic,
            replay_buffer: ReplayBuffer,
            actor_optimizer: torch.optim.Adam,
            critic_optimizer: torch.optim.Adam
    ):

        replays = replay_buffer.sample()

        states = torch.stack([replay.state for replay in replays])
        actions = torch.stack([replay.action for replay in replays])
        rewards = torch.stack([replay.reward for replay in replays])
        next_states = torch.stack([replay.next_state for replay in replays])
        not_dones = torch.stack([replay.not_done for replay in replays])


        # 計算 target_Q
        with torch.no_grad():

            next_actions, next_log_prob_pis = actor.sample(next_states)

            next_Q1s , next_Q2s = critic_target(next_states, next_actions)
            next_Qs = torch.min(next_Q1s, next_Q2s) - self.alpha * next_log_prob_pis
            target_Qs = rewards + not_dones * self.gamma * next_Qs

        # 計算 Q1 , Q2
        Q1s , Q2s = critic(states, actions)

        # MSE loss
        critic_loss = F.mse_loss(Q1s, target_Qs) + F.mse_loss(Q2s, target_Qs)

        # 反向傳播
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # 更新 target network
        with torch.no_grad():
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



        # 訓練 actor
        actor_actions, log_prob_pis = actor.sample(states)

        Q1s , Q2s = critic(states, actor_actions)
        Qs = torch.min(Q1s, Q2s)

        actor_loss = (self.alpha * log_prob_pis - Qs).mean()

        # 反向傳播
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()



        # 更新 log_alpha
        if args.adaptive_alpha == True:

            log_alpha_loss = -self.log_alpha * (log_prob_pis.detach().mean() + self.target_entropy)

            self.log_alpha_optimizer.zero_grad()
            log_alpha_loss.backward()
            self.log_alpha_optimizer.step()

            self.alpha = self.log_alpha.detach().exp()


