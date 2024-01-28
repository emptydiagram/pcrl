from ..utils import get_activation_fn, set_seed

from operator import itemgetter
import random

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

# DQN implementation as a sanity check, since right now none of my PC implementations work at all

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.next_slot = 0

    def add(self, item):
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.next_slot] = item
            self.next_slot = (self.next_slot + 1) % self.capacity

    def sample(self, num_samples):
        return random.sample(self.buffer, num_samples)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, config, device):
        L, batch_size, dims, discount_factor, eps_init, eps_anneal_step, eps_final, fn_act, num_learn_epochs, optimizer_type, optimizer_lr, replay_buffer_capacity, replay_buffer_learn_thresh = \
            itemgetter('L', 'batch_size', 'dims', 'discount_factor', 'eps_init', 'eps_anneal_step', 'eps_final', 'fn_act', 'num_learn_epochs',
                       'optimizer_type', 'optimizer_lr', 'replay_buffer_capacity', 'replay_buffer_learn_thresh')(config)
        self.L = L
        self.batch_size = batch_size
        self.device = device
        self.discount_factor = discount_factor
        self.eps_init = eps_init
        self.eps = eps_init
        self.eps_anneal_step = eps_anneal_step
        self.eps_final = eps_final
        self.num_actions = dims[0]
        self.num_learn_epochs = num_learn_epochs
        self.replay_buffer_learn_thresh = replay_buffer_learn_thresh
        fn_act = fn_act.lower()
        self.fn_act = get_activation_fn(fn_act)

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        q_linears = [
            nn.Linear(dims[l+1], dims[l], bias=True) for l in range(self.L-1, -1, -1)
        ]

        q_layers = [q_linears[0]]
        for l in range(1, self.L):
            q_layers.append(nn.ReLU())
            q_layers.append(q_linears[l])
        self.q_net = nn.Sequential(*q_layers).to(device)

        if optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.q_net.parameters(), lr=optimizer_lr, momentum=0.9)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=optimizer_lr)
        else:
            raise NotImplementedError(f"Optimizer type {optimizer_type} not recognized.")


    def act(self, obs):
        p = random.random()
        if p < self.eps:
            return random.randint(0, self.num_actions - 1)
        else:
            q_out = self.q_net(obs)
            return torch.argmax(q_out).item()

    def save_transition(self, transition):
        self.replay_buffer.add(transition)

    def update_epsilon(self, episode):
        self.eps = max(self.eps_final, self.eps_init - self.eps_anneal_step * episode)

    def learn(self):
        if len(self.replay_buffer) < self.replay_buffer_learn_thresh:
            return

        for _ in range(self.num_learn_epochs):
            batch = self.replay_buffer.sample(self.batch_size)
            obses, acts, rewards, obses_next, is_dones = zip(*batch)
            obses = torch.vstack(obses).to(self.device)
            acts = torch.tensor(acts).to(self.device).unsqueeze_(1)
            rewards = torch.tensor(rewards, device=self.device).unsqueeze_(1)
            obses_next = torch.vstack(obses_next).to(self.device)
            is_dones = torch.tensor(is_dones, dtype=torch.float32, device=self.device).unsqueeze_(1)

            estimated_next_rewards = self.q_net(obses_next)
            # print('-------------------------------')
            # print(torch.mean(estimated_next_rewards, dim=1))
            max_next_rewards = torch.max(estimated_next_rewards, dim=1).values.unsqueeze_(1)
            target_reward_scalars = rewards + self.discount_factor * (1 - is_dones) * max_next_rewards
            #target_rewards = target_reward_scalars * acts
            # print(torch.mean(target_rewards))

            # DQN learning step
            self.optimizer.zero_grad()
            q_out = self.q_net(obses)
            q_out_act = q_out.gather(1, acts)
            loss = F.mse_loss(q_out_act, target_reward_scalars)
            loss.backward()
            self.optimizer.step()



def run_dqn(run_config, agent_config):
    device_name, env_name, num_episodes, seed, trial_name = itemgetter('device_name', 'env_name', 'num_episodes', 'seed', 'trial_name')(run_config)

    set_seed(seed)
    device = torch.device(device_name)

    env = gym.make(env_name)

    agent = DQNAgent(agent_config, device)

    score = 0.
    score_period = 20

    for ep in range(num_episodes):
        if ep == 0:
            obs, _info = env.reset(seed=seed)
        else:
            obs, _info = env.reset()
        obs = torch.tensor(obs, device=device).unsqueeze_(0)

        agent.update_epsilon(ep)
        while True:
            act = agent.act(obs)
            obs_next, reward, is_terminated, is_truncated, info = env.step(act)
            obs_next = torch.tensor(obs_next, device=device).unsqueeze_(0)
            score += reward
            is_done = is_terminated or is_truncated
            agent.save_transition((obs, act, reward, obs_next, is_done))
            obs = obs_next
            if is_done:
                break

        agent.learn()

        if (ep + 1) % score_period == 0:
            print(f"Episode {ep+1}: average reward = {score / score_period}, eps = {(agent.eps * 100):.2f}%")
            score = 0.



if __name__ == '__main__':
    pc_run_config = {
        'device_name': 'cuda' if torch.cuda.is_available() else 'cpu',
        'env_name': 'CartPole-v1',
        'num_episodes': 10000,
        'seed': 299792458,
        'trial_name': 'dqn/try-1',
    }
    num_actions = 2
    obs_dim = 4
    pc_agent_config = {
        'L': 2,
        'batch_size': 32,
        'dims': [num_actions, 128, obs_dim],
        'discount_factor': 0.98,
        'eps_init': 0.08,
        'eps_anneal_step': 5e-5,
        'eps_final': 0.01,
        'fn_act': 'sigmoid',
        'num_learn_epochs': 10,
        'optimizer_type': 'adam',
        'optimizer_lr': 0.0001,
        'replay_buffer_capacity': 50000,
        'replay_buffer_learn_thresh': 2000
    }
    run_dqn(pc_run_config, pc_agent_config)