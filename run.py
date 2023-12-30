from operator import itemgetter
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ANGCAgent:
    def __init__(self, config, device):
        num_actions, obs_dim, num_hiddens, \
        W_gen_hidden_size, E_gen_hidden_size, W_cont_hidden_size, E_cont_hidden_size, \
        T_infer, beta, infer_leak, beta_e, gamma_e, \
        imp_epistemic, imp_instrumental, \
        phi_name, \
        eps_init, eps_decay, eps_min, \
        replay_buffer_size, replay_sample_batch_size, \
        discount_factor, \
        optimizer_gen, lr_gen, optimizer_cont, lr_cont = \
            itemgetter(
                'num_actions', 'obs_dim', 'num_hiddens',
                'W_gen_hidden_size', 'E_gen_hidden_size', 'W_cont_hidden_size', 'E_cont_hidden_size',
                'T_infer', 'beta', 'infer_leak', 'beta_e', 'gamma_e',
                'imp_epistemic', 'imp_instrumental',
                'phi_name',
                'eps_init', 'eps_decay', 'eps_min',
                'replay_buffer_size', 'replay_sample_batch_size',
                'discount_factor',
                'optimizer_gen', 'lr_gen', 'optimizer_cont', 'lr_cont'
            )(config)

        self.device = device
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.num_layers = num_hiddens + 2 # (L+1) in paper notation
        self.W_gen_hidden_size = W_gen_hidden_size
        self.E_gen_hidden_size = E_gen_hidden_size
        self.W_cont_hidden_size = W_cont_hidden_size
        self.E_cont_hidden_size = E_cont_hidden_size
        self.T_infer = T_infer
        self.beta = beta
        self.infer_leak = infer_leak
        self.beta_e = beta_e
        self.gamma_e = gamma_e
        self.imp_epistemic = imp_epistemic
        self.imp_instrumental = imp_instrumental
        self.eps = eps_init
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.replay_sample_batch_size = replay_sample_batch_size
        self.discount_factor = discount_factor

        gen_top_size = num_actions + obs_dim

        ### generator params ###
        # contains [W^1, ..., W^L]
        self.W_gen = ([self.init_weights((W_gen_hidden_size, obs_dim))]
            + [self.init_weights((W_gen_hidden_size, W_gen_hidden_size)) for _ in range(num_hiddens - 1)]
            + [self.init_weights((gen_top_size, W_gen_hidden_size))])

        # contains [E^1, ..., E^L]
        self.E_gen = ([self.init_weights((obs_dim, E_gen_hidden_size))]
            + [self.init_weights((E_gen_hidden_size, E_gen_hidden_size)) for _ in range(num_hiddens - 1)]
            + [self.init_weights((E_gen_hidden_size, gen_top_size))])

        ### controller params ###
        self.W_cont = ([self.init_weights((W_cont_hidden_size, num_actions))]
            + [self.init_weights((W_cont_hidden_size, W_cont_hidden_size)) for _ in range(num_hiddens - 1)]
            + [self.init_weights((obs_dim, W_cont_hidden_size))])

        self.E_cont = ([self.init_weights((num_actions, E_cont_hidden_size))]
            + [self.init_weights((E_cont_hidden_size, E_cont_hidden_size)) for _ in range(num_hiddens - 1)]
            + [self.init_weights((E_cont_hidden_size, obs_dim))])

        if phi_name == 'relu':
            self.phi = nn.ReLU()
        elif phi_name == 'relu6':
            self.phi = nn.ReLU6()
        else:
            raise Exception(f'phi_name {phi_name} not supported')

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        if optimizer_gen == 'adam':
            self.optimizer_gen = torch.optim.Adam(self.W_gen + self.E_gen, lr=lr_gen)
        else:
            raise Exception(f'optimizer_gen {optimizer_gen} not supported')

        if optimizer_cont == 'rmsprop':
            self.optimizer_cont = torch.optim.RMSprop(self.W_cont + self.E_cont, lr=lr_cont)
        elif optimizer_cont == 'adam':
            self.optimizer_cont = torch.optim.Adam(self.W_cont + self.E_cont, lr=lr_cont)
        else:
            raise Exception(f'optimizer_cont {optimizer_cont} not supported')


    def init_weights(self, shape, stddev=0.025):
        return torch.empty(shape).normal_(mean=0, std=stddev).to(self.device)

    def project(self, obs):
        zbar = obs
        # assume g is identity
        for i in range(self.num_layers - 2, -1, -1):
            zbar = self.phi(zbar) @ self.W_cont[i]
        return zbar


    def act(self, obs):
        # generate random probability
        p = random.random()

        if p < self.eps:
            action = random.randint(0,self.num_actions-1)
        else:
            rew_pred = self.project(obs)
            action = torch.argmax(rew_pred).item()

        return action

    # only called for generator circuit. see infer_and_update for generic version
    def infer(self, x_top, x_bot, circuit_type):
        assert circuit_type in ['cont', 'gen']
        if circuit_type == 'gen':
            W = self.W_gen
            E = self.E_gen
            W_hidden_size = self.W_gen_hidden_size
            E_hidden_size = self.E_gen_hidden_size
            e_top_size = self.num_actions + self.obs_dim
        else:
            W = self.W_cont
            E = self.E_cont
            W_hidden_size = self.W_cont_hidden_size
            E_hidden_size = self.E_cont_hidden_size
            e_top_size = self.obs_dim

        # init
        z = [x_bot]
        e = [x_bot - 0.]
        for i in range(self.num_layers - 2):
            z.append(torch.zeros((1, W_hidden_size)).to(self.device))
            e.append(torch.zeros((1, E_hidden_size)).to(self.device))
        z.append(x_top)
        e.append(torch.zeros((1, e_top_size)).to(self.device))

        for k in range(self.T_infer):
            # for l = 1, ..., L
            for l in range(1, self.num_layers):
                z[l] += self.beta * (- self.infer_leak * z[l] - e[l] + e[l-1] @ E[l-1])

            # for l = L - 1, ..., 0
            for l in range(self.num_layers - 2, -1, -1):
                zbar = self.phi(z[l+1]) @ W[l]
                e[l] = (self.phi(z[l]) - zbar) / (2 * self.beta_e)

        return z, e[:self.num_layers-1]

    def infer_and_update(self, x_top, x_bot, circuit_type, c_eps=1e-6):
        assert circuit_type in ['cont', 'gen']
        if circuit_type == 'gen':
            W = self.W_gen
            E = self.E_gen
            optimizer = self.optimizer_gen
        else:
            W = self.W_cont
            E = self.E_cont
            optimizer = self.optimizer_cont

        z, e = self.infer(x_top, x_bot, circuit_type)

        # update
        for l in range(self.num_layers - 1):
            optimizer.zero_grad
            # TODO: modulation matrices
            dWl = self.phi(z[l+1]).T @ e[l]
            dWl = dWl / (torch.norm(dWl) + c_eps)
            dEl = self.gamma_e * dWl.T
            dEl = dEl / (torch.norm(dEl) + c_eps)
            W[l].grad = dWl
            E[l].grad = dEl
            optimizer.step()


    def experience_replay_update(self):
        if len(self.replay_buffer) < self.replay_sample_batch_size:
            # wait until we have enough replay samples for a batch
            return

        # sample mini batch from replay buffer and call infer_and_update for both controller and generator
        batch = self.replay_buffer.sample(self.replay_sample_batch_size)
        for (obs, action, reward, obs_next, is_terminal) in batch:
            target_scalar = reward if is_terminal else reward + self.discount_factor * torch.max(self.project(obs_next)).item()
            action_oh = F.one_hot(torch.tensor(action), self.num_actions).unsqueeze_(0).to(self.device)
            target = target_scalar * action_oh + (1 - action_oh) * self.project(obs)

            self.infer_and_update(obs, target, 'cont')

            act_obs = torch.cat((action_oh, obs), dim=1).to(self.device)
            self.infer_and_update(act_obs, obs_next, 'gen')

    def update_epsilon(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_min)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_angc(env_name, agent_config):
    seed = 628318
    set_seed(seed)

    num_episodes = 200
    num_max_steps = 500
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    env = gym.make(env_name)
    obs, info = env.reset(seed=seed)
    num_actions = env.action_space.n

    angc_agent = ANGCAgent(agent_config, device)

    writer = SummaryWriter()
    epistemic_reward_max = 1


    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        obs, info = env.reset(seed=seed)
        obs = torch.tensor(obs, device=device).unsqueeze_(0)

        ep_inst_reward = 0

        for t in range(num_max_steps):
            action = angc_agent.act(obs)
            obs_next, reward_instrumental, terminated, truncated, _info = env.step(action)
            ep_inst_reward += reward_instrumental
            action_oh = F.one_hot(torch.tensor(action), num_actions).unsqueeze_(0).to(device)
            obs_next = torch.tensor(obs_next, device=device).unsqueeze_(0)
            act_obs = torch.cat((action_oh, obs), dim=1).to(device)
            gen_states, gen_errors = angc_agent.infer(act_obs, obs_next, 'gen')

            # TODO: move into ANGC agent?
            reward_epistemic = sum([torch.norm(el)**2 for el in gen_errors])
            epistemic_reward_max = max(epistemic_reward_max, reward_epistemic)
            reward_epistemic /= epistemic_reward_max
            reward = angc_agent.imp_epistemic * reward_epistemic + angc_agent.imp_instrumental * reward_instrumental

            is_terminal = terminated or truncated
            angc_agent.replay_buffer.add((obs, action, reward, obs_next, is_terminal))

            angc_agent.experience_replay_update()

            # print(obs, reward_instrumental, reward_epistemic.item(), reward.item())
            if terminated or truncated:
                print(f"Resetting on step {t=}: {terminated=} {truncated=}")
                obs, info = env.reset()
                obs = torch.tensor(obs, device=device).unsqueeze_(0)
                break

        angc_agent.update_epsilon()



if __name__ == '__main__':
    # TODO:
    #  - it's unclear what T_infer should be. paper says "we found values K = 10 and K = 20 to be sufficient)"
    #    but it is unclear why two values are listed and what exactly that means. different K values for generator
    #    and controller? or different K values for different problems?
    # - paper doesnt list a value for:
    #    - beta. ngc-learn seems to use ~0.1
    #    - gamma_v (infer_leak). ngc-learn seems to use very small values (~0.001)
    #    - beta_e. no equivalent in ngc-learn. have to guess. TODO: tune
    env_name = 'CartPole-v1'
    agent_config = {
        'num_actions': 2,
        'obs_dim': 4,
        'num_hiddens': 2,
        'W_gen_hidden_size': 128,
        'E_gen_hidden_size': 128,
        'W_cont_hidden_size': 128,
        'E_cont_hidden_size': 128,
        'T_infer': 15,
        'beta': 0.1,
        'infer_leak': 0.001,
        'beta_e': 0.2,
        'gamma_e': 0.95,
        'imp_epistemic': 1.0,
        'imp_instrumental': 1.0,
        'phi_name': 'relu',
        'eps_init': 1.0,
        'eps_decay': 0.97,
        'eps_min': 0.05,
        'replay_buffer_size': 100000,
        'replay_sample_batch_size': 256,
        'discount_factor': 0.99,
        'optimizer_gen': 'adam',
        'lr_gen': 0.001,
        'optimizer_cont': 'rmsprop',
        'lr_cont': 0.0005,
    }
    run_angc(env_name, agent_config)