from ngc import NGC

import datetime
from operator import itemgetter
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, config, device, tb_writer=None):
        num_actions, obs_dim, gen_hidden_sizes, cont_hidden_sizes, \
        K, beta, infer_leak, beta_e, gamma_e, \
        imp_epistemic, imp_instrumental, \
        phi_name, \
        eps_init, eps_decay, eps_min, \
        replay_buffer_size, replay_sample_batch_size, \
        discount_factor, \
        optimizer_gen, lr_gen, optimizer_cont, lr_cont = \
            itemgetter(
                'num_actions', 'obs_dim', 'gen_hidden_sizes', 'cont_hidden_sizes',
                'K', 'beta', 'infer_leak', 'beta_e', 'gamma_e',
                'imp_epistemic', 'imp_instrumental',
                'phi_name',
                'eps_init', 'eps_decay', 'eps_min',
                'replay_buffer_size', 'replay_sample_batch_size',
                'discount_factor',
                'optimizer_gen', 'lr_gen', 'optimizer_cont', 'lr_cont'
            )(config)

        assert len(gen_hidden_sizes) == len(cont_hidden_sizes)
        num_hiddens = len(gen_hidden_sizes)
        self.device = device
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.L = num_hiddens + 1
        self.gen_hidden_sizes = gen_hidden_sizes
        self.cont_hidden_sizes = cont_hidden_sizes
        self.K = K
        self.beta = beta
        self.infer_leak = infer_leak
        self.beta_e = beta_e # TODO: use this
        self.gamma_e = gamma_e
        self.imp_epistemic = imp_epistemic
        self.imp_instrumental = imp_instrumental
        self.eps = eps_init
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.replay_sample_batch_size = replay_sample_batch_size
        self.discount_factor = discount_factor

        gen_top_size = num_actions + obs_dim

        ngc_gen_config = {
            'L': self.L,
            'dims': [obs_dim] + gen_hidden_sizes + [gen_top_size],
            'weight_stddev': 0.025,
            'beta': beta,
            'beta_e': beta_e,
            'gamma': infer_leak,
            'err_update_coeff': gamma_e,
            'fn_phi_name': 'relu',
        }
        # gen: (a_t, o_t) -> predicted o_{t+1}
        self.ngc_gen = NGC(ngc_gen_config, device=device)

        ngc_cont_config = {
            'L': self.L,
            'dims': [num_actions] + cont_hidden_sizes + [obs_dim],
            'weight_stddev': 0.025,
            'beta': beta,
            'beta_e': beta_e,
            'gamma': infer_leak,
            'err_update_coeff': gamma_e,
            'fn_phi_name': 'relu',
        }
        # cont: (o_t) -> predicted reward for each action
        self.ngc_cont = NGC(ngc_cont_config, device=device)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # perform gradient ascent
        if optimizer_gen == 'adam':
            self.optimizer_gen = torch.optim.Adam(self.ngc_gen.parameters(), lr=lr_gen, maximize=False)
        else:
            raise Exception(f'optimizer_gen {optimizer_gen} not supported')

        if optimizer_cont == 'rmsprop':
            self.optimizer_cont = torch.optim.RMSprop(self.ngc_cont.parameters(), lr=lr_cont, maximize=False)
        elif optimizer_cont == 'adam':
            self.optimizer_cont = torch.optim.Adam(self.ngc_cont.parameters(), lr=lr_cont, maximize=False)
        else:
            raise Exception(f'optimizer_cont {optimizer_cont} not supported')

        self.tb_writer = tb_writer


    def act(self, obs):
        # generate random probability
        p = random.random()

        if p < self.eps:
            action = random.randint(0,self.num_actions-1)
        else:
            rew_pred = self.ngc_cont.project(obs)
            action = torch.argmax(rew_pred).item()

        return action


    def infer_and_update(self, x_top, x_bot, circuit_type, perform_write=False, c_eps=1e-6):
        assert circuit_type in ['cont', 'gen']
        if circuit_type == 'gen':
            # W = self.W_gen
            # E = self.E_gen
            optimizer = self.optimizer_gen
            ngc_circuit = self.ngc_gen
        else:
            # W = self.W_cont
            # E = self.E_cont
            optimizer = self.optimizer_cont
            ngc_circuit = self.ngc_cont

        ngc_circuit.infer(x_bot, x_top)
        optimizer.zero_grad()
        ngc_circuit.calc_updates()
        optimizer.step()
        ngc_circuit.normalize_weights()


        # z, e = self.infer(x_top, x_bot, circuit_type)

        # if perform_write and self.tb_writer is not None:
        #     if not(hasattr(self, 'update_step')):
        #         self.update_step = 0

        #     for l in range(len(z)):
        #         self.tb_writer.add_scalar(f'update-step/{circuit_type}/z-{l}', torch.mean(z[l]).item(), self.update_step)

        #     for l in range(len(e)):
        #         self.tb_writer.add_scalar(f'update-step/{circuit_type}/e-{l}', torch.mean(e[l]).item(), self.update_step)

        #     self.update_step += 1

        # update
        # for l in range(self.L):
        #     optimizer.zero_grad
        #     # TODO: modulation matrices
        #     dWl = self.phi(z[l+1]).T @ e[l]
        #     dWl = dWl / (torch.norm(dWl) + c_eps)
        #     W[l].grad = dWl


        #     if l < self.L - 1:
        #         dEl = self.gamma_e * dWl.T
        #         dEl = dEl / (torch.norm(dEl) + c_eps)
        #         E[l].grad = dEl

        #     optimizer.step()

        #     W[l].copy_(2 * W[l] / (torch.norm(W[l]) + c_eps))

        #     if l < self.L - 1:
        #         E[l].copy_(2 * E[l] / (torch.norm(E[l]) + c_eps))


    def experience_replay_update(self):
        if len(self.replay_buffer) < self.replay_sample_batch_size:
            # wait until we have enough replay samples for a batch
            return

        # sample mini batch from replay buffer and call infer_and_update for both controller and generator
        batch = self.replay_buffer.sample(self.replay_sample_batch_size)

        obs, actions, rewards, obs_next, is_terminals = zip(*batch)

        obs = torch.vstack(obs).to(self.device)
        actions = torch.vstack(actions).to(self.device)
        rewards = torch.vstack(rewards).to(self.device)
        obs_next = torch.vstack(obs_next).to(self.device)
        is_terminals = torch.tensor(is_terminals, dtype=torch.float32).unsqueeze_(1).to(self.device)
        obs_next_project = self.ngc_cont.project(obs_next)
        max_future_reward = torch.max(obs_next_project, dim=1, keepdim=True).values
        target_scalars = rewards * is_terminals + (1 - is_terminals) * (rewards + self.discount_factor * max_future_reward)

        obs_project = self.ngc_cont.project(obs)

        targets = target_scalars * actions + (1 - actions) * obs_project
        self.infer_and_update(obs, targets, 'cont', perform_write=False)
        actions_obs = torch.hstack((actions, obs)).to(self.device)
        self.infer_and_update(actions_obs, obs_next, 'gen', perform_write=False)

    def update_epsilon(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_min)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_angc(trial_name, env_name, agent_config):
    write_to_tensorboard = True

    seed = 628318
    set_seed(seed)

    num_episodes = 200
    num_max_steps = 500
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    env = gym.make(env_name)
    obs, info = env.reset(seed=seed)
    num_actions = env.action_space.n

    if write_to_tensorboard:
        curr_dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(f"runs/expt-{curr_dt}-{trial_name}")
    else:
        writer = None

    angc_agent = ANGCAgent(agent_config, device, writer)

    epistemic_reward_max = 1

    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        obs, info = env.reset(seed=seed)
        obs = torch.tensor(obs, device=device).unsqueeze_(0)

        ep_inst_reward = 0
        ep_total_reward = 0

        for t in range(num_max_steps):
            action = angc_agent.act(obs)
            obs_next, reward_instr, terminated, truncated, _info = env.step(action)
            ep_inst_reward += reward_instr
            action_oh = F.one_hot(torch.tensor(action), num_actions).unsqueeze_(0).to(device)
            obs_next = torch.tensor(obs_next, device=device).unsqueeze_(0)
            act_obs = torch.cat((action_oh, obs), dim=1).to(device)
            angc_agent.ngc_gen.infer(obs_next, act_obs)

            # TODO: move into ANGC agent?
            # reward_epist = sum([torch.norm(el)**2 for el in gen_errors])
            reward_epist = sum([torch.norm(el)**2 for el in angc_agent.ngc_gen.e])
            epistemic_reward_max = max(epistemic_reward_max, reward_epist)
            reward_epist /= epistemic_reward_max
            reward = angc_agent.imp_epistemic * reward_epist + angc_agent.imp_instrumental * reward_instr

            ep_total_reward += reward

            is_terminal = terminated or truncated
            angc_agent.replay_buffer.add((obs, action_oh, reward, obs_next, is_terminal))

            angc_agent.experience_replay_update()

            # print(obs, reward_instrumental, reward_epistemic.item(), reward.item())
            if terminated or truncated:
                print(f"Resetting on step {t=}: {terminated=} {truncated=}")
                obs, info = env.reset()
                obs = torch.tensor(obs, device=device).unsqueeze_(0)
                if write_to_tensorboard:
                    writer.add_scalar('episode/reward', ep_inst_reward, episode)
                break

        print(f"Episode {episode} total reward: {ep_total_reward}, instrumental reward: {ep_inst_reward}")

        angc_agent.update_epsilon()

    if write_to_tensorboard:
        writer.close()

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
        'gen_hidden_sizes': [256, 128],
        'cont_hidden_sizes': [256, 128],
        'K': 20,
        'beta': 0.1,
        'infer_leak': 0.001,
        'beta_e': 0.5,
        'gamma_e': 0.95,
        'imp_epistemic': 1.0,
        'imp_instrumental': 1.0,
        'phi_name': 'relu',
        'eps_init': 1.0,
        'eps_decay': 0.97,
        'eps_min': 0.05,
        'replay_buffer_size': 500000,
        'replay_sample_batch_size': 256,
        'discount_factor': 0.99,
        'optimizer_gen': 'adam',
        'lr_gen': 0.001,
        'optimizer_cont': 'rmsprop',
        'lr_cont': 0.0005,
    }
    run_angc('angc/asc-beta_e=0.1', env_name, agent_config)
