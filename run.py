from operator import itemgetter
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ANGCAgent:
    def __init__(self, config):
        num_actions, obs_dim, num_hiddens, \
        W_gen_hidden_size, E_gen_hidden_size, W_cont_hidden_size, E_cont_hidden_size, \
        T_infer, beta, infer_leak, beta_e, \
        imp_epistemic, imp_instrumental, \
        phi_name, \
        eps_init, eps_decay, eps_min = \
            itemgetter(
                'num_actions', 'obs_dim', 'num_hiddens',
                'W_gen_hidden_size', 'E_gen_hidden_size', 'W_cont_hidden_size', 'E_cont_hidden_size',
                'T_infer', 'beta', 'infer_leak', 'beta_e',
                'imp_epistemic', 'imp_instrumental',
                'phi_name',
                'eps_init', 'eps_decay', 'eps_min')(config)

        self.num_actions = num_actions
        self.num_layers = num_hiddens + 2 # (L+1) in paper notation
        self.W_gen_hidden_size = W_gen_hidden_size
        self.E_gen_hidden_size = E_gen_hidden_size
        self.T_infer = T_infer
        self.beta = beta
        self.infer_leak = infer_leak
        self.beta_e = beta_e
        self.imp_epistemic = imp_epistemic
        self.imp_instrumental = imp_instrumental
        self.eps = eps_init
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        gen_bot_size = num_actions + obs_dim

        ### generator params ###
        # contains [W^1, ..., W^L]
        self.W_gen = ([self.init_weights((W_gen_hidden_size, gen_bot_size))]
            + [self.init_weights((W_gen_hidden_size, W_gen_hidden_size)) for _ in range(num_hiddens - 1)]
            + [self.init_weights((obs_dim, W_gen_hidden_size))])

        # contains [E^1, ..., E^L]
        self.E_gen = ([self.init_weights((gen_bot_size, E_gen_hidden_size))]
            + [self.init_weights((E_gen_hidden_size, E_gen_hidden_size)) for _ in range(num_hiddens - 1)]
            + [self.init_weights((E_gen_hidden_size, obs_dim))])

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

        # self.z = [torch.empty((1,1)) for _ in range(self.num_layers)]


    def init_weights(self, shape, stddev=0.025):
        return torch.empty(shape).normal_(mean=0, std=stddev)

    def project(self, obs):
        print(type(obs))
        zbar = obs
        # assume g is identity
        for i in range(self.num_layers - 1, -1, -1):
            zbar = self.phi(zbar) @ self.W_gen[i]
        return zbar


    def act(self, obs):
        # generate random probability
        p = random.random()

        if p < self.eps:
            action = random.randint(0,self.num_actions-1)
        else:
            rew_pred = self.project(obs)
            action = torch.argmax(rew_pred)

        return action

    # only called for generator circuit. see infer_and_update for generic version
    def infer_gen(self, x_bot, x_top):
        # init
        z = [x_bot]
        e = [x_bot - 0.]
        for i in range(self.num_layers - 2):
            z.append(torch.zeros((1, self.W_gen_hidden_size)))
            e.append(torch.zeros((1, self.E_gen_hidden_size)))
        z.append(x_top)
        e.append(torch.zeros((1, self.top_size)))

        for k in range(self.T_infer):
            # for l = 1, ..., L
            for l in range(1, self.num_layers):
                z[l] += self.beta * (- self.infer_leak * z[l] - e[l] + self.E_gen[l-1] @ e[l-1])

            # for l = L - 1, ..., 0
            for l in range(self.num_layers - 1, -1, -1):
                zbar = self.phi(z[l+1]) @ self.W_gen[l]
                e[l] = (self.phi(z[l]) - zbar) / (2 * self.beta_e)

        return z, e[:self.num_layers-1]

    def infer_and_update(self):
        pass

    def update_epsilon(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_min)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def run_angc(env_name, agent_config):
    seed = 628318
    set_seed(seed)
    num_episodes=1
    num_max_steps = 200
    device_name = 'gpu' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    env = gym.make(env_name)
    obs, info = env.reset(seed=seed)
    num_actions = env.action_space.n

    angc_agent = ANGCAgent(agent_config)
    angc_agent.to(device)

    epistemic_reward_max = 1

    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        obs, info = env.reset(seed=seed)

        for t in range(num_max_steps):
            action = angc_agent.act(obs)
            obs_next, reward_instrumental, terminated, truncated, _info = env.step(action)
            action_oh = F.one_hot(torch.tensor(action), num_actions)
            gen_states, gen_errors = angc_agent.infer_gen(torch.cat((action_oh, obs)), obs_next)

            # TODO: move into ANGC agent?
            reward_epistemic = sum([torch.norm(el)**2 for el in gen_errors])
            epistemic_reward_max = max(epistemic_reward_max, reward_epistemic)
            reward_epistemic /= epistemic_reward_max
            reward = angc_agent.imp_epistemic * reward_epistemic + angc_agent.imp_instrumental * reward_instrumental

            print(obs, reward_instrumental, reward_epistemic, reward)
            if terminated or truncated:
                print(f"Resetting on step {t=}")
                obs, info = env.reset()
                print(obs)



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
        'T_infer': 10,
        'beta': 0.1,
        'infer_leak': 0.001,
        'beta_e': 0.2,
        'phi_name': 'relu',
        'eps_init': 1.0,
        'eps_decay': 0.97,
        'eps_min': 0.05
    }
    run_angc(env_name, agent_config)