from operator import itemgetter
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class ANGCAgent:
    def __init__(self, config):
        num_actions, bottom_size, top_size, num_hiddens, \
        W_gen_hidden_size, E_gen_hidden_size, W_cont_hidden_size, E_cont_hidden_size, \
        imp_epistemic, imp_instrumental, \
        phi_name, \
        eps_init, eps_decay, eps_min = \
            itemgetter(
                'num_actions', 'bottom_size', 'top_size', 'num_hiddens',
                'W_gen_hidden_size', 'E_gen_hidden_size', 'W_cont_hidden_size', 'E_cont_hidden_size',
                'imp_epistemic', 'imp_instrumental',
                'phi_name',
                'eps_init', 'eps_decay', 'eps_min')(config)

        self.num_actions = num_actions
        self.bottom_size = bottom_size
        self.num_layers = num_hiddens + 2
        self.eps = eps_init
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        # generator params
        self.W_gen = ([self.init_weights((W_gen_hidden_size, bottom_size))]
            + [self.init_weights((W_gen_hidden_size, W_gen_hidden_size)) for _ in range(num_hiddens - 1)]
            + [self.init_weights((top_size, W_gen_hidden_size))])

        self.E_gen = ([self.init_weights((bottom_size, E_gen_hidden_size))]
            + [self.init_weights((E_gen_hidden_size, E_gen_hidden_size)) for _ in range(num_hiddens - 1)]
            + [self.init_weights((E_gen_hidden_size, top_size))])

        # controller params
        self.W_cont = ([self.init_weights((W_cont_hidden_size, num_actions))]
            + [self.init_weights((W_cont_hidden_size, W_cont_hidden_size)) for _ in range(num_hiddens - 1)]
            + [self.init_weights((top_size, W_gen_hidden_size))])

        self.E_cont = ([self.init_weights((num_actions, E_cont_hidden_size))]
            + [self.init_weights((E_cont_hidden_size, E_cont_hidden_size)) for _ in range(num_hiddens - 1)]
            + [self.init_weights((E_cont_hidden_size, top_size))])

        if phi_name == 'relu':
            self.phi = nn.ReLU()
        elif phi_name == 'relu6':
            self.phi = nn.ReLU6()
        else:
            raise Exception(f'phi_name {phi_name} not supported')

        self.z = [torch.empty((1,1)) for _ in range(self.num_layers)]


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

    env = gym.make(env_name)
    obs, info = env.reset(seed=seed)
    num_actions = env.action_space.n
    print(num_actions)
    # print(env.observation_space.shape)

    angc_agent = ANGCAgent(agent_config)

    for episode in range(num_episodes):
        print(f"Episode: {episode}")
        obs, info = env.reset(seed=seed)

        for _ in range(num_max_steps):
            action = angc_agent.act(obs)
            obs_next, reward_instrumental, terminated, truncated, _info = env.step(action)

            print(obs, reward_instrumental, terminated or truncated)
            if terminated or truncated:
                obs, info = env.reset()
                print(obs)



if __name__ == '__main__':
    env_name = 'CartPole-v1'
    agent_config = {
        'num_actions': 2,
        'bottom_size': 4,
        'top_size': 2,
        'num_hiddens': 2,
        'W_gen_hidden_size': 128,
        'E_gen_hidden_size': 128,
        'W_cont_hidden_size': 128,
        'E_cont_hidden_size': 128,
        'phi_name': 'relu',
        'eps_init': 1.0,
        'eps_decay': 0.97,
        'eps_min': 0.05
    }
    run_angc(env_name, agent_config)