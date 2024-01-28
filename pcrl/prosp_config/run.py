from ..utils import set_seed
from .agent import ProspConfigAgent

from operator import itemgetter

import gymnasium as gym
import torch
import torch.nn.functional as F


def run_pc(run_config, agent_config):
    device_name, env_name, num_episodes, seed, trial_name = itemgetter('device_name', 'env_name', 'num_episodes', 'seed', 'trial_name')(run_config)

    set_seed(seed)
    device = torch.device(device_name)

    env = gym.make(env_name)

    agent = ProspConfigAgent(agent_config, device)

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
            act_enc = F.one_hot(torch.tensor(act), num_classes=env.action_space.n).float().to(device)
            obs_next, reward, is_terminated, is_truncated, info = env.step(act)
            obs_next = torch.tensor(obs_next, device=device).unsqueeze_(0)
            score += reward
            is_done = is_terminated or is_truncated
            agent.save_transition((obs, act_enc, reward, obs_next, is_done))
            obs = obs_next
            if is_done:
                break

        agent.learn()

        if (ep + 1) % score_period == 0:
            print(f"Episode {ep+1}: average reward = {score / score_period}, ε = {(agent.eps * 100):.2f}%")
            score = 0.



if __name__ == '__main__':
    pc_run_config = {
        'device_name': 'cuda' if torch.cuda.is_available() else 'cpu',
        'env_name': 'CartPole-v1',
        'num_episodes': 2000,
        'seed': 299792458,
        'trial_name': 'prosp_conf/try-1',
    }
    num_actions = 2
    obs_dim = 4
    pc_agent_config = {
        'L': 2,
        'T_max': 32,
        'batch_size': 60,
        'dims': [num_actions, 128, obs_dim],
        'discount_factor': 0.98,
        'eps_init': 0.08,
        'eps_anneal_step': 5e-5,
        'eps_final': 0.01,
        'fn_act': 'sigmoid',
        'num_learn_epochs': 10,
        'optimizer_type': 'sgd',
        'optimizer_lr': 0.001,
        'replay_buffer_capacity': 50000,
        'replay_buffer_learn_thresh': 2000,
        'settle_init_type': 'projection',
        'relax_step_size': 0.05,
        'use_relax_early_stopping': False,
        'weight_init': {
            'type': 'kaiming_uniform',
        }
        # 'weight_init': {
        #     'type': 'gaussian_fixed_stddev',
        #     'stddev': 0.0025,
        # }
    }
    run_pc(pc_run_config, pc_agent_config)