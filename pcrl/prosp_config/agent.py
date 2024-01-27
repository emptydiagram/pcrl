from ..utils import get_activation_fn, get_activation_fn_deriv, init_gaussian

from operator import itemgetter
import random

import torch

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


class ProspConfigNet:
    def __init__(self, config, device):
        L, T_max, dims, fn_act, settle_init_type, relax_step_size, use_relax_early_stopping, weight_init = \
            itemgetter('L', 'T_max', 'dims', 'fn_act', 'settle_init_type', 'relax_step_size', 'use_relax_early_stopping', 'weight_init')(config)
        assert len(dims) == L + 1
        self.L = L
        self.T_max = T_max
        self.device = device
        self.dims = dims
        fn_act = fn_act.lower()
        self.fn_act = get_activation_fn(fn_act)
        self.fn_act_deriv = get_activation_fn_deriv(fn_act)
        self.settle_init_type = settle_init_type
        self.relax_step_size = relax_step_size
        self.use_relax_early_stopping = use_relax_early_stopping
        self.weight_init = weight_init

        if weight_init['type'] == 'gaussian_fixed_stddev':
            weight_stddev = weight_init['stddev']
            make_weights = lambda d1, d2: init_gaussian([d1, d2], weight_stddev, self.device)
        else:
            raise NotImplementedError(f"Weight initialization type {weight_init['type']} not recognized.")

        self.W = [make_weights(dims[l+1], dims[l]) for l in range(L)]

    def parameters(self):
        return self.W

    def project(self, x_in):
        z = x_in
        zs = []
        for l in range(self.L - 1, -1, -1):
            z = self.fn_act(z) @ self.W[l]
            zs.append(z)
        zs.reverse()
        return zs


    def settle(self, x_in, x_out):
        # using NGC convention here, where the top layer, z^L, is the input, and z^0 is the output
        # the prospective configuration paper uses the opposite convention
        batch_size = x_in.shape[0]
        z = [x_out]
        if self.settle_init_type == 'zero':
            for l in range(1, self.L):
                z.append(torch.zeros([batch_size, self.dims[l]], device=self.device))
        elif self.settle_init_type == 'projection':
            proj_z = self.project(x_in)
            for l in range(self.L - 1):
                z.append(proj_z[l])
        else:
            raise NotImplementedError(f"Settle initialization type {self.settle_init_type} not recognized.")
        z.append(x_in)

        a = [None for _ in range(len(z))]
        e = [None for _ in range(self.L)]

        # using row-vector convention (1-d vectors are row vectors), so
        #  - z^L = x_in is (B x d_L)
        #  - z^l, p^l, e^l are (B x d_l)
        #  - z^0 = x_out is (B x d_0)

        # for l = 0, ..., L-2:
        #   p^l = phi(z^{l+1}) W^{l+1}
        # p^{L-1} = z^L W^L

        # e^l = z^l - p^l
        # E^l = (1/2) * ||e^l||^2
        # E = Σ_l E^l = (1/2) Σ_l ||e^l||^2

        # also using numerator (Jacobian) convention:
        # d E^l / dz^l = e^l
        # d E^{l-1} / dz^l = -phi'(z^l) * e^{l-1} (W^l)^T
        # dE/dz^l = d(E^l + E^{l-1})/dz^l
        # Δz^l = -λ d(E^l + E^{l-1}) / dz^l
        # Δz^l = -λ e^l + λ phi'(z^l) * e^{l-1} (W^l)^T)

        lambda_ = self.relax_step_size
        num_step_size_reductions = 0
        max_num_ss_reductions = 2

        E = None
        for t in range(self.T_max):
            # prediction errors
            for l in range(0, self.L-1):
                a[l+1] = self.fn_act(z[l+1])
                pl = a[l+1] @ self.W[l]
                e[l] = z[l] - pl
            # use identity activation function for the input
            a[self.L] = z[self.L]
            pl = a[self.L] @ self.W[self.L - 1]
            e[self.L - 1] = z[self.L - 1] - pl

            # relaxation step
            for l in range(1, self.L):
                z[l] -= lambda_ * (e[l] - self.fn_act_deriv(z[l], a[l]) * (e[l-1] @ self.W[l-1].T))

            if self.use_relax_early_stopping:
                prev_E = E
                E = sum([e[l] * e[l] for l in range(self.L)])
                if prev_E is not None and E >= prev_E:
                    lambda_ *= 0.5
                    max_num_ss_reductions += 1
                    if num_step_size_reductions < max_num_ss_reductions:
                        print(f"Halving relaxation step size at t = {t}")
                    else:
                        print(f"Early stopping relaxation at t = {t}")
                        break

        self.a = a
        self.e = e

    def calc_weight_updates(self):
        # dE/dW^l = dE^{l-1}/dW^l = -phi(z^l)^T * e^{l-1}
        for l in range(self.L):
            self.W[l].grad = -self.a[l+1].T @ self.e[l]


class ProspConfigAgent:
    def __init__(self, config, device):
        batch_size, dims, discount_factor, eps_init, eps_anneal_step, eps_final, num_learn_epochs, optimizer_type, optimizer_lr, replay_buffer_capacity, replay_buffer_learn_thresh = \
            itemgetter('batch_size', 'dims', 'discount_factor', 'eps_init', 'eps_anneal_step', 'eps_final', 'num_learn_epochs',
                       'optimizer_type', 'optimizer_lr', 'replay_buffer_capacity', 'replay_buffer_learn_thresh')(config)
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

        q_net_config_keys = ['L', 'T_max', 'dims', 'fn_act', 'settle_init_type', 'relax_step_size', 'use_relax_early_stopping', 'weight_init']
        q_net_config = {key: config[key] for key in q_net_config_keys}

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.q_net = ProspConfigNet(q_net_config, device)

        if optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.q_net.parameters(), lr=optimizer_lr)
        else:
            raise NotImplementedError(f"Optimizer type {optimizer_type} not recognized.")


    def act(self, obs):
        p = random.random()
        if p < self.eps:
            return random.randint(0, self.num_actions - 1)
        else:
            q_out = self.q_net.project(obs)[0]
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
            acts = torch.vstack(acts).to(self.device)
            rewards = torch.tensor(rewards, device=self.device).unsqueeze_(1)
            obses_next = torch.vstack(obses_next).to(self.device)
            is_dones = torch.tensor(is_dones, dtype=torch.float32, device=self.device).unsqueeze_(1)
            estimated_next_rewards = self.q_net.project(obses_next)[0]
            max_next_rewards = torch.max(estimated_next_rewards, dim=1).values.unsqueeze_(1)
            target_reward_scalars = rewards + self.discount_factor * (1 - is_dones) * max_next_rewards
            target_rewards = target_reward_scalars * acts
            self.q_net.settle(obses, target_rewards)
            self.q_net.calc_weight_updates()
            self.optimizer.step()
