from operator import itemgetter

import torch

def init_gaussian(dims, stddev, device):
    return torch.empty(dims, requires_grad=False, device=device).normal_(mean=0.0, std=stddev)


def calc_modulation(W):
    mhat = torch.sum(W, dim=0, keepdim=True)
    m = torch.minimum(2 * mhat / torch.max(mhat), torch.tensor(1.0))
    return m.repeat(W.shape[0], 1)

class NGC:
    def __init__(self, config, device=None):
        L, dims, weight_stddev, beta, beta_e, gamma, err_update_coeff, fn_phi_name = itemgetter(
            'L', 'dims', 'weight_stddev', 'beta', 'beta_e', 'gamma', 'err_update_coeff', 'fn_phi_name')(config)
        assert len(dims) == L + 1
        self.L = L
        self.dims = dims
        self.beta = beta
        self.beta_e = beta_e
        self.gamma = gamma # leak coefficient
        self.err_update_coeff = err_update_coeff

        self.device = torch.device('cpu') if device is None else device

        # assume dims is in order (bottom, ... hiddens ..., top)
        self.W = []
        for i in range(L):
            self.W.append(init_gaussian([dims[i+1], dims[i]], weight_stddev, self.device))

        # the paper shows E^L, but this would only be required if we updated z^L, which I think we don't
        # (x is clamped to it, it appears to be an error in the paper)
        self.E = []
        for i in range(L-1):
            self.E.append(init_gaussian([dims[i], dims[i+1]], weight_stddev, self.device))

        if fn_phi_name == 'relu':
            self.fn_phi = torch.relu
        else:
            raise NotImplementedError("Only relu is supported for phi.")

        self.fn_g = lambda x: x

        self.normalize_weights()


    def parameters(self):
        return self.W + self.E

    def state_dict(self):
        state = {}
        for l in range(self.L - 1):
            state[f'W{l}'] = self.W[l]
            state[f'E{l}'] = self.E[l]
        state[f'W{self.L - 1}'] = self.W[self.L - 1]
        return state

    def load_state_dict(self, state):
        for l in range(self.L - 1):
            self.W[l] = state[f'W{l}']
            self.E[l] = state[f'E{l}']
        self.W[self.L - 1] = state[f'W{self.L - 1}']


    def project(self, x_top):
        zbar = x_top
        for i in range(self.L - 1, -1, -1):
            zbar = self.fn_g(self.fn_phi(zbar) @ self.W[i])
        return torch.softmax(zbar, dim=1)

    def infer(self, x_bot, x_top, K=50):
        batch_size = x_bot.shape[0]
        z = [x_bot]
        e = [x_bot - 0.]
        for l in range(1, self.L):
            z.append(torch.zeros([batch_size, self.dims[l]], device=self.device))
            e.append(torch.zeros([batch_size, self.dims[l]], device=self.device))
        z.append(x_top)

        z_out = [None for _ in range(self.L+1)]
        mu = [None for _ in range(self.L)]

        for _ in range(K):
            # d^1 = -e^1 + E^1 e^0
            # z^1 = z^1 + beta * (-gamma_v z^1 + d^1)
            # ...
            # d^{L-1} = -e^{L-1} + E^L e^{L-1}
            # z^{L-1} = z^{L-1} + beta * (-gamma_v z^{L-1} + d^{L-1})
            for i in range(1, self.L):
                di = -e[i] + e[i-1] @ self.E[i-1]
                z[i] += self.beta * (-self.gamma * z[i] + di)
                z_out[i] = self.fn_phi(z[i])
            z_out[self.L] = self.fn_phi(z[self.L])

            mu[0] = self.fn_g(z_out[1] @ self.W[0])
            e[0] = (z[0] - mu[0]) / (2.0 * self.beta_e)
            for i in range(1, self.L):
                mu[i] = self.fn_g(z_out[i+1] @ self.W[i])
                e[i] = (z_out[i] - mu[i]) / (2.0 * self.beta_e)


        self.z = z
        self.z_out = z_out
        self.e = e

        return mu[0]

    def calc_updates(self, c_eps=1e-6):
        batch_size = self.z[0].shape[0]

        for l in range(0, self.L):
            dWl = self.z_out[l+1].T @ self.e[l]
            dWl = dWl / (dWl.norm() + c_eps)
            self.W[l].grad = -1.0 * dWl * calc_modulation(self.W[l])
            if l < self.L - 1:
                dEl = self.err_update_coeff * dWl.T
                dEl = dEl / (dEl.norm() + c_eps)
                self.E[l].grad = -1.0 * dEl * calc_modulation(self.E[l])


    # the weight clipping function from ngc-learn
    def clip_weights(self):
        # clip column norms to 1
        for l in range(self.L - 1):
            Wl_col_norms = self.W[l].norm(dim=0, keepdim=True)
            self.W[l].copy_(self.W[l] / torch.maximum(Wl_col_norms, torch.tensor(1.0)))
            El_col_norms = self.E[l].norm(dim=0, keepdim=True)
            self.E[l].copy_(self.E[l] / torch.maximum(El_col_norms, torch.tensor(1.0)))

        Wl_col_norms = self.W[self.L-1].norm(dim=0, keepdim=True)
        self.W[self.L-1].copy_(self.W[self.L-1] / torch.maximum(Wl_col_norms, torch.tensor(1.0)))

    # implements W^l = 2 W^l / ||W^l|| + c_eps
    def normalize_weights(self, c_eps=1e-6):
        for l in range(self.L - 1):
            self.W[l].copy_(2.0 * self.W[l] / (self.W[l].norm() + c_eps))
            self.E[l].copy_(2.0 * self.E[l] / (self.E[l].norm() + c_eps))
        self.W[self.L - 1].copy_(2.0 * self.W[self.L - 1] / (self.W[self.L - 1].norm() + c_eps))


    def calc_total_discrepancy(self):
        return sum([torch.sum(e**2) for e in self.e[:self.L]])