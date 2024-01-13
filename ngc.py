import torch

def init_gaussian_dense(dims, stddev, device):
    return torch.empty(dims, requires_grad=False, device=device).normal_(mean=0.0, std=stddev)

class NGC:
    def __init__(self, L, dims, weight_stddev, beta=0.1, gamma=0.001, err_update_coeff=0.95, fn_phi_name='relu', device=None):
        assert len(dims) == L + 1
        self.L = L
        self.dims = dims
        self.beta = beta
        self.gamma = gamma # leak coefficient
        self.err_update_coeff = err_update_coeff

        self.device = torch.device('cpu') if device is None else device

        # self.W = ([init_gaussian_dense([dim_hid, dim_inp], weight_stddev, self.device)]
        #     + [init_gaussian_dense([dim_hid, dim_hid], weight_stddev, self.device) for _ in range(L-2)]
        #     + [init_gaussian_dense([dim_top, dim_hid], weight_stddev, self.device)])

        # self.E = ([init_gaussian_dense([dim_inp, dim_hid], weight_stddev, self.device)]
        #     + [init_gaussian_dense([dim_hid, dim_hid], weight_stddev, self.device) for _ in range(L-1)])

        # assume dims is in order (input, ... hiddens ..., top)
        self.W = []
        for i in range(L):
            self.W.append(init_gaussian_dense([dims[i+1], dims[i]], weight_stddev, self.device))

        self.E = []
        for i in range(L):
            self.E.append(init_gaussian_dense([dims[i], dims[i+1]], weight_stddev, self.device))


        if fn_phi_name == 'relu':
            self.fn_phi = torch.relu
        else:
            raise NotImplementedError("Only relu is supported for phi.")

        self.fn_g = lambda x: x

        self.clip_weights()


    def parameters(self):
        return self.W + self.E

    def state_dict(self):
        state = {}
        for l in range(self.L):
            state[f'W{l}'] = self.W[l]
            state[f'E{l}'] = self.E[l]
        return state

    def load_state_dict(self, state):
        for l in range(self.L):
            self.W[l] = state[f'W{l}']
            self.E[l] = state[f'E{l}']


    def project(self, x_top):
        zbar = x_top
        # assume g is identity
        for i in range(self.L - 1, -1, -1):
            zbar = self.fn_phi(zbar) @ self.W[i]
        return zbar

    def infer(self, x_top, x_bot, K=50):
        batch_size = x_bot.shape[0]
        z = [x_bot]
        e = [x_bot - 0.]
        for l in range(1, self.L):
            z.append(torch.zeros([batch_size, self.dims[l]], device=self.device))
            e.append(torch.zeros([batch_size, self.dims[l]], device=self.device))
        z.append(x_top)
        e.append(torch.zeros([batch_size, self.dims[self.L]], device=self.device))

        mu = [None for _ in range(self.L)]

        for _ in range(K):
            for i in range(1, self.L + 1):
                di = e[i-1] @ self.E[i-1] - e[i]
                z[i] += self.beta * (-self.gamma * z[i] + di)

            mu_W_input = self.fn_phi(z[1]) @ self.W[0]
            mu[0] = self.fn_g(mu_W_input)
            e[0] = z[0] - mu[0]
            for i in range(1, self.L):
                mu[i] = self.fn_g(self.fn_phi(z[i+1]) @ self.W[i])
                e[i] = self.fn_phi(z[i]) - mu[i]

        self.z = z
        self.e = e

        return mu[0]

    def calc_updates(self):
        batch_size = self.z[0].shape[0]
        avg_factor = -1.0 / (batch_size)

        for l in range(0, self.L):
            dWl = self.fn_phi(self.z[l+1]).T @ self.e[l]
            dWl = avg_factor * dWl
            dEl = self.err_update_coeff * dWl.T
            self.W[l].grad = dWl
            self.E[l].grad = dEl


    def clip_weights(self):
        # clip column norms to 1
        for l in range(self.L):
            Wl_col_norms = self.W[l].norm(dim=0, keepdim=True)
            self.W[l].copy_(self.W[l] / torch.maximum(Wl_col_norms, torch.tensor(1.0)))
            El_col_norms = self.E[l].norm(dim=0, keepdim=True)
            self.E[l].copy_(self.E[l] / torch.maximum(El_col_norms, torch.tensor(1.0)))


    def calc_total_discrepancy(self):
        return sum([torch.sum(e**2) for e in self.e[:self.L]])