# Overview

Attempted PyTorch implementation of Active Neural Generative Coding (ANGC) from [Backprop-Free Reinforcement Learning with Active Neural Generative Coding](https://arxiv.org/abs/2107.07046). This builds on Neural Generative Coding (NGC) from [The neural coding framework for learning generative models](https://www.nature.com/articles/s41467-022-29632-7) and applies it to RL problems.

# Variables

My notation differs in spots from the paper. One major difference is that I use row-major convention (1-d vectors are row vectors instead of column vectors), so that I don't have to worry about transposing things when going from math to implementation.


| Variable | Description |
| --- | --- |
| $L + 1$ | number of layers |
| $T$ | number of interence time steps (paper calls this $K$) |
| $\gamma_v$ | inference update leak coefficient |
| $\beta_e$ | prediction error coefficient |
| $\beta$ | inference state update coefficient |
| $\gamma_e$ | error weight update coefficient. "controls the time-scale at which the error synapses are adjusted (usually values in the range of [0.9, 1.0] are used) |
| $J_\ell$ | dimensionality of layer $\ell$ |
| $z^\ell$, $\forall \ell = 1, \ldots, K-1$ | hidden layer state vectors. $z^\ell \in \mathbb{R}^{1 \times J_\ell}$ |
| $z^0$ | bottom sensory vector, clamped to sensory input $z^0 = x^o$ |
| $z^K$ | top sensory vector, clamped to sensory input $z^K = x^i$ |
| $W^{\ell} \in \mathbb{R}^{J_{\ell} \times J_{\ell - 1}}$ | (top-down) prediction weights for layer $\ell$, also called the generative weights |
| $E^{\ell} \in \mathbb{R}^{J_{\ell-1} \times J_{\ell}}$ | (bottom-up) error weights for layer $\ell$ |
| $\phi^\ell$ | activation function for layer $\ell$ |
| $g^\ell$ | another activation function for layer $\ell$. the paper says they use the identity function $g^\ell = \text{id}[J_\ell]$ |
| $\bar{z}^\ell$ | top-down prediction vector of $z^\ell$ |
| $e^\ell$ | prediction error vector for layer $\ell$ |
| $d^\ell$ | bottom-up + top-down inference pressure |
| $M_W^\ell \in \mathbb{R}^{J_{\ell+1} \times J_{\ell}}$ | generative weights modulation matrix for layer $\ell$ |
| $M_E^\ell \in \mathbb{R}^{J_{\ell} \times J_{\ell+1}}$ | error weights modulation matrix for layer $\ell$ |


# Inference

There's an inference phase that iterates, for timesteps $t = 1, \dots, T$:

$$\bar{z}^\ell(t) = g^{\ell}(\phi^{\ell + 1}[z^{\ell + 1}(t)] \cdot W^{\ell + 1})$$

$$e^\ell(t) = \frac{1}{2 \beta_e} (\phi^\ell[z^\ell(t)] - \bar{z}^\ell(t))$$

$$d^\ell(t) = -e^{\ell}(t) + e^{\ell-1}(t) E^{\ell}$$

$$z^\ell(t) = z^\ell(t-1) + \beta (- \gamma_v z^{\ell}(t) + d^{\ell}(t))$$

(where the last equation is slightly simplified to exclude the NGC lateral term, which is not used in ANGC):

TODO pseudocode

# Update

$$\Delta W^\ell = \phi^\ell[z^{\ell + 1}]^\top e^{\ell} \odot M_W^\ell$$

$$\Delta E^\ell = \gamma_e (\Delta W^{\ell})^\top \odot M_E^\ell$$
