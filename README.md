# (WIP) implementation of Active Neural Generative Coding in PyTorch

TODO

# Variables

My notation differs in spots from the paper. One major difference is that I use row-major convention (1-d vectors are row vectors instead of column vectors), so that I don't have to worry about transposing things when going from math to implementation.


| Variable | Description |
| --- | --- |
| $K + 1$ | number of layers (different from the paper, which uses $L+1$) |
| $J_\ell$ | dimensionality of layer $\ell$ | 
| $z^\ell$, $\forall \ell = 1, \ldots, K-1$ | hidden layer state vectors. $z^\ell \in \mathbb{R}^{1 \times J_\ell}$ |
| $z^0$ | bottom sensory vector, clamped to sensory input $z^0 = x^o$ |
| $z^K$ | top sensory vector, clamped to sensory input $z^K = x^i$ |
| $W^{\ell} \in \mathbb{R}^{J_{\ell+1} \times J_{\ell}}$ | top-down prediction weights |
| $\phi^\ell$ | activation function for layer $\ell$ |
| $g^\ell$ | another activation function for layer $\ell$. the paper says they use the identity function $g^\ell = \text{id}[J_\ell \times J_\ell]$ |
| $\bar{z}^\ell = g^{\ell}(\phi^{\ell + 1}(z^{\ell}) \cdot W^{\ell + 1})$ | top-down prediction of $z^\ell$ | 
| $e^\ell = \frac{1}{2 \beta_e} (\phi^\ell(z^\ell) - \bar{z}^\ell)$ | prediction error vector for layer $\ell$ |