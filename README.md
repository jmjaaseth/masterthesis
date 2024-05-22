# About the project

This project...

## Framework for quantum machine learning

## Descriptors

### Deviation from Haar distribution

$$
A^{(t)}(C)= \left \lVert \int_{Haar}^{} (\ket{\psi}\bra{\psi})^{\otimes t}d\psi - \int_{\Theta}^{} (\ket{\phi}\bra{\phi})^{\otimes t}d\theta \right \rVert_{HS}
$$

### Kullback–Leibler divergence

$$
\mathcal{E} (C) = D_{KL} (P(C, F) \parallel P_{Haar}(F)) = \int_{0}^{1} P(C, F) \, log \frac{P(C, F)}{P_{Haar}(F)} \,dF
$$

### The Meyer-Wallach entanglement measure

$$
Ent(C) = \frac{2}{|S|} \sum_{\ket{s} \in S} \left(1 - \frac{1}{n} \sum_{j = 1}^{n} Tr \left [ \rho \left (\ket{s})_j^2 \right ] \right ) \right )
$$