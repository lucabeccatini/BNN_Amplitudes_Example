# BNN_Amplitudes_Example
Example of a Bayesian Neural Network for amplitudes regression

## Table of contents
- [About the project](#about-the-project)
    - [Bayesian NN]
    - [Amplitude regression]
- [Main implementation]


## About the project
This project is an example of a Bayesian Neural Network (BNN) applied to a regression problem in High Energy Physics, in particular to predict the amplitudes of scateering events. Unlike traditional NNs that provide point estimates, BNNs offer a probabilistic approach by estimating a distribution over the network's weights, allowing for more robust predictions and uncertainty quantification. This makes BNNs ideal for tasks like amplitude regression, where precise modeling of particle interactions and scattering amplitudes is essential for accurate physical predictions and analysis.


### Bayesian NN
BNNs extend traditional NNs by modeling network parameters as probability distributions rather than fixed values. This approach allows the network to produce not just a single output, but a distribution of possible outputs, providing both a mean prediction and an associated uncertainty. 
Instead of learning the optimal weights $w$ to approximate the training dataset $D$, as a traditional NNs, a BNN learns the posterior weights distribution $p(w|D)$, which is the probability to a have a set of weight $w$ given the dataset $D$.

Since the posterior cannot usually be evaluated analytically, $p(w|D)$ is approximated by a variational distribution $q_{\theta}(w)$, depending on the network parameters $\theta$. This method is called variational inference. 
In order to obtain the best parameters $\theta$ to approximate the posterior, we can minimize the Kullback-Leibler-divergence (KL) between the two distributions. The KL evaluates the difference between two probability distributions and it is null for identical distributions. 
The KL between $q_{\theta}(w)$ and $p(w|D)$ is:
```math
KL \left(q_{\theta}(w), p(w|D) \right) = \int \, dw \, q_{\theta}(w) \, log \left(\frac{q_{\theta}(w)}{p(w|D)} \right) \quad .
```

Using the Bayes theorem, we can rewrite the posterior as:
```math
p(w|D) = \frac{p(D|w) \cdot p(w)}{p(D)} 
```
where $p(D|w)$ is likelihood distribution of observing the dataset $D=(X, Y)$ for the given network with weights $w$, $p(w)$ is the prior probability and $p(D)$ is the model evidence. 
The prior can be chosen freely and corresponds to our knowledge of the weights distribution before seeing the data. 
The model evidence represents the probability of having the dataset $D=(X, Y)$ with input $X$ and result $Y$. 
Substituting this last formula in the KL formula, we obtain the loss that we want to minimize:
```math
KL \left(q_{\theta}(w), p(w|D) \right) = \int \, dw \, q_{\theta}(w) \, \left[ log \left(\frac{q_{\theta}(w)}{p(w)} \right) - log \left(p(D|w) \right) + log \left(p(w) \right) \right]
```
The first term is a KL term between the variational distribution $q_{\theta}(w)$ and the prior distribution $p(w)$. The second term is called is called negative log-likelihood (NLL) and it depends on the posterior probability. The third term depends only on the the model evidence. It can be omitted because it is independent of $\theta$, so it is not relevant for the minimization of KL respect to $\theta$. Using a MC estimator for the NLL, we can simplified the total KL loss as:
```math
Loss(\theta) = KL \left(q_{\theta}(w), p(w) \right) - \mathcal{E}_{q_{\theta}(w)} \left[log \left(p(D|w) \right) \right]
```
The $KL \left(q_{\theta}(w), p(w) \right)$ depends only on the network parameters and not on the training data. Since the network parameters are compared to the prior, which has no data information and it has a simple distribution, this loss term acts as a regularization term, preventing the network parameters to become too large or complex. 
The NLL depends on the error predictions and their estimated uncertainty. 

As we said, the choice of the prior distribution is free, but a general choice for its simplicity is a gaussian approximation, with null mean and standard deviation equal to 1. Using this approximation, the KL loss term can be simplified as: 
```math
KL \left(q_{\theta}(w), p(w) \right) \approx \sum_i \frac{1}{2} \left( \mu_i^2 + \sigma_i^2 - log(\sigma_i^2) - 1 \right)
```
where $\mu_i, \sigma_i$ are respectively the mean and standard deviation of the weight distribution $i$.
Similarly, using a gaussian approximation for the variational distribution, we can approximate the NLL as:
```math
NLL(y_i, \mu_i, \sigma_i) \approx \frac{(\mu_i - y_i)^2}{2\sigma_i^2} + log(\sigma_i) \quad .
```

