## Introduction and Overview

This unit is primarily concerned with Bayesian computation, which is the crux of this course. We will first talk about the classical, numerical approach. Then, we will look at a more modern (easier) approach using Markov Chain Monte Carlo (MCMC).

It is often said that the most important mathematical tool in classical statistics is optimisation -- least square estimators, minmax rules, and so on. Bayesian statistics on the other hand utilises integration. 

Recall this dependence on integration. Bayes Theorem is given here:
$$
\pi(\theta|x) = \frac{f(x|\theta)\pi(\theta)}{m(x)}
$$
It is easy to observe that the posterior is proportional to the product of the likelihood and the prior, but **this is not a density function**. This product first needs to be normalised, but the normalising constant (the marginal) might not always be feasible.

For conjugate cases, we skipped finding the marginal (no integration required). Normalising $f(x|\theta)\pi(\theta)$ was easy. 

For non-conjugate cases, we typically numerically compute the posterior by integration (which has already been the subject of assignments thus far), or by random sampling. First, let us take a look at a motivating example.
### Motivation

Assume that $x|\theta \sim N(\theta, 1)$ and $\theta \sim Cau(0, 1)$. The likelihood is $f(x|\theta) \propto e^{-\frac{1}{2}(x-\theta)^2}$ and the prior is $\pi(\theta) \propto \frac{1}{1+\theta^2}$.

A Normal/Cauchy pair integral is **not solvable in terms of elementary functions**:
$$
\int_{-\infty}^{+\infty} e^{-\frac{1}{2}(x-\theta)^2}\cdot\frac{d\theta}{1+\theta^2}
$$

This is arguably worse if we want to find the Bayes estimator, which would look something like this: 
$$
\begin{aligned}
\delta_B(x) &= \frac{\int_{\Theta}\theta f(x|\theta)\pi(\theta)\, d\theta}{\int_{\Theta} f(x|\theta)\pi(\theta)\, d\theta} \\ \\
&= \frac{\int_{-\infty}^{+\infty} \frac{\theta}{1+\theta^2} e^{-\frac{1}{2}(x-\theta)^2} \, d\theta}{\int_{-\infty}^{+\infty} \frac{1}{1+\theta^2} e^{-\frac{1}{2}(x-\theta)^2}\, d\theta}
\end{aligned}
$$
What if we try sampling instead? We can either sample from the likelihood, or sample from the prior. First, let us rewrite the above, bearing in mind that $e^{-\frac{1}{2}(x-\theta)^2} \equiv e^{-\frac{1}{2}(\theta-x)^2}$
$$
\delta_B(x) = \frac{\int_{-\infty}^{+\infty} \frac{\theta}{1+\theta^2} e^{-\frac{1}{2}(\theta-x)^2} \, d\theta}{\int_{-\infty}^{+\infty} \frac{1}{1+\theta^2} e^{-\frac{1}{2}(\theta-x)^2}\, d\theta}
$$
Consider the case where we sample $\theta_1, \theta_2, \cdots, \theta_N$ from $N(x, 1)$. Skipping ahead, we would get that the Bayes estimator is equivalent to the following:
$$
\large
\delta_B(x) \approx \frac{\sum_{i=1}^N\frac{\theta_i}{1+\theta_i^2}}{\sum_{i=1}^N\frac{1}{1+\theta_i^2}}
$$
How does this work?
- Firstly, see that if we have $\int f(x)g(x) \, dx$ (where $f(x)$ is a density), the overall integral is essentially the **expectation of $g(x)$**, $\mathbb{E}(g(x))$.
- Therefore, if we take multiple samples from the density $f$, where $X \sim f$, The overall integral is approximately equal to $\frac{1}{n} \sum_{i=1}^n g(x_i)$. Notice that the integral is approximated as finite sums.
- Treating the normal distribution here as the density, we would arrive at the Bayes estimator as given above.

The other case where we sample instead from $Ca(0,1)$ would give us the following:

$$
\large
\delta_B(x) \approx \frac{\sum_{i=1}^N\theta_i e^{-\frac{1}{2}(\theta_i-x)^2}}{\sum_{i=1}^N e^{-\frac{1}{2}(\theta_i-x)^2}}
$$
In both cases, we are calculating the ratio of two integrals by approximation.
## Approximation

We have (across this course, thus far) motivated the need to circumvent numerical integration when dealing with Bayes theorem. We looked briefly at a way to approximate the Bayes estimator with sampling. 

Here we will look at more advanced methods for approximating integrals involved in Bayes computation in general.
### Laplace's Method

This method pulls from some time-honoured work by Laplace, and approximates the un-normalised posterior by a normal distribution. More concretely, we are trying to approximate $g(\theta) = f(x|\theta)\pi(\theta)$.

There are a few requirements that have to be satisfied.
- $g(\theta)$ should be unimodal and not too skewed (not too asymmetric)
- It should be easy to find the mode, $\hat{\theta}$ of $g(\theta)$. 
- $\theta$ is allowed to be multivariate

One might reasonably ask: since we are approximating using a normal distribution, why not perform moment matching instead (i.e., use mean and variance)? After all, a normal distribution is fully described by the mean and the variance. However, the mean and the variance depend on $m(x)$, which we do not have. 

The logarithm of $g(\theta)$ can be expanded around the mode:
$$
\begin{aligned}
\log g(\theta) \cong \log g(\hat{\theta})- \frac{1}{2}(\theta-\hat{\theta})'Q(\theta-\hat{\theta}) \\ \\
Q_{ij} = \left[-\frac{\partial^2}{\partial\theta_i\partial\theta_j}\log g(\theta)\right]_{\theta=\hat{\theta}}
\end{aligned}
$$
- The first term involves taking the logarithm at the mode, which is simply 0.
- The second term is a vector. $Q$ here is a matrix given by the definition on the second line. 

For a univariate parameter $\theta$ (which is the case for most of this course), matrix $Q$ becomes a scalar. 

In short, the approximation of the posterior is given below as
$$ \theta|x \sim N(\hat{\theta}, Q^{-1}) $$
We can also approximate the marginal as (note the proportional symbol):
$$
m(x) \propto \int_{\Theta} g(\theta)d\theta \cong \frac{g(\hat{\theta})}{\sqrt{\text{det}(Q/2\pi)}}
$$
The derivation for the posterior is shown here:

$$
\begin{aligned}
\int_{\Theta} g(\theta) d\theta &= g(\hat{\theta}) \int e^{-\frac{1}{2} (\theta - \hat{\theta})' Q (\theta - \hat{\theta})} d\theta \\
&= g(\hat{\theta}) \sqrt{2\pi \, \det Q^{-1}} \\&= \frac{g(\hat{\theta})}{\sqrt{\det(Q / 2\pi)}}
\end{aligned}
$$
#### Example

Let us now look at an example. 

Given $x|\theta \sim Ga(r, \theta)$ and $\theta \sim Ga(\alpha, \beta)$. Find Laplace's approximation to the posterior, and compare it with the exact posterior (the model is conjugate).

The exact posterior is proportional to 
$$
\begin{aligned}
f(x|\theta) \pi(\theta) &= \frac{x^{r-1} \theta^r}{\Gamma(r)} \times e^{-\theta x} \times \frac{\theta^{\alpha-1} \beta^{\alpha}}{\Gamma(\alpha)} \times e^{-\beta \theta} \\ \\
&\propto \theta^{r + \alpha - 1} e^{-(\beta + x)\theta}
\end{aligned}
$$
Notice here that the kernels are similar to that of the $Ga(\alpha + r, \beta + x)$ distribution. 

Let us proceed with Laplace approximation. We first take the log of the likelihood times prior, which gives us:
$$
\log g(\theta) = (r + \alpha - 1) \log(\theta) - (\beta+x)\theta
$$
The first derivative (w.r.t $\theta$) can be easily calculated as:
$$
\frac{\partial}{\partial \theta} \log g(\theta) = \frac{r + \alpha - 1}{\theta} - (\beta - x)
$$
If we set this expression to 0 and solve for $\theta$, we will get the mode, which can be represented as:
$$
\hat{\theta} = \frac{r + \alpha - 1}{\beta + x}
$$
We can check that this value is the mode by taking the second derivative and plugging in the value of $\hat{\theta}$. Quite simply, 
$$
\frac{\partial^2}{\partial \theta^2} \log g(\theta) = -\frac{r + \alpha - 1}{(\beta - x)^2} < 0
$$
- Therefore $\hat{\theta}$ is maximum.

! What the fuck !
Next, we find the second derivative of the logarithm of $g(\theta)$. The result is shown as following:
$$ Q = \left( -\frac{\partial^2}{\partial \theta^2} \log g(\theta) \right)_{\theta = \hat{\theta}} 
= \frac{r + \alpha - 1}{\hat{\theta}^2} = \frac{(\beta + x)^2}{r + \alpha - 1}
$$

As the variance of the approximated posterior is the reciprocal of $Q$, we may approximate the posterior as:
$$
Q^{-1} = \frac{r + \alpha - 1}{(\beta + x)^2} \Rightarrow \theta | x \stackrel{\text{approx}}\sim N \left( \frac{r + \alpha - 1}{\beta + x}, \frac{r + \alpha - 1}{(\beta + x)^2} \right)
$$

Also, we may approximate the marginal as:
$$
\begin{aligned}
\int_{\Theta} g(\theta) d\theta &= \frac{g(\hat{\theta})}{\sqrt{\text{det}(Q/2\pi)}} \\ \\
& \approx \sqrt{2\pi}\frac{(r+\alpha-1)^{r+\alpha-\frac{1}{2}}}{(\beta + x)^{r+\alpha}}e^{-(r+\alpha-1)}
\end{aligned}
$$
### Markov Chain Monte Carlo (MCMC)

We are not going into too much detail about MCMC, but note that this method is very important in Bayesian computation. In this section, we will cover the older Metropolis algorithm, followed by Gibbs sampling. The lecturer disclaims that this topic is actively being researched, but these newer algorithms are beyond the scope of this course.
#### Definition (Markov Chain)

Quite simply, we say that variables $X_0, X_1, X_2, \cdots, X_{n-1}, X_n, X_{n+1}$ form a Markov Chain if:
$$P(X_{n+1} \in A | X_0, X_1, \cdots, X_n) = P(X_{n+1} \in A | X_n)$$
In other words, given $X_n$, future events are independent of the past. Another popular way of phrasing this is: "the future depends only on the present, and not on the past".

![[Pasted image 20241003062441.png]]

One of the key points of Markov Chains would be the **transition kernel**:
$$ P(X_{n+1} \in A|X_n) = Q(A|X_n) $$
$$ Q(A|X_n = x) = \int_A q(x, y) dy = \int_A q(y|x) $$
- $Q$ here is simply a probability density. 

The notion of an **invariant distribution** is also key. $\Pi$ is an invariant distribution if:
$$ \Pi(A) = \int Q(A|x) \Pi(dx) $$
- In other words, if we take the transition kernel and integrate with respect to some distribution $\Pi$ -- we change nothing. 

Let $\pi$ represent the density for $\Pi$. It is **stationary** if:
$$ q(x|y)\pi(y) = q(y|x)\pi(x) $$
- This is called "detailed balance equation". $\pi$ is stationary if it satisfies this equation. 

In some Markov Chains (which we are not going to discuss), if:
$$ \lim_{n \rightarrow\infty} Q^n (A|x) = \Pi(A) $$
- Then $\Pi$ is an **equilibrium distribution**. 
- $Q^n (A|x) = P(X_n \in A | X_0 = x)$

Why is this all relevant? Basically, our goal here is to construct a Markov Chain such that the equilibrium distribution corresponds to the posterior. 

Given the definition of $Q^n$ above, it seems that the equilibrium distribution depends on where we start (i.e., where $X_0$ is). While this is true, we ignore the start for our purposes, as we assume that the initial condition is "forgotten" when $n$ is large. 
#### Definition (Monte Carlo)

The term "Monte Carlo" was coined by Nicole Metropolis for an approximation methodology based on sampling. 

We have actually encountered this before, when we were trying to approximate $\mu_\pi(g) = \int g(\theta)\pi(\theta)\,d\theta$. We attempted to take $n$ samples of $\theta$ (assuming they are $iid$ from $\pi$), then we got:
$$
\mu_{\pi}(g) \approx \frac{1}{n}\sum_{i=1}^n g(\theta_i)
$$
However, it is not always the case where we have $iid$. In that case, if we can fashion the Markovian dependence as described above, we still get the following based on **ergodic-type theorems**:
$$
\mu_{\pi}(g) \approx \frac{1}{n}\sum_{i=1}^n g(\theta_i)
$$
#### Summary

In sum, MCMC can be distilled into these few points:
- Form the Markov Chain such that the stationary distribution is the distribution of the posterior. 
- Start calculating from the posterior. Any function $g$ of the posterior (e.g., mean, variance) can be estimated using these ergodic-type theorems.
## Metropolis Algorithm

### Requirements

First, recall the detailed balance equation that is characteristic of a stationary distribution: $q(y|x)f(x) = q(x|y)f(y)$. Think of $q$ here as a transition kernel density in the Markov chain, and if $f$ satisfies detailed balance equation, it is a stationary distribution.

We assume that $\pi$ is our target distribution (posterior), and we are going to set the Markov chain to sample from the posterior.

Choosing the transition kernel density, $q$ is up to us. There are many possible choices, but bear in mind that $q$ must be **admissible**. More formally, $q$ is admissible if $\text{support} (\pi_x) \subset \bigcup_x \, \text{support} \, q(\cdot|x)$.
- Quoting the lecturer, "the support of this target density is a subset of the union with respect to all possible values of $x$, of the the support of the conditional density"
- To be honest I have no idea what this means but I am noting it down regardless.

In general, the detailed balance equation will not hold. That is, $q(y|x)\pi(x) \neq q(x|y)\pi(y)$.
- Say (WLOG), that the left hand side is larger than the right hand side.
- Then, we will need to multiply the left hand side by some quantity that is less than 1 in order to enforce equality. Let us denote this quantity $\rho(x, y)$. In other words, we have
$$
q(y|x)\rho(x, y)\pi(x) = q(x|y)\pi(y) \times 1
$$
- We can then express this balancing quantity as:
$$
\rho(x, y)= \frac{q(x|y)\pi(y)}{q(y|x)\pi(x)} \land1
$$

- This is nice, as $\rho$ depends on the ratio of the target distribution (recall that $\pi$ is the target). Therefore, if our target distribution is the posterior, the normalising marginals within that ratio **is cancelled**. 
- This is important, as the examples moving forward will only use the numerator of the Bayes theorem. If that happens, recall back here that the marginals have been cancelled out in this ratio.
### Algorithm

These four steps describe the Metropolis algorithm:
1. Start with arbitrary $x_0$ from the support of target $\pi$. 
2. At stage $n$, general proposal $y$ from $q(y|x_n)$.
3. Update the value of $x_{n+1}$ to $y$ with probability $\rho(x_n, y)$. Keep the value of $x_{n+1}$ as $x_n$ with probability $1-\rho(x_n, y)$. Programmatically, simply generate a uniformly random number between 0 and 1. Accept the proposal $y$ if this number if $\leq \rho(x_n, y)$.
4. Increase $n$ and go to step (2).

We mentioned above that the choice of $q$ is not really defined by any strict criteria. All that is required is for $q$ to be admissible. However, it is still possible to choose $q$ in certain ways.
- If $q(x|y) = q(y|x)$ (i.e., the kernel density is symmetric), then:
$$
\rho(x, y) = \frac{\pi(y)}{\pi(x)} \land 1
$$
- If $q(x|y) = q(y|x) = q(|x-y|)$, the algorithm is called the Metropolis random walk. This is the original proposal by Metropolis, and note that this condition is stricter than the one above.
- If $q(y|x) \equiv q(y)$, this algorithm is called Independence Metropolis. 
### Examples

#### Example 1

Let us take a look at how the Metropolis algorithm works for $X|\theta \sim N(\theta, 1)$ and $\theta \sim Ca(0, 1)$. This is an example encountered before (refer above).

Recall that our target is the posterior distribution. We have:
$$
\large
\pi(\theta|x) \propto \frac{e^{-\frac{(x-\theta)^2}{2}}}{1+\theta^2}
$$
Let $\theta$ be the current status, and $\theta'$ be the proposal.

Assume that we take our proposal from the density of $N(x, \tau^2)$. In other words,
$$
\large
q \propto e^{\frac{1}{2\tau^2}(\theta'-x)^2}
$$
Notice that this density does not depend on $\theta$. This is the Independence Metropolis algorithm. 

Take $\tau^2 = 1$. Now, we need to find the balancing term $\rho$. We can do this simply by taking the ratio of:
$$
\large
\frac{\pi(\theta') q(\theta | \theta')}{\pi(\theta) q(\theta' | \theta)} = \frac{\frac{e^{-\frac{(x - \theta')^2}{2}}}{1 + \theta'^2} e^{-\frac{(\theta - x)^2}{2}}}{\frac{e^{-\frac{(x - \theta)^2}{2}}}{1 + \theta^2} e^{-\frac{(\theta' - x)^2}{2}}} = \frac{1 + \theta^2}{1 + (\theta')^2}
$$
In other words, we have that:
$$
\rho = 1 \land \frac{1+\theta_n^2}{1+(\theta')^2}
$$
Therefore, we select $\theta_{n+1} = \theta'$ if $\rho$, and $\theta_{n+1} = \theta_n$ otherwise ($1-\rho$).

#### Example 2

Let us now look at the Weibull distribution. Suppose that we have $T_1, \cdots T_n$ that we believe follow a Weibull distribution with parameters $\alpha$ and $\eta$. More concretely,
$$
f(t|\alpha, \eta) = \alpha\eta t^{\alpha-1}e^{-\eta t^{\alpha}}
$$
Note that when $\alpha=1$, the Weibull distribution _is_ the exponential distribution. Perhaps more correctly, the Weibull distribution is a generalisation of the exponential distribution. 

Assume that we put priors on both $\alpha$ and $\eta$. We say that the prior on $\alpha$ is exponential with rate=1, and the prior on $\eta$ is gamma with parameters $\beta$ and $\xi$. The combined prior on both would therefore be:
$$
\pi(\alpha, \eta) \propto e^{-\alpha}\cdot\eta^{\beta-1}e^{-\xi\eta}
$$
The proposal would be the product of two exponentials:
$$
q(\alpha',\eta'|\alpha, \eta) = \frac{1}{\alpha\eta}\text{exp}\{-\frac{\alpha'}{\alpha} - \frac{\eta'}{\eta}\}
$$
As usual, we will accept the proposal with the probability of some value $\rho$, or keep the old value with probability $1-\rho$. This is what $\rho$ looks like:

$$
\rho = 1 \land \frac{ \left[ \prod_{i=1}^{n} \alpha' \eta' t_i^{\alpha'-1} e^{-\eta' t_i^{\alpha'}} \right] e^{-\alpha'} (\eta')^{\beta-1} e^{-\xi \eta'} \cdot\frac{1}{\alpha' \eta'} e^{-\frac{\alpha}{\alpha'} - \frac{\eta}{\eta'}} } { \left[ \prod_{i=1}^{n} \alpha \eta t_i^{\alpha-1} e^{-\eta t_i^{\alpha}} \right] e^{-\alpha} \eta^{\beta-1} e^{-\xi \eta} \cdot\frac{1}{\alpha \eta} e^{-\frac{\alpha'}{\alpha} - \frac{\eta'}{\eta}} }
$$
Let's break it down.
- The left most term is simply the kernel density, $q$. The numerator reflects $q$ of "current" given "proposed" (i.e.,  $\alpha, \eta | \alpha', \eta'$). The denominator reflects $q$ of "proposed" given "current" (flip the symbols around).
- The right term is nothing more than the target. In our case, this is the posterior, which can be evaluated as the product of the likelihood and the prior (left: likelihood, right: prior). Recall that the marginals are nowhere to be found as they will cancel out -- we can safely ignore them. 

We first set $\beta=2$ and $\xi=2$ as hyperparameters. We take note of our observations for $T$: $0.2, 0.1, 0.25$. We also set our initial values of $\alpha=\eta=2$.

After running the Metropolis Algorithm, we get:
- $\hat{\alpha} \cong 0.9$
- $\hat{\eta} \cong 1.85$
## Gibbs Sampling

### Requirements

Gibbs Sampling, at its core, is a **special case** of the Metropolis algorithm. It involves component-wise updates with the "proposals" being **full conditional distributions** of components. 

Simply put, if we have multiple parameters, each parameter will have its own proposal. These proposals are in turn full conditional distributions of that particular component, given everything else. 

Formally, let $f(\tilde{X}|\tilde{\theta}) \pi(\tilde{\theta})$ be the numerator of the posterior. Ideally, we would need to find full conditionals for all components of $\tilde{\theta} = \theta_1, \cdots, \theta_p$:
$$
\begin{aligned}
\pi(\theta_1|\theta_2, &\cdots, \theta_n, \tilde{X}) \\
\pi(\theta_2|\theta_1, &\cdots, \theta_n, \tilde{X}) \\
&\vdots \\
\pi(\theta_n|\theta_1, &\cdots, \theta_{n-1}, \tilde{X})
\end{aligned}
$$
Notice that the conditional also factors in the data, $\tilde{X}$. 

It can be shown that $\rho=1$. In other words, the Gibbs proposal is accepted at every step.
### Algorithm

Assuming that the full conditionals have been found, the Gibbs sampler itself is relatively simple.
1. Start with some initial value $\theta^0 = (\theta_1^0, \theta_2^0, \cdots, \theta_p^0)$
2. Sample $\theta_1^{n+1}$ from $\pi(\theta_1|\theta_2^n, \cdots, \theta_p^n, \tilde{X})$
	- Repeat for all $p$, but use the accepted values at the $n+1$ stage if already encountered. In other words,
	- Sample $\theta_2^{n+1}$ from $\pi ( \theta_2 | \theta_1^{n+1}, \theta_3^n, \dots, \theta_p^n, \tilde{X})$,
	- Sample $\theta_3^{n+1}$ from $\pi(\theta_3 | \theta_1^{n+1}, \theta_2^{n+1}, \theta_4^n, \dots, \theta_p^n, \tilde{X})$, and so on until
	- Sample $\theta_p^{n+1}$ from $\pi(\theta_p|\theta_1^{n+1}, \cdots, \theta_{p-1}^{n+1}, \tilde{X})$
3. Increment $n$ and repeat step (2)
### Finding Full Conditionals

So how would we go about finding the full conditionals? 

First, simply form a kernel of joint distribution -- this involves all parameters and the observed data. In other words, we take the likelihood (observed data) and multiply by all priors (all parameters). We may ignore constants in this. 

To find the full conditional for a component $\theta_i$, select only the parts of the kernel that contain $\theta_i$. All other $\theta_j$, where $i \neq j$, and data can be considered constant.

Next, normalise the selected part as a distribution. Often, it is possible to recognise the distribution from the form of the kernel. Importantly, note that we should be able to sample from all conditionals (as that is the point of the algorithm).

### Examples

#### Example 1

Suppose we have $n$ observations ($X_1, \cdots, X_n$) coming from a normal distribution $N(\mu, \frac{1}{\tau})$. The parameter $\tau$ is of interest -- this is simply the precision and is defined as $\frac{1}{\sigma^2}$.

We have two priors:
- $\mu \sim N(0, 1)$, and
- $\tau \sim Ga(2, 1)$

Recall from above that we first need to find a kernel of joint distribution. We do this by multiplying the likelihood and all priors. In other words, we would get:
$$
\large
\text{joint} \propto (2\pi)^{-\frac{n+1}{2}} \tau^{\frac{n}{2}} e^{-\frac{\tau}{2} \sum_{i=1}^{n}(x_i-\mu)^2} \cdot e^{-\frac{1}{2} \mu^2}\cdot \tau e^{-\tau}
$$
- The first half ('$\cdot$' separates the components) represents the likelihood.
- The second half represents the product of both priors.
- Notice that all constants have been removed.

Now, we select (for each full conditional distribution) only the parts that contain the parameter of interest. Let's take a look at the following:

$\mu$ here is our parameter of interest.
$$
\large
\begin{aligned}
\pi (\mu| \tau, \tilde{X}) &\propto e^{-\frac{\tau}{2} \sum_{i=1}^{n} (x_i-\mu)^2} e^{-\frac{1}{2} \mu^2}  \\ &\propto e^{-\frac{1}{2} (1+n\tau)\left(\mu-\frac{\tau \sum_{i}x_i}{1+n\tau}\right)^2}
\end{aligned}
$$
Notice that this resembles a normal distribution! In other words:
$$
\mu | \tau, \tilde{X} \sim N\left( \frac{\tau\sum_{i} x_i}{1+n\tau}, \frac{1}{1+n\tau} \right)
$$

Shifting our focus to the full conditional of $\tau$.
$$
\large
\begin{aligned}
\pi (\tau | \mu, \tilde{X}) &\propto \tau^{\frac{n}{2}} e^{-\frac{\tau}{2} \sum_{i=1}^{n}(x_i-\mu)^2} \tau e^{-\tau} \\ &\propto \tau^{\frac{n}{2} + 1} e^{-\tau \left[1 + \frac{1}{2} \sum_{i=1}^{n}(x_i-\mu)^2 \right]}
\end{aligned}
$$
This resembles a gamma distribution. More completely:
$$
\tau | \mu,\tilde{X} \sim Ga\left( \frac{n}{2} + 2, 1 + \frac{1}{2} \sum_{i=1}^{n}(x_i-\mu)^2 \right)
$$

Finally, using these full conditionals, we would simply run the Gibbs Sampling algorithm.
#### Example 2

Here, we will revisit an example we are already familiar with:
- $X|\theta \sim N(\theta, 1)$, and
- $\theta \sim Ca(0, 1)$

Again, we are interested in finding $\delta(2)$ by Gibbs sampling. 

Notice firstly that there is only one parameter of interest, $\theta$. However, there is a way to tease out a second parameter. The fact that the prior follows a Cauchy distribution with some parameters $\mu$ and $\tau$ is equivalent to the following hierarchical structure:

We may condition $\theta$ on some $\lambda$ and have $\theta |\lambda \sim N(\mu, \frac{\tau^2}{\lambda})$. Then, we take $\lambda$ to be $\sim Ga(\frac{1}{2}, \frac{1}{2})$. We are using this probabilistic identity to represent Cauchy here as the integral shown below:

$$
\begin{aligned}
\pi(\theta) &\propto \frac{1}{\tau^2+(\theta-\mu)^2} \\ \\
&\propto \int_0^{+\infty} \sqrt{\frac{\lambda}{2\pi\tau^2}}\cdot\exp\{-\frac{\lambda}{2\tau^2}(\theta-\mu)^2\} \cdot\lambda^{\frac{1}{2}-1}e^{-\frac{\lambda}{2}}\,d\lambda
\end{aligned}
$$

If we take the product of the likelihood and the priors (normal for $\theta|\lambda$ and gamma for $\lambda$), we will end up (after much simplification) with the full conditionals for:
$$
\begin{aligned}
\theta|\lambda,x &\sim N\left(\frac{\tau^2}{\tau^2+\lambda\sigma^2}x + \frac{\lambda\sigma^2}{\tau^2+\lambda\sigma^2}\mu, \,\frac{\tau^2\sigma^2}{\tau^2+\lambda\sigma^2}\right) \\ \\
\lambda|\theta, x &\sim Exp\left(\frac{\tau^2+(\theta-\mu)^2}{2\tau^2}\right)
\end{aligned}
$$
As an exercise, derive the conditionals given above.

Finally, run the Gibbs Sampling algorithm and estimate the parameter $\theta$.

#### Example 3

Let us take a look now at a particularly interesting example. Firstly, let us define "coal mining disasters" as an accident in a coal mining environment in which 10 or more people died. Let us examine coal mining disasters from 1851 to 1962.

It was noted that in some earlier portion of the years (1851 to 1851 + $m$), the number of accidents was much higher. Somewhere around the 1900s, things improved and the number of accidents decreased. We could therefore ask if there is any **change point** that we can infer about when the rate of disasters went down?

Let us assume that the worse portion is modelled by a Poisson distribution with rate $\lambda$. In a similar vein, let us assume that the better portion is modelled by Poisson with rate $\mu$. More concretely, we have the following:
$$
\begin{aligned}
&x_i|\lambda \sim \text{Poi}(\lambda) \quad i = 1, 2, \cdots, m \\
&x_i|\mu \sim \text{Poi}(\mu) \quad i = m+1, \cdots, n \\ \\
&m \sim \text{DU}(n): \, P(m=k) = \frac{1}{n}, \, k=1, \cdots,n \,(n=112) \\ \\
&\lambda\sim\text{Ga}(\alpha, \beta) \\
&\mu\sim\text{Ga}(\gamma,\delta)
\end{aligned}
$$
- For the first $m$, we put Poisson with parameter $\lambda$. For the remaining, we put Poisson with parameter $\mu$. 
- $m$ represents the change point, which we shall model with a discrete uniform (DU) prior.
- To complete the model, we put Gamma priors on $\lambda$ and $\mu$, both with their hyperparameters as shown above.

With Gibbs sampling, the first step is always to find the product of likelihood and all priors. We get:
$$
\begin{aligned}
&L\left(\lambda, \mu, m \mid X \right) \pi(\lambda) \pi(\mu) \pi(m) \\\\
&= \prod_{i=1}^{m} \frac{\lambda^{x_i}}{x_i!} e^{-\lambda} \prod_{i=m+1}^{n} \frac{\mu^{x_i}}{x_i!} e^{-\mu} \frac{\beta^\alpha \lambda^{\alpha-1} }{\Gamma(\alpha)}e^{-\beta\lambda} \frac{\delta^\gamma \mu^{\gamma-1}}{\Gamma(\gamma)} e^{-\delta\mu}\cdot \frac{1}{n} \\ \\
&\propto e^{-m\lambda} \lambda^{\sum_{i=1}^{m} x_i} e^{-(n-m)\mu} \mu^{\sum_{i=m+1}^{n} x_i} \lambda^{\alpha-1} e^{-\beta \lambda} \mu^{\gamma-1} e^{-\delta \mu} \\ \\
&= \lambda^{\alpha + \sum_{i=1}^{m} x_i - 1} e^{-(m + \beta)\lambda} \mu^{\gamma + \sum_{i=m+1}^{n} x_i - 1} e^{-(\delta + n - m)\mu}
\end{aligned}
$$
Then, we will arrive at the following full conditionals:
$$
\lambda \mid \mu, m, X \sim \text{Ga} \left( \alpha + \sum_{i=1}^{m} x_i, \beta + m \right)
$$
$$
\mu \mid \lambda, m, X \sim \text{Ga} \left( \gamma + \sum_{i=m+1}^{n} x_i, \delta + (n - m) \right)
$$
What about the full conditional for $m$? Note that $m$ appears in all terms! It doesn't seem as easy to simplify as we would for the above two full conditionals.

First, let us take a look at the products (i.e., before expansion and simplification). We find that $m$ only appears in two terms there:
$$
\pi(m \mid \lambda, \mu, X) \propto \prod_{i=1}^{m} \frac{\lambda^{x_i}}{x_i!} e^{-\lambda} \prod_{i=m+1}^{n} \frac{\mu^{x_i}}{x_i!} e^{-\mu}
$$
This can be simplified into:
$$
\left[ \prod_{i=1}^{n} \frac{\mu^{x_i}}{x_i!} e^{-\mu} \right] e^{m(\mu - \lambda)} \left( \frac{\lambda}{\mu} \right)^{\sum_{i=1}^{m} x_i} = f(x \mid \mu) g(x \mid m)
$$
In other words the left-hand term $f(x|\mu)$ is constant with respect to $m$ and the right-hand term $g(x|m)$ (i.e., everything after the right closing square bracket) depends on $m$. We have:
$$
\pi(m) \propto e^{m(\mu - \lambda)} \left( \frac{\lambda}{\mu} \right)^{\sum_{i=1}^{m} x_i}
$$
Recall that $m$ is discrete, and the range is from 1 to $n$. The probabilities for $m$ are proportional to the quantity given above. To normalise the probabilities, we simply take:
$$
P(m = k) = \frac{\pi(k)}{\sum_{i=1}^{n} \pi(i)}
$$
- In other words, we normalise the full conditional where $m=k$ (for each $k$), then divide by the sum of all full conditionals from $1$ to $n$.

Finally, we run Gibbs sampling. The only special case here involves the sampling for $m$, since we have to build our custom distribution as shown above. The rest can simply be sampled by invoking the appropriate distributions. 