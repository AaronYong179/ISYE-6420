## Bayesian Inference in Conjugate Cases
### Estimation

Recall that for a Bayesian, the posterior is an **ultimate summary** of the experiment, prior information, and observed data.

![[Pasted image 20240907153457.png]]

The only coherent approach to incorporate prior information into the **inference** is via Bayes Theorem. The inference, then, can be divided into two main parts:
- Estimation (which we can perform with point or interval estimators), and  
- Testing
#### Estimators

Characteristics of the posterior provide us with useful estimators. For example, the mean, median, and mode of the posterior are all examples of a **point estimation of the parameter**. 

Let us now see how these are related to Bayesian decision theory -- the part of statistics that is essentially concerned with minimising losses with some optimal action(s).

The Bayes estimator is a function over $X_1, \cdots, X_n$. When $X_1,\cdots,X_n$ are observed, the Bayes estimator becomes a constant. This estimator is now called an action, $a$.
##### Posterior Mean

The **posterior mean** is therefore an action that minimises 
$$\mathbb{E}^{\theta|X}(\theta-a)^2$$with respect to $a$. 

This is essentially trying to minimise the **square error loss**.  $\theta$ is the true parameter, and the **action** that minimises this loss is the posterior mean. The posterior mean is the most common Bayes estimator of the parameter. The posterior mean is also connected to **Bayes risk**, although we will return to the posterior mean as the Bayes estimator in a later subsection.

##### Posterior Median

The **posterior median** is an action that minimises $$\mathbb{E}^{\theta|X}|\theta-a|$$ with respect to $a$.

##### Posterior Mode

The **posterior mode** is an action that minimises
$$
\lim_{c\rightarrow0}\mathbb{E}^{\theta|X}L_c(\theta, a)
$$
with respect to $a$.

Where: 
$$
L_c(\theta) =
\begin{cases}
0, \quad |\theta-a| \leq c\\
1, \quad \text{else}
\end{cases}
$$
#### The Bayes Estimator

The posterior mean is also connected with **Bayes risk** (though note that Bayes risk is not _really_ Bayesian). Bayes risk is given as:
$$
r(\theta, \delta) = \mathbb{E}^{\theta}\mathbb{E}^{X|\theta}(\theta - \delta(X)^2) \equiv \mathbb{E}^{X}\mathbb{E}^{\theta|X}(\theta - \delta(X)^2) 
$$
$\delta(X)$ here denotes the rule (i.e., the action) that minimises the overall risk. 
- We first take the expectation with respect to the model, $\mathbb{E}^{X|\theta}$. 
- Then, we take the expectation with respect to the parameter, $\theta$. 

If $X$ is observed, that is, if $\delta_B$ (posterior mean) is conditioned on $X$, the result is Bayes action. 

The Bayes estimator, $\delta_B(x)$ (mean) is the expectation of $\theta$ with respect to the posterior distribution. In other words:
$$\delta_B(x) = \int_{\Theta} \theta \cdot \pi(\theta|x) d\theta $$
> Take a moment to think about why this makes sense. Recall that for discrete distributions, the expectation $\mathbb{E}$ is defined as the sum of all values of $x$ multiplied by the probability of $x$ occurring. The same thing is happening here.
>
> For all values of $\theta$, we compute the product of $\theta$ and $\pi(\theta|x)$. We take the sum across all values, which for a continuous distribution is simply the integral.

Now recall that the posterior can be defined as:
$$
\pi(\theta|x) = \frac{f(x|\theta)\pi(\theta)}{m(x)} = \frac{f(x|\theta)\pi(\theta)}{\int_{\Theta}f(x|\theta)\pi(\theta)d\theta} 
$$

Plugging in the above into our original definition of the Bayes estimator, we get:
$$
\delta_B(x) = \frac{\int_{\Theta}\theta\cdot(x|\theta)\pi(\theta)d\theta}{\int_{\Theta}f(x|\theta)\pi(\theta)d\theta}
$$
In summary, 
- Bayes rule, $\delta_B(X)$, is the rule in which $X$ is not specified and it minimises Bayes risk.
- Bayes action, $a*$, is the Bayes rule for observed data and minimised posterior expected square error loss.  

### Credible Sets

#### Definition

Recall that given a continuous distribution, the probability of any particular point is zero. In other words, there is no way to be correct with an exact, point estimator. To alleviate this, we can find the estimator in the form of an interval -- this interval in turn contains a known parameter with pre-assigned high probability. 

Here is where the Classical and Bayesian statisticians differ. 
- In Classical statistics, we have some unknown, constant parameter. By conducting experiments, we find the confidence interval. The confidence interval is interpreted as follows: in the long run, the proportion of times that this random interval will cover the unknown parameter is $1-\alpha$ (e.g., 95% confidence interval).
- The interpretation of a confidence in terms of a credible set is much more natural. Simply put, it is the probability that the parameter belongs to a credible set (e.g., 95% confidence).

When finding credible sets, suppose that the posterior $\pi(\theta|X)$ is already found. Assume that we have some subset $C$ of the parameter space $\Theta$. In other words, $C \subset \Theta \equiv$ parameter space.

Then $C$ is a credible set with credibility $1-\alpha$ if:
$$
\int_C \pi(\theta|x)d\theta \geq 1-\alpha
$$

#### Credible Set Types

We distinguish between two types of credible sets (or, intervals):
- The HPD (highest posterior density) credible set, and
- The equitailed credible set.

##### HPD Credible Set

The HPD credible set is defined as:
$$C = \{\theta \in \Theta | \pi(\theta|x) \geq k(\alpha)\}$$

Let's break the above down step by step.
- Recall that $\alpha$ is involved in the "credibility" or "confidence" such that our confidence that $\theta$ falls in $C$ is $1-\alpha$.
- $k(\alpha)$ here refers to some set level involving $\alpha$. Think of this as a cutoff with some y-value.
- Then, anything in the posterior distribution that **exceeds** the level $k(\alpha)$ would form an interval. If the distribution is not unimodal, then it would form a set of intervals.
- A HPD credible set is illustrated below:

![[467d58d841a4d679f95d18b0111801520a392232fab7141cace9d0e27a06d298.png]]

- $k(\alpha)$ is set in such a way that the area, $P^{\theta|X}(\theta \in C) \geq 1-\alpha$.

##### Equitailed Credible Set

In brief, we take the posterior distribution $\pi(\theta|x)$ and form the interval $[L, U]$, where $L$ and $U$ denotes "lower" and "upper" respectively.

If we want to have our confidence that $\theta$ is contained within $[L, U]$ to be greater than $1-\alpha$; i.e. $P^{\theta|X}(\theta \in [L,U]) \geq 1-\alpha$, we:
- ensure that the area of the region to the **left of** $L$ is less than $\frac{\alpha}{2}$, and
- ensure that the area of the region to the **right of** $U$ is less than $\frac{\alpha}{2}$.

More formally,
$$
\int_{-\infty}^L\pi(\theta|x)d\theta \leq \frac{\alpha}{2}; \quad \int_{U}^{+\infty}\pi(\theta|x)d\theta \leq \frac{\alpha}{2};
$$

![[a301739aaa54d292e4e16fa05a8994df681eb07addbd4cadb9561bc450a22b63.png]]

In practive, the equitailed credible set is much easier to calculate (all we need are quantiles of the posterior). Finding the HPD credible set requires finding the $k(\alpha)$ such that the area is $\geq 1-\alpha$.

##### Example 1

We will revisit Jeremy's IQ. Recall the following:
- $X|\theta \sim N(\theta, \sigma^2)$
- $\theta \sim N(\mu, \tau^2)$
- $X = 98$, $\sigma^2 = 80$, $\tau^2 = 120$
- We found that the posterior is modelled as $\theta|X \sim N(102.8, 48)$

For credible sets, all we really need is the posterior. In this case, both the HPD and equitailed credible set coincide, as a normal distribution is symmetric in nature.

The 95% credible set (CS) will be $102.8 \pm z_{1-\frac{\alpha}{2}} \cdot \sqrt{48}$
- We have to divide $\alpha$ by two here since we are interested in an equitailed credible set. It is easy to find that $z_{0.975} = 1.96$. 
- Finally, we get $\theta \in [89.2207, 116.3793]$. The length, $L$, of this credible set is about $27.1586$

Now let us see how a frequentist would approach this, by calculating the 95% confidence interval (CI). The frequentist would make no use of priors, nor will they have a posterior -- they simply have the data as given, which would be values of $X$ and $\sigma^2$. 
- A 95% CI can then be calculated as $98 \pm z_{1-\frac{\alpha}{2}}\cdot\sqrt{80}$. 
- They get $\theta \in [80.4692, 115.5308]$. The length, $L$, of this confidence interval is about $35.0615$.

Notice here that the length of the CS is smaller than the CI. This should make sense intuitively, since we are building the CS based on some prior knowledge (which should make us more certain about the parameter, since we have more information).

Of course, note that the interpretations for the CS and the CI are **different**. 
- For the CS, we say that there is a probability of 0.95 that $\theta$ belongs to this credible set.
- For the CI, we say that if multiple samples are taken, the confidence interval will cover $\theta$ 95% of the time.
##### Example 2

Let us now consider another example. Assume that we have a sample size $n$, and the samples are distributed exponentially with the rate parameter $\lambda$. 
$$ X_1, X_2, \cdots, X_n \sim \text{Exp}(\lambda)$$
Note that when we sum up exponentials, we arrive at a gamma distribution. In other words:
$$ \sum_{i=1}^n X_i \sim Ga(n, \lambda)$$
- In fact, the exponential distribution is simply a special case of the gamma distribution, where the shape parameter, $n$, is equal to 1. 

Suppose also that we have the $\lambda$ parameter following a gamma distribution with hyperparameters $\alpha$ and $\beta$. In other words:
$$ \lambda \sim Ga(\alpha, \beta) $$
It is given that:
$$
\begin{aligned}
f(x|\lambda) &\propto \lambda^n \cdot e^{-\lambda\sum x_i} \\ \\
\pi(\lambda) &\propto \lambda^{\alpha-1} e^{-\beta\cdot\lambda}
\end{aligned}
$$
The likelihood and prior form conjugate pairs, and the posterior is also given as:
$$
\pi(\lambda|\sum X_i) \sim Ga(n + \alpha, \sum x_i + \beta)
$$

Now, let $X_1 = 2$, $X_2 = 7$, and $X_3 = 12$ be the lifetimes of a particular device. Assume that $X_i$'s are exponential $\text{Exp}(\lambda)$ with unknown parameter $\lambda$. Let the prior on $\lambda$ be $Ga(1, 8)$.

Find Bayes estimators for $\lambda$ and credible sets (HPD and Equitailed).

Substituting the given values into the posterior, we get that:
$$ \lambda | \sum X_i \sim Ga(3+1, 21+8)  \equiv Ga(4, 29)$$
Calculating Bayes estimators:
- The posterior mean is $\frac{4}{29} = 0.1379$
- The posterior mode is $\frac{4-1}{29} = 0.1034$
- The posterior median must be determined numerically. We can use the 50th percentile (0.5 quantile) of the gamma inverse (`gaminv` in Matlab). This gives us $0.1266$

Recall that $\mathbb{E}(X_i|\lambda) = \frac{1}{\lambda}$.

Therefore, if we are interested in estimating the lifetimes using our calculated Bayes estimators, we get ${7.2516, 9.6712, 7.8975}$ if we use the posterior mean, mode, or median respectively.

### Bayesian Testing

#### Introduction

Assume that $\Theta_0$ and $\Theta_1$ are two non-overlapping sets of parameter $\theta$. Note that $\Theta_0$ and $\Theta_1$ do not necessarily split the parameter space. We want to test
$$ H_0: \theta \in \Theta_0 \quad \text{v.s.} \quad H_1: \theta \in \Theta_1 $$
Note that in Bayesian statistics, we are not concerned with the designation of the null and alternative hypothesis.
> This is not true in classical statistics, where the power of analysis critically depends on what is assigned as the null or alternative.

Here, we simply find the posterior probability of $\Theta_0$ as well as $\Theta_1$. Then, pick the hypothesis with the larger posterior probability. 

First we calculate the following. Take note here that the **posterior probability** is denoted as $p_i$.
$$
\begin{aligned}
p_0 = \int_{\Theta_0} \pi(\theta|x)d\theta = \mathbb{P}^{\theta|X}(H_0) \\
p_1 = \int_{\Theta_1} \pi(\theta|x)d\theta = \mathbb{P}^{\theta|X}(H_1)
\end{aligned}
$$
We might have different **prior probabilities** of hypotheses, which we will denoted with $\pi_i$. 
$$
\begin{aligned}
\pi_0 = \int_{\Theta_0} \pi(\theta)d\theta \\
\pi_1 = \int_{\Theta_1} \pi(\theta)d\theta 
\end{aligned}
$$
We can then determine $B_{01}$, the Bayes factor in favour of $H_0$. Note that $B_{10}$ is in favour of $H_1$.
$$
\begin{aligned}
B_{01} &= \frac{p_0/p_1}{\pi_0/\pi_1} \\ \\
B_{10} &= \frac{1}{B_{01}}
\end{aligned}
$$
- Notice that $B_{01}$ is simply the posterior odds over the prior odds.
- Another way to think about this is to consider that $\frac{p_1}{p_0} = B_{10} \times \frac{\pi_1}{\pi_0}$.

The above has been concerned with asking if $\theta$ belonged to some set, $\Theta$. Testing a **precise null**, where $H_0: \theta = \theta_0$ is more involved, as it requires a prior with point mass at $\theta_0$. Of course, it is very unlikely to have point mass specifically at $\theta_0$, hence the posterior probability of such precise hypotheses will always be zero.

#### Precise Hypothesis Testing

Let us now consider how to deal with precise hypotheses. Suppose we have:
$$ H_0: \theta = \theta_0 \quad \text{v.s.} \quad H_1: \theta \neq \theta_0 $$
We have said above that we need a prior with point mass at $\theta_0$ (otherwise the posterior probability of $H_0$ is simply zero). We can therefore think of the prior as "split" between two components:
$$ \pi(\theta) = \pi_0\cdot\delta_{\theta_0} + (1-\pi_0)\cdot\xi(\theta)$$
$\xi$ is some spread distribution of $\theta$. Let's take a step back:
- The prior now has two components (a mixture prior). A point mass at $\theta_0$, and a spread distribution outside of $\theta_0$.\

Let $\pi_1$ denote $1-\pi_0$. 

If we take the marginal, we get:
$$
m(x) = \pi_0\cdot f(x|\theta_0) + \pi_1\cdot \int_{\{\theta\neq\theta_0\}} f(x|\theta)\xi(\theta)d\theta
$$
Again, let's take a step back.
- Recall that the marginal is calculated by integrating the product of the likelihood and the prior.
- At the point $\theta_0$, this is simply $f(x|\theta_0)$. 
- Outside of $\theta_0$ (i.e., $\{\theta\neq\theta_0\}$), we take the integration as usual

For simplicity, let us denote $m_1 = \int_{\{\theta\neq\theta_0\}} f(x|\theta)\xi(\theta)d\theta$

Then, the posterior can be determined as:
$$
\begin{aligned}
\pi(\theta|x) &= \frac{\pi_0 f(x|\theta_0)}{\pi_0 f(x|\theta_0) + \pi_1m_1(x)} \\ \\
&= \left(1 + \frac{\pi_1}{\pi_0}\cdot\frac{m_1(x)}{f(x|\theta_0)}\right)^{-1}
\end{aligned}
$$
The Bayes factor, $B_{01}$ is therefore defined as:
$$
B_{01} = \frac{f(x|\theta_0)}{m_1(x)}
$$
- Simply put, this is likelihood evaluated $\theta_0$ divided by the marginal distribution for part $\xi$ of the prior. 

#### Calibration

After calculating the Bayes Factor, we need to determine if there is sufficient evidence for or against our null hypothesis, $H_0$. Recall that $B_{10}$ refers to the Bayes Factor in favour of $H_1$.

The table below is given by Harold Jeffreys (which we will mention later).

| Value                             | Evidence against $H_0$ |
| --------------------------------- | ---------------------- |
| $0 \leq \log_{10}B_{10} \leq 0.5$ | Poor                   |
| $0.5 \leq \log_{10}B_{10} \leq 1$ | Substantial            |
| $1 \leq \log_{10}B_{10} \leq 1.5$ | Strong                 |
| $1.5 \leq \log_{10}B_{10} \leq 2$ | Very Strong            |
| $\log_{10}B_{10} \geq 2$          | Decisive               |
#### Examples
##### Jeremy's IQ

Let us take Jeremy's IQ as the example yet again. Test the hypotheses:
$$ H_0: \theta \leq 100 \quad \text{v.s.} \quad H_1: \theta > 100$$
Recall that with the Jeremy IQ example, the posterior is $\theta|x \sim N(102.8, 48)$.

In order to calculate the posterior probabilities, we simply integrate.
$$
\large{
\begin{aligned}
p_0 &= \int_{-\infty}^{100} \frac{1}{\sqrt{2\pi\cdot48}}\cdot e^{-\frac{(\theta-102.8)^2}{2\cdot48}} d\theta \\ \\
&= 0.3431 \\ \\
p_1 &= 1- 0.3431 = 0.6569
\end{aligned}}
$$
> We can calculate $p_0$ programmatically by calling `normcdf(100, 102.8, sqrt(48))`

Since the hypotheses are complementary, we can simply take $p_1$ to be $1-p_0$. Note that this might not be true for all scenarios, in which case we would have to perform integration to calculate $p_1$ as well.

Since $p_1$ is larger than $p_0$, we would "prefer" the alternative hypothesis, and say that $\theta > 100$. Of course, we will still need to determine the evidence in favour of either hypothesis (this is equivalent to calculating the p-value).

For that, we would need the prior probabilities:
$$
\large{
\begin{aligned}
\pi_0 &= \int_{-\infty}^{100} \frac{1}{\sqrt{2\pi\cdot120}}\cdot e^{-\frac{(\theta-110)^2}{2\cdot120}} d\theta \\ \\
&= 0.1807 \\ \\
p_1 &= 1- 0.1807 = 0.8193
\end{aligned}}
$$

Calculating the Bayes Factor $B_{10}$, 
$$
B_{10} = \frac{p_1/p_0}{\pi_1/\pi_0} = \frac{0.6569/0.3431}{0.8193/0.1807} = 0.4223
$$
Then, the calibration is:
$$ \log_{10}B_{01} = -\log_{10}B_{10} = 0.3744 $$
which is poor evidence against the $H_1$. Sure, $H_1$ is preferred _a priori_ and _a posteriori_, but the evidence is not high enough.

##### 10 Coin Flips; Precise Hypothesis

Here we will use the 10 coin flips example once more. We will be attempting to test a precise hypothesis. 

Recall that coin flips can be modelled as $X|p \sim Bin(n, p)$, where $p$ is the probability of heads. $p$ in turned can be modelled as $p \sim Be(500, 500)$.

We performed 10 coin flips ($n=10$) and realised that $X = 0$ (no heads at all). We had also evaluated our posterior to be $p|X \sim Be(500, 510)$. 

We have also found Bayes estimators in previous subsections:
- The posterior mean is 0.4950495, and
- The posterior mode is 0.4950397
- The median again is not explicit, and requires quantile analysis. We do this by calling `betainv(0.5, 500,510)` in Matlab, which gives us 0.4950462
> As an aside, if $\alpha$ and $\beta$ hyperparameters are very large, we can actually approximate the median by $\frac{\alpha - 1/3}{\alpha + \beta - 2/3}$

Let us first test the following hypotheses (non-precise):
$$ H_0: p \leq 0.5 \quad \text{v.s.} \quad H_1: p > 0.5$$
We can then calculate the prior and posterior probabilities.
$$
\begin{aligned}
p_0 &= \int_0^{0.5} \frac{1}{B(500, 500)}p^{500-1}(1-p)^{510-1}dp \\
&= \text{betacdf}(0.5, 500, 510) \\
&= 0.6235 \\ \\
p_1 &= 1 - p_0 = 0.3765 \\ \\
\pi_0 &= \int_0^{0.5} \frac{1}{B(500, 500)}p^{500-1}(1-p)^{500-1}dp \\\
&= \text{betacdf}(0.5, 500, 510) \\
&= 0.5 \\ \\
\pi_1 &= 1 - \pi_0 = 0.5 \\ \\
\end{aligned}
$$
Calculating the Bayes factor and the calibration, we get:
$$
\begin{aligned}
B_{01} = \frac{p_0/p_1}{\pi_0/\pi_1} = 1.656 \\ \\
\log_{10}B_{01} = 0.2191
\end{aligned}
$$
We see that we have poor evidence against $H_1$. 
> Again, $B_{01}$ means that we are looking at the Bayes Factor in favour of $H_0$, and against $H_1$. we say that the above has poor evidence against $H_1$. 

That was merely a repeat exercise for what was already done in the Jeremy's IQ example. Let us now take a look at a precise null hypothesis:
$$ H_0: p = 0.5 \quad \text{v.s.} \quad H_1: p \neq  0.5$$Suppose we set:
$$ \pi(p) = 0.8\cdot \delta_{0.5} + 0.2 \cdot Be(500, 500)$$
From the above, it is easy to see that
$$ \pi_0 = 0.8, \quad \pi_1=0.2$$
Now let us evaluate the marginal of the distribution outside of $\delta_{p=0.5}$
$$
\begin{aligned}
m_1(x) \rvert_{X=0} &= m_1(0) \\ \\
&= \int_0^1 {10\choose0}p^0(1-p)^{10} \cdot\frac{1}{B(500, 500)}p^{500-1}(1-p)^{500-1} dp \\ \\
&= \frac{B(500, 510)}{B(500, 500)} \\ \\
&= 0.001021
\end{aligned}
$$
> First, notice that we are taking the marginal at $X=0$. Next, the marginal can be broken down into two components: (i) the likelihood as defined by Binomial distribution, and the (ii) prior as defined by the Beta distribution. These two components have been separated by a dot, $\cdot$

The likelihood at $p=0.5$ is calculated as:
$$ 
\begin{aligned}
f(x|p) \rvert_{X=0, \,p=0.5} &= f(0|0.5) \\ \\
&= {10\choose5}0.5^0\cdot 0.5^{10} = \frac{1}{1024} = 0.0009765
\end{aligned}
$$
Finally, recall that the formula for the Bayes Factor, $B_{10}$ is:
$$
B_{10} = \frac{f(0|0.5)}{m_1(0)} = \frac{0.0009765}{0.001021} = 0.9564
$$
The calibration is:
$$ \log_{10}B_{01}= -\log_{10}B_{10} = 0.0194$$
This shows that there is very poor evidence **against $H_0$.

### Bayesian Prediction

Recall that the marginal $m(x)$ is defined as $\int f(x|\theta)\pi(\theta)d\theta$. This is also sometimes called the **prior predictive distribution** (because it involves the distribution of $x$'s).

If we take the integral of the likelihood with respect to the posterior, then this is called the **posterior predictive distribution**. This is defined as:
$$ f(x_{n+1}|x_1, \cdots, x_n) = \int f(x_{n+1}|\theta)\pi(\theta|x_1,\cdots,x_n)d\theta$$
- Notice here that the argument is denoted $x_{n+1}$ instead of $x_n$. 
- This means that we are about to take the $n+1$ observation, given that we already have $n$ observations. 
- We may then predict -- using the posterior predictive distribution (previous $n$ observations) -- the expected value and the distribution of this $n+1$ observation.

Then, the very prediction itself is simply the **predictive mean**, $\hat{X}_{n+1}$.
$$\hat{X}_{n+1} = \int x_{n+1} \times f(x_{n+1}|x_1, \cdots, x_n) dx_{n+1} = E^fX_{n+1}$$
The **predictive variance** is defined in the usual way: the difference squared integrated with respect to the posterior predictive distribution.
$$ \int (x_{n+1} - \hat{X}_{n+1})^2 \times f(x_{n+1}|x_1, \cdots, x_n) dx_{n+1} $$
#### Example

Suppose that we observed $n$ exponential random variables, that is $x_1, \cdots, x_n \sim Exp(\lambda)$. We put a prior on the parameter $\lambda$ with hyperparameters $\alpha$ and $\beta$, such that $\pi(\lambda) = \frac{\beta^{\alpha}\lambda^{\alpha-1}}{\Gamma(\alpha)}e^{-\beta\alpha}$, where $\lambda \geq 0$.

We have previously shown that the posterior distribution is:
$$
\begin{aligned}
\pi(\lambda|x_1, \cdots, x_n) &\propto \lambda^{n+\alpha-1}\text{exp}\{-(\sum_{i=1}^mx_i+\beta)\lambda \} \\ \\
&\sim Ga(\alpha + n, \beta + \sum x_i)
\end{aligned}
$$The posterior predictive distribution is then the integral of the product between the likelihood and the posterior distribution. In other words:
$$
f(x_{n+1} \mid x_1, \cdots, x_n) = \int \lambda e^{-\lambda x_{n+1}} \pi(\lambda| x_1, \cdots, x_n) d\lambda = \frac{(n + \alpha)(\sum_{i=1}^{m} x_i + \beta)^{n + \alpha}}{(\sum_{i=1}^{m} x_i + \beta + x_{n+1})^{n + \alpha + 1}}, \quad x_{n+1} > 0
$$
Note that this appears to be a Pareto distribution. 

For convenience, the Pareto distribution is given here by:
$$
f_X(x) = \begin{cases}
\frac{\alpha}{c} (\frac{c}{x})^{\alpha+1} \quad x \geq c, \\
0, \quad\quad\quad\quad\quad \text{else}
\end{cases}
$$

Rewriting the Pareto distribution, we get:
$$
x_{n+1} + \sum x_i + \beta \sim Pa(\sum_{i=1}^{m} x_i + \beta, n + \alpha)
$$
> Try and match term $x_{n+1} + \sum x_i + \beta$ to the general formula for the Pareto distribution given above.

The expectation is simply:
$$ \frac{\alpha c}{\alpha-1}, \quad \alpha > 1 $$
And the variance is:
$$ \frac{\alpha c^2}{(\alpha -1)^2(\alpha-2)}, \quad \alpha >2$$

We still have not gotten to the crux of this example: what exactly would be our prediction?

The prediction would be:
$$
\hat{X}_{n+1} = \mathbb{E}X_{n+1} = \frac{\left(\sum_{i=1}^{m} x_i + \beta\right)(n + \alpha)}{n + \alpha - 1} - \sum x_i - \beta = \frac{\sum_{i=1}^{m} x_i + \beta}{n + \alpha - 1}
$$

As an exercise, show that the variance for the predicted $X_{n+1}$ is $\large{\frac{(\sum_{i=1}^{m} x_i + \beta)^2(n + \alpha)}{(n + \alpha - 1)^2(n + \alpha - 2)}}$.

Now, let us concretise the above with real values: $x_1 = 2.1$, $x_2 = 5.5$, $x_3 = 6.4$, $x_4 = 8.7$, $x_5 = 4.9$, $x_6 = 5.1$, and $x_7 = 2.3$ and $\lambda \sim Ga(2, 1)$. 

Then show that $\hat{X}_8 = \frac{9}{2}$ and $\hat{\sigma}_{\hat{x}_8}^2 = 26.0357$.

It might be helpful to know that if only $\hat{X}_{n+1}$ is wanted, 
$$
\hat{X}_{n+1} = \int_{\Theta} \mu(\theta)\pi(\theta|x_1, \cdots, x_n) d\theta
$$
where $\mu(\theta) = \int x f(x|\theta) dx$ is the mean of $X$.