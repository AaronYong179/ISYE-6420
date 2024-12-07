## Introduction and Overview

In real life, we might have to deal with missing or censored data. Here is where Bayesian statistics shines in strength. The taxonomy (or maybe, hierarchy?) of missingness is as follows:

1. Missing Completely at Random (MCAR)
   - Missingness does not depend on observed or unobserved data.
2. Missing at Random (MAR)
   - Missingness depends only on the observed data.
3. Missing not at Random (MNAR)
   - Missingness may depend on data that is missing (e.g., data magnitude)

The first two, MCAR and MAR, are easy to deal with. MNAR not so much. This is because the value of the missing data might _cause_ the data to be missing in the first place. MCAR and MAR are also known as **ignorable missingness**, while MNAR is known as **non-ignorable missingness**.

## Multiple Imputations

To alleviate the problem of missing data, Rubin D. B. suggested **multiple imputations**. Simply put, we have data $Y$ the consists of $(Y_{\text{obs}}, Y_{\text{mis}})$.

We can then think about the important characteristics of the posterior distribution of $\theta$, all while taking into account missing and observed data.

The posterior distribution of $\theta$ can be given as the average of the complete data posterior distribution of $\theta$:
$$
P(\theta|Y_{\text{obs}}) = \int P(\theta|Y_{\text{obs}}, Y_{\text{mis}})P(Y_{\text{mis}}|Y_{\text{obs}}) \, dY_{\text{mis}}
$$

The posterior mean of $\theta$ can be computed by averaging repeated complete data posterior means of $\theta$:
$$
E(\theta|Y_{\text{obs}}) = E(E(\theta|Y_{\text{obs}}, Y_{\text{mis}})|Y_{\text{obs}})
$$

The posterior variance of $\theta$ can be computed by averaging repeated complete data variances of $\theta$, added to the variance of repeated complete data posterior means of $\theta$. More succinctly:
$$
Var(\theta|Y_{\text{obs}}) = E(Var(\theta|Y_{\text{obs}}, Y_{\text{mis}})|Y_{\text{obs}}) + Var(E(\theta|Y_{\text{obs}}, Y_{\text{mis}})|Y_{\text{obs}})
$$

More concretely, we would **impute** (insert possible values) into the missing data and compute various simulations of $\hat{\theta}$. An estimator of the parameter $\theta$ is then computed at the end. The image below depicts this more clearly:

![[Pasted image 20241104175510.png|500]]

Notice that this is extremely straightforward to do in a Bayesian manner:
1. Simply simulate $\theta_1^*, \theta_2^*, \cdots, \theta_M^*$ from the posterior derived wrt observed data. $M$ here is the number of simulations that one wishes to take.
2. Then, simulate the missing data from the likelihood $f(y|\theta_i^*)$. Notice that we do this for all $\theta_i^*$.  
## Time-to-Event Models

This is a more focused subtopic, particularly with regards to "time" as a variable. These models are often applied in reliability theory (when does something fail?), survival theory (when does something die?), or geoscientific prediction. 

### Lifetime Distributions

Let $T$ be the lifetime, or time-to-event. Naturally $T \geq 0$.

The CDF is then $F(t) = P(T\leq t)$. The survival function $S(t)$ is often the complement of the CDF, that is $1 - F(t) = P(T\geq t)$.

The density of $T$, $f(t)$, is simply the derivative of the CDF. However, since the survival function is also connected to the CDF, we have that $f(t) = \frac{dF(t)}{dt} = -\frac{dS(t)}{dt}$.

The probability that a lifetime is between $a$ and $b$ can be calculated as:
$$
P(a \leq T \leq b) = F(b) - F(a) = S(a) - S(b)
$$
#### Hazard Function

Importantly, modeling lifetimes often sees the use of a hazard function $h(t)$. Let's first understand the notion of "hazard". Basically, it is the conditional probability that a device will fail in between some interval $t$ and $t + dt$, given that it is working at time $t$. Formally:
$$
h(t) dt = (t\leq T\leq t + dt|t\geq t) = \frac{S(t)-S(t+dt)}{S(t)}
$$
If we divide the above equation with $dt$ and let $dt$ approach 0, we get:
$$
h(t) = \lim_{dt\rightarrow0} (\frac{S(t)-S(t+dt)}{dt}\cdot\frac{1}{S(t)}) = -\frac{S(t)'}{S(t)} = -\log S(t)'
$$
- Note that $S(t)'$ stands for the derivative of $S(t)$.

However, we have also defined the negative derivative of the survival function as simply the density of $T$. Finally, we arrive at the classic definition of a hazard function:
$$
f(t) = -\frac{S(t)'}{S(t)} = \frac{f(t)}{S(t)}
$$

It should be obvious, but do note the difference between the density and hazard function (the hazard is a conditional probability):
$$
\begin{aligned}
f(t)dt &= P(t\leq T\leq t + dt) \\ \\
h(t)dt &= P(t\leq T\leq t + dt |T\geq dt)
\end{aligned}
$$

The **cumulative hazard** function $H(t)$ is defined as:
$$
\begin{aligned}
H(t) &= \int_0^t h(u)du = -\int_0^t (\log S(u))' du = -\log S(t) + 0 \\ \\
&\Rightarrow S(t) = e^{-H(t)}
\end{aligned}
$$
### Censoring

In time-to-event models, missing data often manifests as "censored" data. Consider the figure below:

![[Pasted image 20241104203627.png|400]]

Suppose we are interested in the lifetime $Y_i$ of some device. Unfortunately, the experiment is made to stop at time $C$, which results in $Y_3$ and $Y_4$ being censored data. The rest are fully observed.

We could indicate this with a simple binary variable, termed the **censoring indicator**. Let $\delta = 0$ if the data is fully observed and $\delta=1$ otherwise. Given the example above, we have:

| Time     | $Y_1$ | $Y_2$ | $Y_3$ | $Y_4$ | $Y_5$ |
| -------- | ----- | ----- | ----- | ----- | ----- |
| $\delta$ | 0     | 0     | 1     | 0     | 1     |
#### More on Censoring

It is typical for most censored data to be **right-censored**, as shown in the example. 

There are also two main types of censoring:
1. Type I, where the censoring time is fixed but the number of observed data (censored or otherwise) is random. This is most common.
2. Type II, where the censoring time is random but the number of observed is fixed. Think of this as "I will run the experiment until I observe five lifetimes" or "I will run the experiment until I observe 50% failure".
#### Likelihood with Censored Observations

Suppose that we have the model $Y_i \sim f(y_i|\theta)$, where $i = 1, \cdots, n$. 

However, only $k$ of them are observed. In other words, we have $\delta_i = 0$ for $i = 1, \cdots, k$ and $\delta_i=1$ for $i=k+1, \cdots, n$.

The likelihood is provided below:
$$
L(\theta|y_1,\cdots,y_n) = \prod_{i=1}^n(f(y_i|\theta))^{1-\delta_i} \cdot(S(y_i|\theta))^{\delta_i}
$$
- Notice that using $\delta_i$ as a power is rather elegant.
	- Consider the first term involving the density $f$. We are simply selecting for data that is observed here (i.e., $\delta_i = 0$).
	- The second term selects for data that is missing (i.e., $\delta_i = 1$)
- Recall from above that $S(y_i|\theta)$ is essentially saying $P^{\theta}(Y_i > y_i)$. We did not observe these data, but we know for a fact that they are larger (by virtue of being right-censored).

Then the Bayesian inference about $\theta$ proceeds in the traditional way (we already have the likelihood above, and we just need to put some prior on $\theta$) 

### Examples

#### Example 1

Suppose that $Y_1, Y_2, \cdots, Y_n \sim \mathcal{E}(\lambda)$. In other words, these are exponentials. Again, we suppose that only $k$ of them are observed. We have $\delta_i = 0$ for $i = 1, \cdots, k$ and $\delta_i=1$ for $i=k+1, \cdots, n$.
##### Frequentist
What would be the frequentist estimator of $\lambda$? Let us first assume that we have all data observed. In that case, the estimator of $\lambda$ is quite simply:
$$
\frac{n}{\sum_{i=1}^n Y_i} = \frac{1}{\bar{Y}}
$$

What happens then if we have censored data? Shall we ignore them? This is incorrect, as the estimator is biased:
$$
\hat{\lambda} = \frac{k}{\sum_{i=1}^kY_i} \quad \text{[WRONG]}
$$

Shall we consider censored data as observed? This is also incorrect, as the censored data are likely larger than the observed ones:
$$
\frac{n}{\sum_{i=1}^n Y_i} = \frac{1}{\bar{Y}} \quad \text{[WRONG]}
$$

The correct method is to first take the likelihood of the data:
$$
L(\theta|y_1, \cdots, y_n) = \lambda^ke^{-(\lambda \sum_{i=1}^n Y_i)}
$$
The MLE is then:
$$
\hat{\lambda} = \frac{k}{\sum_{i=1}^nY_i} = \frac{k}{n\bar{Y}} \quad \text{[CORRECT]}
$$

Details of the correct frequentist approach is left as an exercise. 

#### Example 2

Suppose that we have three observations: $Y_1 = 2, Y_2 = 3, Y_3 = 1*, Y_4=\frac{5}{2}, Y_5=3*$. Those marked with an asterisk are censored data -- this is how it is typically notated.

Assume that $Y_i  \sim Wei(v, \lambda)$ and that we know $v = \frac{3}{2}$. Using prior on $\lambda \sim Ga(2, 3)$, estimate $\lambda$.

We have the following:
$$
\begin{aligned}
f(y_i|v,\lambda) &= v\lambda y_i^{v-1}e^{-\lambda y_i^v}, \quad y_i\geq0 \\ \\
S(y_i|v,\lambda) &= e^{-\lambda y_i^v}, \quad y_i\geq0
\end{aligned}
$$

Since we have censored data, the likelihood is therefore:
$$
\prod_{i=1}^{k} \nu \lambda y_i^{\nu - 1} e^{-\lambda y_i^{\nu}} \prod_{i=k+1}^{n} e^{-\lambda y_i^{\nu}} = \nu^k \lambda^k \left( \prod_{i=1}^{k} y_i \right)^{\nu - 1} e^{-\lambda \sum_{i=1}^{n} y_i^{\nu}}
$$
If we eliminate all constants, we notice that the likelihood shares kernels with a Gamma distribution. 
$$
L(\lambda|v, y_1,\cdots,y_n) \propto \lambda^ke^{-\lambda\sum_{i=1}^{n} y_i^{v}}
$$
We have a gamma prior on $\lambda$. Let us give the general form first:
$$
\pi(\lambda|\alpha,\beta) \propto \lambda^{\alpha-1}e^{-\beta\lambda}
$$
These form conjugate pairs. Therefore, we update the posterior to be:
$$
\pi(\lambda \mid \nu, y_1, \ldots, y_n, \alpha, \beta) \propto \lambda^{k + \alpha - 1} e^{-\lambda \left( \beta + \sum_{i=1}^{n} y_i^{\nu} \right)}
$$
Finally, we sub in the known values to get the posterior as modeled by $Ga(3+2, 18.1736 + 3)$


