## Prior Elicitation

### Preamble

In all of the previous lectures, we have _always_ assumed knowledge of the priors -- this information was always freely given. Here, we shall focus on how to elicit an unknown prior.

Priors are popularly called the sword and Achilles heel of Bayesian inference.
- Priors are powerful (sword-like) because they are able to incorporate previous information or perhaps information from an expert.
- At the same time, the mis-specification of the prior may impair and affect the inference made. This was the main criticism of Bayesian inference -- priors are subjective, therefore the whole inference is not really objective.

I'm not sure why this is relevant, but the lecturer raises a quote by Garthwhite & Dickey:
> ... expert personal opinion is of great potential value and can be used more efficiently, communicated more accurately, and judged more critically if it is expressed as a probability distribution.
### Methods of Prior Elicitation

There are multiple ways to elicit priors and this section will touch on a few.
#### Known Family, Known Numerical Characteristic

If we are sure about the family of distributions that the prior belongs to and some (but not all) numerical characteristics (e.g., variance, higher moments, quantiles, modes, ...), use those to specify the prior.

##### Known Mean
As an example, suppose we know that the prior is exponential for the parameter $\theta$. Suppose also that we elicited $\theta$ to be 2. Then theoretically, $\frac{1}{\lambda} = 2$ that is $\lambda = \frac{1}{2}$.
##### Known Median
Alternatively, suppose that we know the prior is exponential for the parameter $\theta$. Suppose also that we know the median of $\lambda$ to be 4.
- This means that 50% of the time, $\lambda < 4$ and 50% of the time, $\lambda > 4$.
- In other words, $F(4) = 1/2$, where $F$ here is the exponential distribution. 
- Substituting the values into the formula for an exponential distribution, we get $\frac{1}{2} = 1 - e^{-4\lambda}$.
- Solving for $\lambda$, we get 0.1733
##### Known Mean and Variance
Let us elicit a beta prior on $\theta$ if we know that the elicited $\theta = \frac{1}{2}$ and the variance of $\theta = \frac{1}{8}$.

Specifying the problem exactly, we are trying to determine $\alpha$ and $\beta$ hyperparameters of the prior on $\theta \sim Be(\alpha, \beta)$.

For beta distributions, the expectation is given as $\frac{\alpha}{\alpha + \beta}$. Equating this to $\frac{1}{2}$, we get that $\alpha = \beta$.

We also know that the variance of beta distributions is given as $\frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}$. Solving for $\alpha$ and $\beta$, we get:
$$
\begin{aligned}
\frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)} &=  \frac{\alpha^2}{4\alpha^2(2\alpha+1)} = \frac{1}{4(2\alpha + 1)} \\ \\
\frac{1}{4(2\alpha + 1)} &= \frac{1}{8} \Rightarrow \alpha = \beta = \frac{1}{2}
\end{aligned}$$
Note that in general, if $\mathbb{E}(\theta) = \mu$ and $\text{Var}(\theta) = \sigma^2$, then we can simply elicit the hyperparameters of the Beta distribution using these formulas:
$$ \alpha = \mu \left(\frac{\mu(1-\mu)}{\sigma^2} - 1\right); \quad\beta=(1-\mu) \left(\frac{\mu(1-\mu)}{\sigma^2} - 1\right)$$

#### Non-Informative Priors

What happens if we are unwilling to specify any information regarding the priors? This leads to a rather popular subdomain of Bayesian statistics -- the eliciting of non-informative priors. Although termed "non-informative", we will see shortly that this term is vaguely defined and we can frame this problem in many ways.

The elicitation of non-informative priors was driven by the criticism of Bayesian methodology; that is, priors are subjective in nature. With informative priors, there is a possibility that any incorrect information might poison the inference as well.
##### Invariance Principle for Priors

Let $X|\theta \sim f(x-\theta)$; density is a function of $(x-\theta)$; $\theta$ is the **location** parameter.

We would like to elicit a prior such that the **location** is invariant. In other words, $\pi(\theta) - \pi(\theta-\theta_0)$ for any $\theta_0$. 
- Notice that the solution to this involves a constant prior. This is often called a **flat prior**.

The above was for location parameters. Now if the parameter of interest is a **scale** parameter, $X|\theta \sim \frac{1}{\theta}f(\frac{x}{\theta})$, then the invariance principle suggest $\pi(\theta) \sim \frac{1}{c} \pi\left(\frac{\theta}{c}\right)$
- What does "scale" mean? Basically, look for $x$ being divided by the parameter (you may think of it as normalising $x$)
- The choice that satisfies scale invariance is $\pi(\theta) = \frac{1}{\theta}, \, \theta > 0$.

Note that both these priors are improper (i.e., not bona-fide densities). In other words, the integral of these priors are not equal to 1. In fact:
$$\int_{\mathbb{R}} c \, d\theta = \infty; \quad \int_{0}^{+\infty} \frac{1}{\theta} d\theta = \infty$$
- Generally, this property does not present any difficulties. We seldom integrate priors by themselves in Bayesian analysis anyway, and the posteriors could be (and most of the time are) proper densities.
##### Jeffreys' Priors

The Jeffreys' Priors are very popular, and they are based on the likelihood. More specifically, Fisher Information, $I(\theta)$ is used:
$$ I(\theta) = -\mathbb{E}^{x/\theta} \left( \frac{\partial^2 \log f(x|\theta)}{\partial \theta^2} \right)
$$
> We are taking the expectation with respect to the likelihood $\mathbb{E}^{x|\theta}$ of the second derivative of the log of the likelihood.  

Jeffreys suggested that a non-informative prior should be proportional to the square root of determinant of the Fisher Information:
$$ \pi(\theta) \propto \det(I(\theta))^{1/2} $$
- Note that if the parameter is univariate (single-dimension), the Fisher Information is simply a constant.

Jeffreys wanted the prior to be invariant with respect to any transformation on the parameter $\theta$. 
- Simply put, suppose that $h$ and $g$ are transformations. $h(\theta) = \phi$ and $g(\phi) = \theta$ -- both transformations map $\theta$ and $\phi$ to and from each other.
- To satisfy this invariance, $I^{1/2}(\phi) = I^{1/2}(\theta) \times \left| \frac{d\theta}{d\phi} \right|$
- Basically, even if $\theta$ is transformed into $\phi$, both values should equate (multiplied by the absolute value of the Jacobian on the right).
###### Important Jeffreys' Priors

1. If $x|\theta \sim N(\theta, \sigma^2)$ and $\sigma^2$ is known (unknown mean):
	- $\pi(\theta) \propto 1$ 
2. If $x|\theta \sim N(\mu, \theta)$ and $\mu$ is known (unknown variance):
	- $\pi(\theta) = \frac{1}{\theta}$
3. If $x|\theta \sim Poi(\theta)$ (unknown rate):
	- $\pi(\theta) = \frac{1}{\sqrt{\theta}}$
4. If $x|\theta \sim Bin(n, \theta)$ and $n$ is known (unknown probability):
	- $\pi(\theta) \propto \theta^{-\frac{1}{2}}(1-\theta)^{-\frac{1}{2}} \sim Be(\frac{1}{2}, \frac{1}{2})$
5. If $x|\theta \sim N(\mu, \theta^2)$ and $\mu$ is known (unknown standard deviation):
	- $\pi(\theta) = \frac{1}{\theta}$
	- $\log\theta$ is uniform on the real line. Therefore $\log\theta^2 = 2\log\theta$ is also uniform on the real line.
6. $x|\theta \sim Bin(n, \theta)$. But we put a flat prior on the logit${(\theta)}$. 
	- logit${(\theta)}$ is defined as $\log\frac{\theta}{1-\theta}$
	- Here the prior is given by Zellner's prior, $\pi(\theta) \propto \theta^{-1}(1-\theta)^{-1}$
	- Now this form appears to be similar to that of a Beta distribution, but the degrees (power) of the Beta distribution must be greater than $-1$.

##### Objective (Reference) Priors

A more modern take on non-informative priors was started by a Spanish statistician, Bernardo. They are also known as **reference priors**. This method involves maximising the divergence (measure of distance) between the prior and the posterior.
- The rationale for this is as follows: the larger the divergence between the prior and the posterior, $\theta$ is more influential.

There are many ways one can measure distance, but the KL-divergence is shown here.

For the prior and posterior, the KL-divergence looks like:
$$
\int \pi(\theta \mid t) \log \frac{\pi(\theta \mid t)}{\pi(\theta)}
$$
$t$ here represents the sufficient statistic, $t=t\left(x_1, \ldots, x_n\right)$. Recall that a sufficient statistic contains all information needed to compute any estimate of the parameter.

This is fine, but we don't have $\theta$ yet -- how do we go about maximising the distance without $\theta$?

We can maximise the expected distance instead. More formally:

$$
\begin{aligned}
\mathrm{I}&=\int \mathrm{m}(t)\left(\int \pi(\theta \mid t) \log \frac{\pi(\theta \mid t)}{\pi(\theta)} d \theta\right) d t \\
&=\iint \mathrm{h}(t, \theta) \log \frac{\mathrm{h}(t, \theta)}{\mathrm{m}(t) \pi(\theta)} d \theta d t \\ \\
\pi^*(\theta)&=\underset{\pi(\theta)}{\arg \max } \mathrm{I}
\end{aligned}
$$
The second line here is simply a different representation -- this representation is also called **mutual information** between $t$ and $\theta$. It is therefore also possible to view objective priors as maximising the information between data and the parameter $\theta$.

For one-dimensional parameters (i.e., most of the parameters encountered in this course), the reference priors and Jeffreys' priors coincide.

### Prior Sample Sizes

The notion of "sample sizes" in priors arose as an attempt to address the following question: "how do we calibrate the amount of information carried by the prior?"

For example, consider two beta-binomial conjugate models for successful coin flips as depicted below:

![[Pasted image 20240922174443.png]]

Both have a mean of 0.5, but one is clearly much stronger. Recall from previous units that the lecturer uses the parameters of values 500.

Effective sample size (ESS), as defined in the lecture, is an informal measure of the strength of a prior as it interacts with a particular likelihood to contribute to the posterior. Using the examples above, we would say that the ESS of $Be(10, 10)$ is $20$, while the ESS of $Be(500, 500)$ is $1000$.

I think of it as how the input parameters modify the actual sample size. A few distributions and their respective ESS are given below:
- For $X|\theta \sim Bin(n, \theta)$ and $\theta \sim Be(\alpha, \beta)$
$$
\frac{\alpha}{\alpha + \beta} \rightarrow \frac{\alpha + x}{\alpha + \beta + n} \Rightarrow \text{ESS} = \alpha + \beta
$$
- For $X|\theta \sim Poi(\theta)$ and $\theta \sim Ga(\alpha, \beta)$
$$\frac{\alpha}{\beta} \rightarrow \frac{\sum X_i +\alpha}{\beta + n} \Rightarrow \text{ESS} = \beta$$
- For $X|\theta \sim N(\mu, \theta^{-1})$, where $\theta$ is the precision ($\frac{1}{\sigma^2}$),  and $\theta \sim Ga(\alpha, \beta)$
$$\Rightarrow \text{ESS} = 2\alpha$$
The final ESS is more involved. For starters, the mean of posterior gamma distribution can be shown to be:
$$
\frac{\alpha}{\beta} \rightarrow \frac{\alpha + n/2}{\beta + 1/2 \cdot \sum_{i=1}^n (x_i - n)^2} \rightarrow \text{ESS} = 2\alpha
$$
The professor appears to be informally comparing the prior mean to the posterior mean in all cases.

#### Community of Priors

Another way to cope with the informativeness of priors would be to have a family (community) of priors instead. This was suggested by Spiegelhalter. We can therefore have:
- Vague, non-informative priors
- Skeptical priors that should be adopted for the reviewer (say, an auditing body)
- Enthusiastic priors that should be adopted for the proposer (say, the research team)

## Empirical Bayes

Empirical Bayes was divided into two approaches: parametric and non-parametric by Carl Morris (1983, JASA paper).

### Parametric Approach

Suppose we have the following:
$$
\begin{aligned}
&X_i|\theta_i \stackrel{ind}{\sim} f_i(x_i|\theta_i), \quad i = 1, 2,\cdots, n \\ \\
&\theta_i \stackrel{iid}{\sim} \pi(\theta_i|\xi)
\end{aligned}
$$
Note that the likelihoods are only independent (and not identical). For one, the function $f_i$ might be different. Also, the parameter $\theta_i$ might be different for each $i$ as well. $\xi$ here is just some common hyperparameter.

Then, the marginal by definition is $\int f_i(x_i|\theta_i) \cdot \pi(\theta_i|\xi) d\theta_i$

Let us denote all observations $x_i$ from $1$ to $n$ as $\tilde{x}$. Then, 
$$
\begin{aligned}
m(\tilde{x}|\xi) &= \int \prod_{i=1}^n f_i(x_i|\theta_i) \cdot \prod_{i=1}^n \pi(\theta_i|\xi)d\theta_1 \cdots d\theta_n \\ \\
&=\prod_{i=1}^n \int f_i(x_i|\theta_i) \cdot \pi(\theta_i|\xi)d\theta_i \\ \\
&= \prod_{i=1}^n m_i(x_i|\theta_i)
\end{aligned}
$$
Therefore, we can say that $X_i$ are marginally independent if $\theta_i \stackrel{iid}{\sim} \pi(\theta_i|\xi)$.

Additionally, if we further set $f_i$ to be the same $f$ for all $i$, then $X_i$ are $iid$ (marginally). Also, the posterior is:
$$
\pi(\theta_i|X_i, \xi) = \frac{f(x_i|\theta_i) \cdot \pi(\theta_i|xi)}{m(x_i|\xi)}
$$

More importantly, if the hyperparameter $\xi$ is unknown, it can be estimated from $X_1, X_2, \cdots, X_n$ via:
- Maximum Likelihood Estimation (MLE, also called MLE II)
- Moment Matching (MM)
#### Example: Jeremy's IQ

Let's assume that Jeremy took five IQ tests ($n=5$) with scores $98, 107, 89, 88, 108$. 

For this data, assume the following:
$$
\begin{aligned}
X_i|\theta_i &\stackrel{ind}{\sim} N(\theta_i, \sigma^2), \quad \sigma^2 = 80 \\ \\
\theta_i &\stackrel{iid}{\sim} N(\mu, \tau^2)
\end{aligned}
$$
The goal here is to estimate $\theta_i$s

Recall firstly that the marginal in the normal-normal conjugate case can be defined as:
$$
X_i \stackrel{iid}{\sim} N(\mu, \sigma^2 + \tau^2) 
$$
Also, recall from above that if the prior is identically and independently distributed, so will the marginals. In other words:
$$
\large
m(\tilde{x}|\mu, \tau^2) = \prod_{i=1}^n \frac{1}{2\pi(\sigma^2 + \tau^2)}e^{\frac{(x_i - \mu)^2}{2(\sigma^2 - \tau^2)}}
$$
Then, the MLE of $\mu$ is $\hat{\mu} = \bar{X}$ and of $\tau^2$ is $\hat{\tau}^2 = (s^2 - \sigma^2)_+ \equiv \text{max}\{0, s^2 - \sigma^2\}$, where $s^2$ is the sample variance for $X$.

With these estimators from the data, the estimated posterior becomes:
$$
\pi(\theta_i|X_i, \hat{\mu}, \hat{\tau}^2) = N(\hat{B}\hat{\mu} + (1-\hat{B})x_i, \,(1-\hat{B})\cdot\sigma^2)
$$
where $\hat{\mu} = \bar{X}$, $\hat{\tau}^2 = (s^2 - \sigma^2)_+$, and $\hat{B} = \frac{\sigma^2}{\sigma^2 + \hat{\tau}^2}$.

Thus, for Jeremy's data, since $s^2 = 101$, 
$$
\hat{B} = \frac{80}{101}
$$
Then, we are able to estimate the values of $\theta_i$ for all $i$. More explicitly put,
- $\theta_1 = \frac{80}{101} \cdot 98 + \frac{21}{101} \cdot 98 = 98$,
- $\theta_2 = \frac{80}{101} \cdot 98 + \frac{21}{101} \cdot 107 = 99.8713$, and so on

We may wonder why it is necessary to estimate each individual theta using all observations?
- We borrow strength from all observations, and weights themselves depend on all observations.
#### Example: Poisson, Exponential

Consider now another example. We have the following:
$$
\begin{aligned}
&X_i \sim \text{Pois}(\lambda_i), i=1, 2, \cdots, n \\
&\lambda_i \sim \text{Exp}(\mu), \quad \pi(\lambda_i) = \mu e^{-\mu\lambda_i}
\end{aligned}
$$
Find the Empirical Bayes estimators of $\lambda_i$.

Firstly, note that the posterior distribution is:
$$
\lambda_i |X_i \sim Ga(x_i+1, 1+\mu)
$$
> This is a conjugate case. Recall that the exponential distribution is really just a _special_ gamma distribution.

The expectation of the posterior is given as $\mathbb{E}(\lambda_i|X_i) = \frac{x_i+1}{1+\mu}$. Here, $\mu$ may not be known.

Simplifying the marginal:
$$
\begin{aligned}
m(x_i) &= \int_0^{+\infty} \frac{\lambda_i^{x_i}}{x_i!} e^{-\lambda_i} \cdot \mu e^{-\lambda_i \mu} d\lambda_i \\ \\
&= \frac{1}{(1 + \mu)^{x_i + 1}} \cdot \mu \int_0^{+\infty} \frac{(1 + \mu)^{x_i + 1} \lambda_i^{x_i}}{\Gamma(x_i+1)} e^{-(1 + \mu)\lambda_i} d\lambda_i \\ \\
&= \left(\frac{1}{1 + \mu}\right)^{x_i} \cdot \frac{\mu}{1 + \mu}, \quad x_i = 0, 1, \dots
\end{aligned}
$$
> Take some time to parse the algebra. The left hand side is simply added as a normalisation factor. The integral of a gamma distribution simply results in $1$. 

Finally, we end up with the geometric distribution! Denote $p = \frac{\mu}{1+\mu} \Rightarrow Ge(p)$ 

In the usual fashion, we then take the product of all individual marginals, which would give us this expression:
$$
\prod_{i=1}^n m(x_i) = (1-p)^{\sum x_i} \cdot p^n
$$
Taking the logarithm, we get:
$$
\sum x_i \cdot \log(1-p) + n \cdot \log p
$$
Taking the first derivative, we get:$$
-\frac{\sum x_i}{1-p} + \frac{n}{p} = 0
$$Solving for $p$ (or rather, finding the MLE of $p$, $\hat{p}$):
$$
\hat{p} = \frac{n}{n + \sum x_i} = \frac{1}{1+\bar{x}}
$$
Therefore, $\mu$ as estimated by the data is $\frac{1}{\bar{x}}$. We obtain that estimate by solving:
$$
\frac{\mu}{1+\mu} = \frac{1}{1+\bar{x}}
$$
We can therefore estimate $\hat{\lambda}_i$ as $\frac{\bar{X}}{1+\bar{X}}(X_i + 1)$

Let's take a few steps backwards to understand what we did. 
1. Firstly, the Bayes estimator (expectation of the posterior) was shown to be $\frac{X_i +1}{1+\mu}$. 
2. Then, we managed to estimate $\mu$ using the observed data. We denoted this $\hat{\mu}$. Our empirical Bayes estimator is therefore $\frac{X_i +1}{1+\hat{\mu}}$
3. Finally, we simplify the empirical Bayes estimator to get the final estimator for $\hat{\lambda}_i$

### Non-Parametric Approach

This approach is not generally applicable to most cases but will be covered regardless. Here, we assume only that parameters $\theta_i$ are $iid$, but no family of distribution is specified (as was the case with the parametric approach). We use data to estimate the marginal or the prior directly. This approach actually _precedes_ the parametric approach, and was pioneered by Herbert Robbins in the 1950s.

Let's take a closer look at how this works. Assume that we have Poisson distributed observations, that is $X_i |\lambda_i \sim \text{Pois}(\lambda_i)$, where $i = 1, \cdots, n$.

The only assumption we will make here is that the parameters are $iid$ with some prior:
$$
\lambda_i \stackrel{iid}{\sim} \pi(\lambda_i)
$$
If we were to formally write down the Bayes estimator of $\lambda_i$, we would get:
$$
\hat{\lambda}_i = \frac{\int \lambda_i \frac{\lambda_i^{x_i}}{x_i!} e^{-\lambda_i} \pi(\lambda_i) d\lambda_i}{\int \frac{\lambda_i^{x_i}}{x_i!} e^{-\lambda_i} \pi(\lambda_i) d\lambda_i} 
= \frac{(x_i + 1) \int \frac{\lambda_i^{x_i+1}}{(x_i + 1)!} e^{-\lambda_i} \pi(\lambda_i) d\lambda_i}{\int \frac{\lambda_i^{x_i}}{x_i!} e^{-\lambda_i} \pi(\lambda_i) d\lambda_i}
= (x_i + 1) \frac{m_{\pi}(x_i + 1)}{m_{\pi}(x_i)}

$$

Therefore, given $X_1, \cdots, X_n$, we attempt to estimate $m$ as $\hat{m}$ and use $\hat{m}$ in:
$$
(\hat{\lambda})_{EB} = (x_i + 1) \frac{\hat{m}_{\pi}(x_i + 1)}{\hat{m}_{\pi}(x_i)}
$$
There are a few ways to perform this estimation of $m$, where the most trivial would be to take $\hat{m}(x_i)$ as the relative frequency of $x_i$ in the data. More concretely, we trivially estimate $\hat{m}$ as:
$$
\frac{1}{n}\sum_{i=1}^n \mathbb{1}(X_i=x_i)
$$
Other (perhaps better) alternatives would be to use some kernel smoothers to estimate $m$. Often, we would also see the denominator being added to with some small number, to avoid a division by zero:
$$
(\hat{\lambda})_{EB} = (x_i + 1) \frac{\hat{m}_{\pi}(x_i + 1)}{\frac{1}{n}+\hat{m}_{\pi}(x_i)}
$$