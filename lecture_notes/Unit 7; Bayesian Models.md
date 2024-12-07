## Hierarchical Models

### Introduction

Suppose that we have some prior $\pi(\theta)$. This prior can be represented as the integral of the following chain of priors:
$$
\int \pi_1(\theta|\theta_1)\pi_2(\theta_1|\theta_2)\cdots\pi_n(\theta_{n-1}|\theta_n)\pi_{n+1}(\theta_n) \, d\theta_1d\theta_2,\cdots d\theta_n
$$
- Think of this as having $\theta_1$ as the hyperparameter of $\pi_1$. 
- $\theta_1$ has its own prior $\pi_2$ with hyperparameter $\theta_2$ and so on. 

If we look at the complete Bayes model, we notice that the chain of priors resolves to the simple prior given on the right-hand side.
$$
\left\{ \begin{array}{l} X \mid \theta \sim f(x \mid \theta) \\ \theta \mid \theta_1 \sim \pi_1(\theta \mid \theta_1) \\ \theta_1 \mid \theta_2 \sim \pi_2(\theta_1 \mid \theta_2) \\ \vdots \\ \theta_{n-1} \mid \theta_n \sim \pi_n(\theta_{n-1} \mid \theta_n) \\ \theta_n \sim \pi_{n+1}(\theta_n) \end{array} \right. \Leftrightarrow \left\{ \begin{array}{l} X \mid \theta \sim f(x \mid \theta) \\ \theta \sim \pi(\theta) \end{array} \right.
$$

Suppose we have the chain of priors and hyperparameters. Suppose also that we have an observation $X$. Then, we have:
$$ X \leftarrow \theta\leftarrow \theta_1\leftarrow \theta_2\leftarrow \cdots\leftarrow \theta_n$$
Recall that information flow is not constrained to the direction of the arrows. Rather, information will flow both ways. This means that when we observe $X$, all parameters $\theta, \theta_1, \cdots, \theta_n$ will be affected. 

Let's take a look at the joint distribution of $(X, \theta, \theta_1, \cdots, \theta_n)$:
$$
\begin{aligned}
f(X, \theta, \theta_1, \cdots, \theta_n) \propto\,\, &f(x|\theta, \theta_1, \cdots, \theta_n) \times \\
&\pi_1(\theta|\theta_1, \theta_2, \cdots, \theta_n) \times \\
&\pi_2(\theta_1|\theta_2, \cdots, \theta_n) \times \\
&\quad\quad\quad\quad\vdots \\
&\pi_n(\theta_{n-1}|\theta_n) \times \\
&\pi_{n+1}(\theta_n)
\end{aligned}
$$
But by the Markovian property, we get the following simplified form:
$$
f(X, \theta, \theta_1, \cdots, \theta_n) \propto f(x|\theta)\pi_1(\theta|\theta_1)\pi_2(\theta_1|\theta_2)\cdots\pi_n(\theta_{n-1}|\theta_n)\pi_{n+1}(\theta_n)
$$
### Motivation

The most obvious rationale for having the hierarchy of priors would be that the modeling requirements simply ask for hierarchy. For instance, Bayesian Meta-Analysis involves putting together different studies, precisely in this hierarchical manner.

Secondly, the priors may have some **structural** and **subjective/noninformative** components. We can separate these components out and place the subjective components at higher levels of the hierarchy. 

Hierarchy also offers robustness and objectivity. This rationale isn't really clearly explained, but the lecturer claims that having the data inform the hyperparameters is a good thing. 

Finally, a hierarchy of priors might be computationally easier to calculate compared to a single prior. This might not be too relevant today. 
#### Example

Let us take a look at an example of a hierarchical model. 

First, let us consider a model that we are familiar with at this point, one with a single parameter $\theta$:
![[Pasted image 20241026154726.png|150]]
As shown above, the posterior is proportional to the product of the likelihood and the prior. We already know this. 

If we have $\theta$-s that are independent, we have the following model instead:
![[Pasted image 20241026154845.png|175]]

Now if we have a hierarchical structure where all $\theta$ share a hyperparameter $\phi$, we get the following:
![[Pasted image 20241026155055.png|190]]

Notice that $\theta$-s are no longer independent. We say that they are **exchangeable** instead.
- $Y_1, Y_2, \cdots, Y_n$ are said to be exchangeable if the distribution of $(Y_1, Y_2, \cdots, Y_n)$ is equal to the distribution of ($Y_{\pi1}, Y_{\pi2}, \cdots, Y_{\pi n}$), where $(\pi_1, \cdots, \pi_n)$ is any permutation of $(1, 2, \cdots, n)$.
- Think about the joint distribution of the vector $(Y_1, Y_2, \cdots, Y_n)$. Permuting this vector does not affect the joint distribution, since multiplication is commutative -- this is exchangeability.
- Exchangeability does _not_ imply independence. A simple example is provided as follows:
	- Suppose we have a two-dimensional variable $(X,Y) \sim MVN_2 \left(0, \begin{pmatrix} 1 \quad \rho \\ \rho \quad 1\end{pmatrix}\right)$, where $\rho \in (-1, 1)$.
	- Then, it is easy to see that the distribution of $(Y, X)$ is the same as the distribution of $(X, Y)$. $X$ and $Y$ are also correlated (not independent) because of the factor $\rho$.
### Priors with Structural Information

Assume that we have the following:
$$
\begin{aligned}
X|p &\sim Bin(n, p), \\ 
p|k &\sim Beta(k, k), \, k\in\mathbb{N}\quad \\
k|r &\sim Geom(r),\quad p(k=i) = (1-r)^{i-1}r, \, i=1,2,\cdots \,\, 0<r<1 \\
r &\sim Beta(2,2)
\end{aligned}
$$
Suppose that we know from prior information that $p$ is around $0.5$. Therefore, we set the prior on $p$ to be $Beta(k,k)$, where the mean is $\frac{k}{k+k} = 0.5$.

We know that the value of $p$ should be around $0.5$, but we choose to set the parameters for the prior as some $k$ here as we wish to make no assumptions about the prior. If we had set $k$ to be $1$, then we get a very flat prior. Alternatively, if we had set $k$ to be a large number (say, $20$), then the distribution will be concentrated about $0.5$.

In that case, we could say that the prior on $p$, $p|k \sim Beta(k, k)$ is the **structural** part -- we know that $p$ should be around $0.5$, after all.

Then, the priors following that would be **subjective** priors. These might be informative or non-informative. 

Let us look at how these priors play out.
$$
\begin{aligned}
\[p|k] \times [k \mid r] \times [r] &\propto \frac{1}{B(k, k)} p^{k-1}(1 - p)^{k-1}(1 - r) \frac{1}{B(2, 2)} r (1 - r), \\ \\
[p \mid k] \times [k] &\propto \frac{B(3, k + 1)}{B(k, k) B(2, 2)} p^{k-1}(1 - p)^{k-1} \\ \\
[p] &\propto \sum_{k=1}^{\infty} \frac{B(3, k + 1)}{B(k, k) B(2, 2)} p^{k-1} (1 - p)^{k-1} \\ \\
&= \frac{2p^4 (4a - 15) - 4p^3 (4a - 15) + 2p^2 (11a - 25) - 2p (7a - 10) + (3a - 3)}{20p^4 (1 - p)^4}, \\
&, p \in (0, 1) \quad a = \sqrt{(2p - 1)^2}
\end{aligned}
$$
Notice that we get $a = \sqrt{(2p-1)^2}$. This is the density of the single prior, which is equivalent to the hierarchy that we set. However, a hierarchical prior allows us to model the structural, non-informative, etc. components, while a single prior does not.

A concrete example might help. Suppose we have $X|p \sim Bin(n, p)$ and a single prior $p \sim \pi(p)$. 

We have already established that the prior mean for $p$ is $0.5$. Assume $X = 3$ and $n=5$. Then, we have observed $\hat{p} = 0.6$. We find the posterior mean as follows:
$$
\frac{80\log2-55}{56\log2-38} = 0.553481
$$
Now consider the hierarchical priors. We will simply use a PPL here, and we end up with the following values: $k=3.713$, $p=0.5536$, and $r=0.474$. The posterior mean of $p$ is the same as what we calculated analytically with a single prior. However, we managed to obtain more data about $k$ and $r$. 
### Priors as Hidden Mixtures

This was mostly useful when computational power was not as great as it was today. However, there is still some benefit to using hierarchical models in terms of distribution mixtures. Let's first take a look at what is meant by "mixtures".

There is a statistical theory that states:
> Any unimodal distribution is a scale (precision) mixture of normals. 

Suppose we take the t-distribution. The t-distribution is unimodal, and thus can be represented as a mixture of normals. More concretely, the t-distribution can be represented as $\sim t(\mu_0, \tau, df)$.

This is equivalent to if we set $\mu \sim N(\mu_0, \text{prec})$, and $\text{prec} \sim Gamma(a, b)$. 
- $a=df/2$ and $b=df/2^\tau$
- No worries about deriving this, this is pulled from some statistical theorem. 

There is _another_ statistical theory that states:
> Any symmetric unimodal distribution is a scale mixture of uniforms

Consider this equivalence:

$$
y|\mu, \delta^2 \sim N(\mu, \delta^2) \Leftrightarrow \left\{ \begin{array}{l} y|\mu,\delta^2 \sim U(\mu - \sqrt{\delta^2d}, \,\,\mu+\sqrt{\delta^2d}) \\ d \sim Ga(\frac{3}{2}, \frac{1}{2}) \end{array} \right.
$$
Why is this hierarchical representation even necessary? If a pure normal distribution is used, we are only allowed to toggle the shift and scale parameter (mean and precision/variance). However, since we have broken it down into a Gamma hyperprior in the end, we can tweak the tails as well!

Let us turn our attention to:
$$
d\sim Ga(\frac{3}{2}, \frac{S}{2})
$$
For a normal distribution, $S$ is set to 1. However, if we tweak $S$ to be $<1$, we get tails heavier than normal, and tails lighter than normal if $S>1$. This allows us more control in modeling.
#### Example: Jeremy's IQ

Suppose as before, $X\sim N(\theta, \delta^2)$. The prior is $\theta \sim N(\mu,\tau^2)$
We also have that $\delta^2 = 80, \tau^2 = 120, \mu=110, X=98$. In previous units, we have elicited $\hat{\theta} = 102.8$.

Now suppose that instead of putting a single normal prior on $\theta$, we instead use the hierarchical model with uniform prior and gamma hyperprior as discussed above. Again, we will use a PPL for obtaining these values:
- When $s=1$, $\hat{\theta} = 102.8$, as this is essentially the normal distribution.
- When $s < 1$, $\theta \sim$ **heavy tailed**. If $s = 0.5$, $\hat{\theta} = 101.0$
- When $s > 1$, $\theta \sim$ **light tailed**. If $s = 2$, $\hat{\theta} = 104.9$ 
### Meta-Analysis 


## Bayesian Linear Models

### One-Way ANOVA

#### Recap

Firstly, a recap of one-way ANOVA. Quite simply, this statistical test aims to determine if there is any difference in means from three or more unrelated samples/groups.

The assumptions inherent in that aim are therefore:
- Independence,
- Normality, and
- Shared variance, $\sigma^2$.

The null hypothesis $H_0$ is that $\mu_1 = \mu_2 = \cdots = \mu_a$. The alternative hypothesis therefore is that the means are not equal. This violation could be _any_ violation (i.e., it is sufficient for just one sample/group to be different).

The model can be described as $y_{ij} \sim N(\mu_i, \sigma^2)$. 
- $i$ here represents the group; that is, which group/sample is being considered?
- $j$ represents the value observed within the $i$-th group. 
- Therefore, $\mu_i$ here refers to the mean of group $i$.
- We can think of $\mu_i$ as being split into two components: $\mu$ that is shared between all groups, and some component $\alpha_i$ (also known as "effect"). Think of this as the groups having individual deviations $\alpha_i$ from the shared mean $\mu$. 

Then, we can instead shift our attention to $\alpha_i$. It is possible to reframe the null hypothesis in this manner, such that we get: $H_0 : \alpha_1 = \alpha_2 = \cdots = \alpha_a = 0$.

We will encounter this later, but we will need to balance $\mu$ and $\alpha_i$. In that case, some initial conditions need to be set. We can do this one of two common ways:
- Set $\sum_{i=1}^a \alpha_i = 0$. This is known as the Sum-To-Zero (STZ) constraint.
- Set $\alpha_1 = 0$. This is known as the Corner (CR) constraint.
#### Classical Statistics

In classical statistics, an ANOVA table is used to test $H_0$. If $H_0$ is accepted, there is no evidence of differing means and we stop. Otherwise, if $H_0$ is rejected, multiple pairwise comparisons need to be performed to tease out the differing groups.

#### Bayesian Statistics

In Bayesian statistics, we instead treat $\mu$ (referred to as the **grand mean**), all $\alpha_i$, and $\sigma^2$ to be random, given noninformative priors. In essence, we treat them all as parameters. If we _do_ have prior information, then we could use informative priors -- nothing wrong there.

Since $\alpha$'s are random variables, any function of $\alpha$;s can be estimated and tested. This is not true in classical statistics, where only linear combinations of $\alpha$'s can be handled.

Also, we will see later that the Bayesian model allows us richer information:
- Correlations among parameters,
- Easy multiple comparisons, 
- Ranking of parameters, and so on.

### Two-Way ANOVA (Factorial Designs)

The two-way ANOVA is the next generalisation of one-way ANOVA. In this case, we have two factors, $i$ and $j$ contributing to the "response" (if we think of this as treatment vs control).

Think of this as a 2D table:

|       | 1   | 2   | ... | $n_j$ |
| ----- | --- | --- | --- | ----- |
| 1     |     |     |     |       |
| 2     |     |     |     |       |
| ...   |     |     |     |       |
| $n_i$ |     |     |     |       |

Each cell $i,j$ corresponds to some observation. Therefore, if we have multiple groups as we did in one-way ANOVA, we would have multiple observations at the $i,j$-th cell. Let us denote this $y_{ijk}$ -- the observation $k$ at cell $i,j$.

Then, we have that:
$$ y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk} $$
Let's break this down.
- $\mu$ is the grand mean for all observations. This is the same concept as encountered in one-way ANOVA.
- $\alpha_i$ and $\beta_j$ are effects of the factor $i$ and $j$ respectively.
- The interaction between the factors are captured by $(\alpha\beta)_{ij}$.
- And finally some random error $\epsilon$. We assume this to be normally distributed -- this is the same across all observations.

Therefore,
$$
\begin{aligned}
\mu_{ij} &= \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} \\ 
\epsilon_{ijk} &\sim N(0, \sigma^2)
\end{aligned}
$$
#### Constraints

Let us set constraints again. We can either do Sum-To-Zero (STZ):
$$ \sum \alpha_i = 0, \, \sum \beta_j = 0, \quad \sum_i (\alpha\beta)_{ij} = \sum_j(\alpha\beta)_{ij} = 0 $$
Or we could do Corner (CR):
$$ \alpha_1 = 0, \, \beta_1 = 0,\quad (\alpha\beta)_{1j} = (\alpha\beta)_{i1} = 0$$

#### Hypotheses

Perhaps more interestingly, let us now turn our attention to testing hypotheses. The very first thing to test in two-way or multiple-way ANOVA would be to the interactions between the factors. Therefore, let us ask if $(\alpha\beta)_{ij} = 0$ as the null hypothesis $H_{01}$. The subscripts here simply represent the first (1) null hypothesis.

If there _is_ a significant interaction between the two factors, the actual effect of the individual factors $\alpha$ and $\beta$ might be questionable. Ideally, we would want to accept the null hypothesis that there is no significant interaction between the two factors.

Then, we can move on to testing the main effects, $H_{02}: \alpha_i = 0$ and $H_{03}: \beta_i = 0$.

### Regression

Regression should need no introduction, being a classic example of a linear model. In fact, ANOVA and more complex experimental designs are at the end of the day, regressions.
#### Simple Regression

A simple regression involves data in the form $(x_1, y_1), \cdots (x_n, y_n)$. These can be thought of as points on a Cartesian plane.

The model is then given by:
$$ y_i = \beta_0 + \beta_1 \cdot x_i + \epsilon_i, \quad\quad i = 1, \cdots, n, \quad \quad\epsilon\stackrel{iid}{\sim} N(0, \sigma^2)$$Notice that the model takes on the form $y=c + mx + \epsilon$, which should be rather familiar.

Either way, classical statistics is rather elegant here. Let us first delineate the sums that are relevant in computing the model parameters:
$$
\begin{aligned}
SYY &= \sum_{i=1}^n (y_i - \bar{y})^2 \\
SXY &= \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) \\
SXX &= \sum_{i=1}^n (x_i - \bar{x})^2 \\ \\
SSE &= \sum_{i=1}^n (y_i - \hat{y}_i)^2 \\
SSR &= \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 \\
\end{aligned}
$$

Then, we can calculate the slope $\beta_1$ and the intercept $\beta_0$ via the following:
$$
\hat{\beta}_1 = \frac{SXY}{SXX}, \quad\quad  \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \cdot \bar{x}
$$
The difference between the predicted value $\hat{y}_i$ and the observed value $y_i$ is referred to as the **residual**, $e_i$. This is often used in classical statistics and Bayesian statistics to measure goodness-of-fit.

It is also typical to calculate $R^2$, whose definition is given below:
$$ R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST} $$
- $SSR$ is the amount of variability predicted by the $\hat{y}$-s, and $SST$ is the total variability.
- In a way, $R^2$ is a ratio of the variability explained by the regression over the total variability. Of course, if you explain more ($R^2$ closer to 1), your regression is better.
 
#### Multiple Regression

In the case of multiple regression, we have data in the form $(x_{i1}, x_{i2}, \cdots, x_{ik}, y_i)$, where $i = 1, \cdots, n$. We may think of it this way: we take $n$ measurements and each measurement $i$ has $k$ different $x$s. 

The model is given by:
$$ y_i = \beta_0 + \beta_1 \cdot x_{i1} + \beta_2 \cdot x_{i2}+ \cdots + \beta_k \cdot x_{ik} + \epsilon_i $$
$$ \epsilon\stackrel{iid}{\sim} N(0, \sigma^2), \quad \quad k+1 =p $$
The response $y_i$ is simply a linear combination of all $x$-s. Again, $\beta_0$ is the intercept and the error is normal with mean $0$ and variance $\sigma^2$.

Note that we have $k$ different predictors ($\beta_1$ to $\beta_k$). However, the total number of parameters $p$ is $k+1$, since we have to include the parameter $\beta_0$ as well.

The above model can be written (very) compactly as:
$$ y = X\beta + \epsilon $$
Here, we have:
$$
y = \begin{bmatrix} 
y_1 \\ 
\vdots \\
y_n 
\end{bmatrix}_{n \times1} \quad
X = \begin{bmatrix}
1 \,\, X_{11} \,\, X_{12} \,\, \cdots \,\, X_{1k} \\
1 \,\, X_{21} \,\, X_{22} \,\, \cdots \,\, X_{2k} \\
\vdots \\
1 \,\, X_{n1} \,\, X_{n2} \,\, \cdots \,\, X_{nk} \\
\end{bmatrix}_{n\times p} \quad
\beta = \begin{bmatrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_k
\end{bmatrix}_{(k+1)\times1} \quad
\epsilon = \begin{bmatrix}
\epsilon_1 \\
\vdots \\
\epsilon_n \\
\end{bmatrix}_{n \times1}
$$

##### Least Squares Estimation

The very elegant least squares estimator $\hat{\beta}$ is given by
$$ \hat{\beta} = (X^TX)^{-1} X^Ty $$
Once we get the estimator $\hat{\beta}$, we can simply multiply it by the design matrix $X$ to get predictions of $\hat{y}$. That is, $\hat{y} = X\hat{\beta}$.

Alternatively, if we write $(X^TX)^{-1} X^T$ as $H$, we get $\hat{y}= Hy$. $H$ is referred to as the "hat matrix".

The residual $e$ is simply $(I - H)y$, where $I$ is the identity matrix.
##### $R^2$

Similar to the simple regression case, we calculate $R^2$ by taking $1 - \frac{SSE}{SST}$. Here:
$$
SST = y^T(I - \frac{1}{n}J)y, \quad\quad SSE = y^T(I-H)y
$$
- $I$ is the identity matrix and $J$ is a matrix of ones.
- Notice that $SST = SSR + SSE$.

Alternatively, we may use an adjusted $R^2$ where we penalise by number of parameters:
$$ R^2_{adj} = 1 - \frac{(n-1)SSE}{(n-p)SST}$$
##### Bayesian Model

The Bayesian model is as one would expect. Rather simply, we put priors on all parameters. The posteriors define the best fit. 
###### Typical Prior
So how would go about putting priors on all $\beta_i$? A common way to do this is to assume that all parameters are independent, and that they are normally distributed.
$$ \beta_j \sim N(0, \xi), \quad\quad \xi=\frac{1}{\text{Var}(\beta_j)},\, j=0,\cdots,k  $$
- $\xi$ is the precision parameter (reciprocal of variance) and is typically assigned a very small value (say $10^{-5}$)
- This is a non-informative prior, hence we try to allow a larger variance to "catch" more possible values.

The likelihood, on the other hand requires a precision parameter of the observation $\tau$. We typically assume
$$ \tau \sim Ga(a, b)$$
- $a,b$ are typically equal and small (say, $10^{-3}$)

###### Normal-Inverse Gamma Prior
There are other methods for defining priors such as the Normal-Inverse Gamma, which may help to account for covariance between the predictors. In other words, $\beta$ are not independent on the precision/variance.

The conditional $\beta |\sigma^2$ are given a multivariate normal distribution, $\sim MVN(\mu, \sigma^2V)$. $V$ here refers to the covariance matrix. Usually, $\mu$ is assigned to the value 0 but it is possible to have a non-zero value.

Note that the variance is modeled after an Inverse Gamma distribution, $\sigma^2 \sim IG(a,b)$.
The precision on the other hand follows Gamma, $\tau = \frac{1}{\sigma^2} \sim Ga(a, b)$.

The estimates are:
$$ \hat{\beta} = \mathbb{E}[\beta|y] = A\beta^{OLS} + (I + A)\mu $$
- $\beta^{OLS}$ refers to the Ordinary Least Squares estimator that we have encountered above.
- If the prior mean $\mu$ is $0$, then we simply have the matrix $A$ times the OLS estimator.
- $A$ is defined as $(X^TX + V^{-1})^{-1} X^TX$. Notice that the prior covariance matrix $V$ is allowed some influence here.

The use of these priors is quite popular, and is known as Ridge regression. 
###### Zellner's Prior

Zellner's Prior involves the Normal-Inverse Gamma prior. There are slight differences, but only with regards to the variance of the MVN distribution:
$$ \beta |\sigma^2 \sim MVN(\mu, g \cdot \sigma^2V)$$
$$\sigma^2 \sim IG(a,b)$$
Here, the covariance matrix $V$ is calculated using the design matrix $X$. In certain PPLs (probabilistic programming languages), $V$ is defined as:
$$ \frac{\tau}{g}V^{-1} = \frac{\tau}{g}X^TX $$
$g$ can be custom, but is usually set as $g=n$ (sample size), $g=p^2$ (num parameters squared), or the max of $n$ and $p^2$.
## Other Models
### Generalized Linear Models

What exactly is _generalised_ here? The observations $y_i$ remain independent, but the **distribution is generalised from normal to exponential family**. We will see later what this means exactly.

Recall that in linear models, we directly equate $y_i$ to $\beta_0 + \beta_1x_{i1} + \cdots + \beta_kx_{ik} = l_i$; that is, $y_i = l_i$.
However, in GLMs, we equate the mean of $y_i$ to some **link** function, $g$ of $l_i$. More concretely, we have:
$$ \mu_i = \mathbb{E}y_i = g(l_i)  $$
For linear models, the variance of $y_i$ is constant, but this is not true for GLMs. Here, the variance depends on $\mu_i$. 
#### Exponential Family

Let us first take a look at the general form of distributions in the exponential family.
$$
f(y|\theta, \phi) = \exp\{\frac{y\theta-b(\theta)}{\phi} + c(y, \phi)\}
$$
- $\theta$ is called the natural (or canonical) parameter.
- $b$ is a function that links the mean of $y$ and the variance of $y$.
- $\phi$ is called the dispersion parameter and is linked with the variance.
- $c$ is a function that is free of the parameter $\theta$.

The normal distribution is part of the exponential family, where $\theta = \mu$. We may represent the density of a normal distribution as follows:
$$
f(x|\mu, \delta^2) = \exp\{\frac{y\mu - \mu^2/2}{\delta^2} - \frac{1}{2}\left[\frac{y^2}{\delta^2}+\log(2\pi\delta^2)\right]\}
$$
- The canonical parameter $\theta = \mu$. 
- The dispersion parameter is denoted here as $\delta^2$, which is essentially the variance of a normal distribution.

Another example is the Bernoulli distribution. 
- The canonical parameter $\theta = \text{logit}(p) = \log\frac{p}{1-p}$

Finally, the Poisson distribution is also part of the exponential family.
- The canonical parameter $\theta = \log(\lambda)$.

The derivations for Bernoulli and Poisson distributions are left as exercises.  
#### More on Links

A typical link for binary (binomial) observations involves the logit link. 
$$
\log\frac{p}{1-p} = F^{-1}(p) = \beta_0 + \beta_1x_1 + \cdots + \beta_kx_k
$$
- Here, $F$ is the logistic CDF. So why are we taking the inverse of $F$?
- Let us consider the case where $l$ (the linear combination of predictors) is infinite.
- The logistic CDF also has x-values that span both infinities, with y-values that are bounded by 0 and 1. 
- By taking the inverse mapping, we are "constraining" the value of $p$ to be between 0 and 1 -- exactly right for a probability.

This $F$ could really be any distribution that is continuous, and is supported by $[-\infty, +\infty]$. As it turns out, the normal CDF also works. This is known as the **probit** link. Similarly, we have:
$$
F^{-1}(p) = \beta_0 + \beta_1x_1 + \cdots + \beta_kx_k
$$
- $F$ here is the normal CDF.

Alternatively, the complementary log-log CDF is used quite often ($F^{-1}$ of cloglog). $F$ here is the CDF of a Gumbel Type I distribution, and has the form $F(x) = 1 - \exp(-\exp(x))$. Therefore, 
$$
F^{-1}(p) = \text{cloglog}(p) = \log(-\log(1-p)) = \beta_0 + \beta_1x_1 + \cdots + \beta_kx_k
$$

A binary regression with observations $y=0$ or $y=1$ could be performed with any of these links (logit, probit, cloglog), or even any general link. But what exactly are we estimating here?

We are estimating the value of $p$, the probability that the observation is equal to $1$. This makes GLMs very popular for binary classification problems, where we are interested in the probability of observing class $1$ given a bunch of covariates.

For Poisson regression, the link if simply $\log$.
$$
\log(\lambda) = \beta_0 + \beta_1x_1 + \cdots + \beta_kx_k
$$
- Again, why is this? Recall that the right-hand side (denoted $l$) ranges from $[-\infty, +\infty]$.
- The exponential function on the other hand is allowed $[-\infty, +\infty]$ and ranges from $[0, +\infty]$. Therefore, we map the final value to $[0, +\infty]$.
#### Measuring Performance

In linear models, we had the $R^2$ value that explains how much of the variance is explained by the model compared to the actual data.

For GLMs, we use **deviances** as standard measures of model performance instead. In statistics, we either have an unnormalised deviance:
$$
D = -2\log(\text{likelihood of the fitted model})
$$
Or a normalised deviance:
$$
D = -2\log(\frac{\text{likelihood of the fitted model}}{\text{likelihood of the saturated model}})
$$
- The saturated model is the "best possible model", where we model the observations $y_i$ by themselves. 
- Of course, the smaller the deviance, the better the model performs.
##### Logistic
For a logistic regression, we get the deviance $D$ to be:
$$
D= -2\sum_{i=1}^ky_i\log\frac{\hat{y}_i}{y_i} + (n_i - y_i)\log\frac{n_i-\hat{y}_i}{n_i-y_i}
$$
- Where $\hat{y}_i$ = $n_ip_i$ is the model fit for $y_i$.
- For the saturated model, $p_i = \frac{y_i}{n_i}$, $\hat{y}_i = y_i$

##### Poisson
For a Poisson regression,
$$
D= -2\sum_{i=1}^ky_i\log\frac{y_i}{\hat{y}_i} + (y_i - \hat{y}_i)$$
- Where $\hat{y}_i = \exp\{\beta_0 + \beta_1x_{i1} +\cdots+\beta_kx_{ik}\}$
### Multinomial Regression 

In multinomial regression, we have more than two categories. Recall that in binary or logistic regression, we have two categories (0 or 1) as a function of covariates linked by logit, probit, or cloglog.

We will look at multinomial logit here, which is a generalisation of logit. More formally, we have $n$ observations ($y$).
$$ y_1, y_2, \cdots, y_n \sim Mn(p, 1)$$
Suppose that we have $K$ categories. Then, we will need to predict the probabilities $\textbf{p}$ of each category, where $\textbf{p} = (p_1, p_2, \cdots, p_K)$.

Each observation $y_i$ will be a tuple of a single one and zeroes:
$$y_i = (y_{i1}, y_{i2}, \cdots, y_{iK}), \quad y_{ij} = 1, y_{i\neq j=0}, \quad j \in \{1, \cdots, K\}$$
For example, if $y_i$ is of category 4, we denote $y_i = (0, 0, 0, 1, 0)$.
- Here, $K=5$, $y_{i4} = 1$, $y_{i\neq 4} = 0$.
- Then, $\textbf{p} = (p_1, p_2, \cdots, p_K)$ is of interest.

So how does prediction occur? Suppose we already have our data, and we have the $i$-th subject incoming with covariates $x_{i1}, \cdots, x_{i, p-1}$. Note that $p$ here refers to the number of parameters and not the probability (the notation here is potentially confusing).

Calculating the probability $p_{ij}$ is actually rather straightforward. First, we calculate the linear combination for subject $i$ favouring the category $j$ as $\eta_{ij}$.
$$
\eta_{ij} = \beta_{0j} + \beta_{1j}x_{i1} + \cdots + \beta_{p-1}x_{i, p-1}
$$
Then, we simply find calculate the proportion against the sum of probabilities for all categories:
$$
p_{ij} = \frac{\exp\{\eta_{ij}\}}{\sum_{k=1}^K\exp\{\eta_{ik}\}}
$$

We can also pick one category to be a **reference category**. Suppose we pick the first category, $j=1$. Then:
$$
p_{i1} = P(y_{i1} = 1) = \frac{1}{1 + \sum_{k=2}^K\exp\{\eta_{ik}\}}
$$
Since all $\beta$ parameters are equal to zero, the numerator turns out to be $e^0 = 1$. Then, the other categories will have the probability:
$$
p_{ij} = P(y_{ij} = 1) = \frac{\exp\{\eta_{ij}\}}{1 + \sum_{k=2}^K\exp\{\eta_{ik}\}}
$$
- Note the denominator and how it is different from the general formula for $p_{ij}$ given above.

We consider the use of a reference category as it allows us to compare the probabilities of all other categories relative to the reference. In fact, the log of this ratio is exactly equal to the linear combination of predictors for the particular subject:
$$
\log\frac{p_{ij}}{p_{i1}} = \eta_{ij} = \beta_{0j} + \beta_{1j}x_{i1} + \cdots + \beta_{p-1}x_{i, p-1}
$$
### Multilevel Models

As per its namesake, this involves modeling at different levels for its subjects. For example, a subject may be a part of a larger community, which in turn may be a part of a district, then a country, and so on. Of course, the highest resolution occurs at the lowest level (at the subject level).



