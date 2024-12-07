## Model Fit

### Deviance

We have briefly covered the concept of deviance before. Briefly, a model with smaller deviance is better, as it manages to capture more of the provided data. Unfortunately, testing for statistical significance with deviances is rather tough as there is no predefined cutoff to use. However, it is still possible to calibrate (compare) deviances of different models.

There are two common definitions of deviance. For a model with likelihood $f(y|\theta)$, the deviance is given by:
$$
D(\theta) = -2\log f(y|\theta)
$$
Alternatively, we can compute the **saturated** deviance (denoted by subscript 's') as:
$$
D_s(\theta) = -2(\log f(y|\theta) -\log f(y|\theta_s))
$$
> The saturated model $f(y|\theta_s)$ can be thought of as the "optimal" model, in which data are modeled exactly.
#### Deviances of Common Models

We have already encountered these before, but here they are anyway for convenience:
$$
\begin{aligned}
y_i \sim Bin(n_i, \theta_i) &: \quad\quad D_s(\hat\theta) = 2 \sum_i \big(y_i \log\frac{y_i/n_i}{\theta_i} + (n_i-y_i)\log\frac{1-y_i/n_i}{1-\theta_i}\big) \\ \\
y_i \sim Poi(\theta_i) &: \quad\quad D_s(\hat\theta) = 2 \sum_i \big(y_i \log\frac{y_i}{\theta_i} + (y_i-\theta_i)\big) \\ \\
y_i \sim N(\theta_i, \delta_i^2) &: \quad\quad D_s(\hat\theta) = \sum_i \big(\frac{y_i-\theta_i}{\delta_i}\big)^2 \\ \\
\end{aligned}
$$
#### Example

Let us consider two models for the data $y=(1,1,2,2,3,4,4,5,8)$. We will consider the Weibull and Exponential model. We will simply take non-informative priors on the parameters.

Which model is better (i.e., have smaller deviance)? Using a PPL, we get that the deviance for the Weibull model is 43, while that of the Exponential model is 46. Therefore, the Weibull model is a better fit.
### Deviance Information Criterion

Information criteria often consist of goodness-of-fit as well as a penalty (to avoid overfitting). Let us first look at the Akaike Information Criterion (AIC).
$$
AIC = 2\log f(y|\hat\theta) + 2p = D(\hat{\theta}) + 2p 
$$
- $\hat\theta$ is the MLE of $\theta$, and $p$ is the number of parameters in the model.
- The model with a smaller AIC is favoured.

The Bayesian version of AIC was proposed in 2002 by Spiegelhalter, and is known as Deviance Information Criterion (DIC).
1. $\bar{D} = \mathbb{E}^{\pi(\cdot|y)}(-2\log f(y|\theta))$. This is the posterior expectation of the function $f$.
2. $D(\bar{\theta}) = -2 \log f(y|\bar{\theta})$. At the same time, we can plug in the Bayes estimator of $\theta$, $\bar{\theta}$.
3. $p_D = \bar{D} - D(\bar{\theta})$. Then, calculate the effective number of parameters.
4. DIC = $\bar{D} + p_D = D(\bar\theta) + 2p_D$. Finally, we have two definitions of DIC using either the posterior expectation, or the Bayes estimator.
## Model Selection

### Laud-Ibrahim Criterion

This is essentially a model comparison tool. Suppose that we have $M$ family of models, and $\theta^{(m)}$ are parameters under model $m \in M$.

The posterior given data and a particular model $m$ is simply:
$$
\pi(\theta^{(m)}|m, \bar{y}) = \frac{f(y|\theta^{(m)},m)\pi(\theta^{(m)})}{\int f(y|\theta^{(m)}, m)\pi(\theta^{(m)})d\theta^{(m)}}
$$
> While messy, this is simply traditional Bayes Theorem.

Recall that the posterior predictive distribution is essentially integrating the product of the likelihood and posterior:
$$
f(z|y,m) = \int f(z|\theta^{(m)}, m)\pi(\theta^{(m)}|y,m)d\theta^{(m)}
$$
> This is essentially asking "how would you predict $z$ if you have data $y$ and model $m$"

We can think of $z$ as the predicted data, $y$ as the training data, and $m$ as the selected model. Therefore, we have $Z \sim f(z|y,m)$. Now we can start sampling from this distribution and ask ourselves if there is a difference between the predicted and observed data.

Define for $m$ fixed:
$$
L_m^2 = \mathbb{E}(Z-y)^T(Z-y) = \sum_{i=1}^n (\mathbb{E}Z_i - y_i)^2 + \text{Var}(Z_i)
$$
- Let us dissect the the final equation. The summation term is essentially asking how well $Z$ from the predictive distribution fits the observed data. The better the fit, the smaller the value.
- We then penalise by the variance of the predicted data.

The model that minimizes $L_m$ is favoured.

### Outlier Testing

#### Conditional Predictive Ordinate

This is essentially trying to find the predictive distribution of the observation $y_i$ if all other observations are not given. 

We therefore have the following posterior predictive distribution
$$
(CPO)_i = f(y_i|y_{-i}) = \int f(y_i|\theta)\pi(\theta|y_{-1}) \, d\theta
$$
where $y_{-1} = y_1, \cdots, y_{i-1}, y_{i+1}, \cdots, n$

It is easy to evaluate CPO with the following formula which will not be derived here:
$$
(CPO)_i^{-1} = \frac{1}{B} \sum_{b=1}^B \frac{1}{f(y_i|\theta^b)}
$$
- Notice that this is the average of the reciprocal of the likelihood. $\theta^b$ here is an MC realisation from the posterior (i.e., sampled via runs of MCMC). 

##### Diagnostics

If we observe a $(CPO)_i$ value of $< 0.02$, this means that there is a potential outlier. In other words, the observation $y_i$ is rather unlike the rest of the other observations.

However if we observe a $(CPO)_i$ is $<0.01$, this likely means that we are dealing with an outlier w.r.t. the model $f$. 

A common measure of predictive quality of a model using $(CPO)_i$ is provided as follows:
$$
-2 \sum_{i=1}^n \log(CPO)_i
$$
- This is similar in structure to what we have encountered in deviance.
- The smaller this value, the better the model's performance.

Note that $(CPO)_i$ depends on the likelihood, and a reasonable argument against using $(CPO)_i$ is that models with different likelihoods are not comparable. Therefore, it is also possible to compute the following:
$$
C_i = \frac{(CPO)_i}{f(y_i|\theta)}
$$
- Values of $C_i$ differing from 1 indicate potential outliers.

### Cumulative

Most PPLs should have this ability to find $F(y_i)$, where $F$ is the model CDF and $y_i$ are the observations. This is simply re-applying the model back on the observed data.

If $F$ is indeed the correct distribution for $y_i$, then $F(y_i) \sim Uniform(0,1)$. This is a basic theorem that is sometimes used to generate random numbers. 

Values $F(y_i) < \frac{c}{n}$ and $F(y_i) > 1- \frac{c}{n}$ for small $c$ are potential outliers. $c$ can be manually set, but taking $c=\alpha$ (confidence level) is approximately equivalent to the level of classical test for outliers. 

### SSVS

SSVS stands for Stochastic Search Variable Selection. Consider:
$$
\begin{aligned}
\mu &= \beta_0 + \beta_1x_1 + \cdots + \beta_kx_k \\ 
\beta_i &= \delta_i\alpha_i \\ 
\alpha_i &\sim N(0, \tau) \\ 
\delta_i &\sim Bern(p_i)
\end{aligned}
$$
Here, we define the coefficients $\beta_i$ as a product of a coefficient $\alpha_i$ (drawn from some non-informative prior) and an **indicator** $\delta_i$, drawn from a Bernoulli distribution (i.e., values 0 or 1). $p_i$ here is the prior probability that variable $x_i$ is in the model. 

If the indicator has a value of 0, then the predictor coefficient $\beta_i$ will be 0 and the variable $x_i$ is not in the model. A posteriori, we will see how many times the Markov chain visited models, and we will simply choose models that are visited the most. 