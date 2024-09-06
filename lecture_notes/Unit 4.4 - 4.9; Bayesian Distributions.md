## Bayesian Inference

### Components

#### Initial Model and Joint Distribution

We start with the **model**. Observations $X_1, \cdots, X_n$ from the experiment are modelled by the statistical model, denoted as: $$X_i \sim \stackrel{iid}f(x_i|\theta) , \quad i = 1, \cdots, n $$The $iid$ here indicates that $X_i$ are independently distributed. $\theta$ here is the parameter.

The joint distribution of the sample $X_1, \cdots, X_n$ is simply the product of all individual components:
$$f(x_1|\theta) \times f(x_2|\theta) \times \cdots \times f(x_n | \theta) = \prod_{i=1}^nf(x_i|\theta)$$
#### Likelihood

Notice that the $x$ -es are observations, which might not have any value _before_ the experiment. However, once the experiment is done, we have our observations that we can plug into the expression above. More formally, as a function of $\theta$, the joint distribution $\prod_{i=1}^nf(x_i|\theta)$ is called the **likelihood**.
$$ L(\theta|x_1, \cdots, x_n) = \prod_{i=1}^nf(x_i|\theta)$$
According to the **Likelihood Principle**, all information about the experiment is contained within the likelihood -- the model, parameters, observations, and so on are all included.

As an example, consider the following:

Each $X_i$ is sampled from an exponential distribution, $Exp(\lambda)$. Let $X_1 = 2$, $X_2 = 3$, and $X_3 = 1$ to be the observations. Then the likelihood is:
$$ L(\lambda | x_1, x_2, x_3) = \lambda e^{-2\lambda}\times\lambda e^{-3\lambda} \times \lambda e^{-\lambda} = \lambda^3e^{-6\lambda}$$
If the data are kept unspecified:
$$ \large{L(\lambda|x_1, x_2, x_3) = \lambda^3e^{-\lambda\sum_{i=1}^3x_i}}$$
#### Parameters and Priors

Let $\theta$ be a parameter in $f(x|\theta)$. 

Recall that the Classical statistician views the parameter as some fixed, unknown value. The inference in that case is concerned with this unknown number. However, the Bayesian statistician views the parameter as a **distribution**.
- This distribution represents the uncertainty of this parameter. The distribution can be very "certain" -- concentrated at a point mass at some value. The distribution might have a very large variance if we are less certain about this parameter.

More formally, **a prior is a distribution on** $\theta$.

$$ 
\begin{aligned}
\theta \sim \pi(\theta), \quad &\theta \in \Theta \\
&\Theta \equiv \text{parameter space}
\end{aligned}$$
Now we may think of $X$, which are our measurements, and $\theta$ as a vector of two random variables. Therefore, we can also find the joint distribution of $(X, \theta)$, which is $h(x, \theta)$.

The marginal distribution of $X$ then is:
$$ m(x) = \int_\Theta h(x,\theta) \, d\theta$$
### Bayes' Theorem

Recall Bayes' Rule for events:

$$P(AH_i) = P(A|H_i)P(H_i)= P(H_i|A)P(A) $$
$$ \Rightarrow P(H_i|A) = \frac{P(A|H_i)P(H_i)}{P(A)}$$

For distributions, we use **Bayes' Theorem** instead, which we can derive by analogy:
$$h(x, \theta) = f(x|\theta)\pi(\theta) = \pi(\theta|x)m(x)$$
The notations are as follows:
- $h(x,\theta)$ is the joint distribution
- $f(x|\theta)$ is the likelihood
- $\pi(\theta)$ is the prior
- $\pi(\theta|x)$ is the posterior
- $m(x)$ is the marginal

In other words, the joint distribution of $x$ and $\theta$ can be represented as:
- The product of the likelihood of the model and the prior, or
- The product of the posterior and the marginal

Rearranging the rightmost two terms, we get Bayes' (or Bayes) Theorem:
$$ \pi(\theta|x) = \frac{f(x|\theta)\pi(\theta)}{m(x)}$$
## Conjugate Families

### Introduction

#### Normal Likelihood, Normal Prior

Let's consider an example involving a normally distributed likelihood and a normally distributed prior.

Suppose we have an observation $x$ given $\theta$ which follows a normal distribution. Let $x|\theta \sim N(\theta, \sigma^2)$ and assume that $\sigma^2$ is known (this is the likelihood). The parameter of interest here is $\theta$, which represents the shift of this distribution.

Now we know that the prior on $\theta$ is normally distributed. Let $\theta \sim N(\mu, \tau^2)$. Here, $\mu$ and $\tau$ are elicited. 
- $\mu$ and $\tau$ are also called **hyperparameters**, as they are the parameters of the distribution of the parameter.

Let us now determine  the joint distribution $h(x|\theta)$. From the definitions given above, we will take the product of the likelihood and the prior.
$$ \large{
h(x|\theta) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{1}{2\sigma^2}(x-\theta)^2} \times \frac{1}{\sqrt{2\pi\tau^2}}e^{-\frac{1}{2\tau^2}(\theta-\mu)^2}
}$$
To make life (much) easier for us, let us simply consider the exponent portions. These will be denoted with $\text{exp}\{\}$.
$$\begin{aligned}
&\text{exp}\left\{-\frac{1}{2\sigma^2}(x-\theta)^2 - \frac{1}{2\tau^2}(\tau-\mu)^2\right\} \equiv \\ \\
&\text{exp}\left\{-\frac{\sigma^2+\tau^2}{2\sigma^2\tau^2}\left(\theta - \left(\frac{\tau^2}{\sigma^2+\tau^2}x + \frac{\sigma^2}{\sigma^2+\tau^2}\mu\right)\right)^2 - \frac{1}{2(\sigma^2+\tau^2)}(x-\mu)^2\right\}
\end{aligned}$$

Notice that the exponent above splits into two parts, one containing $\theta$ and the other $\theta$-free. 
- Recall from above that the joint distribution can be represented either as $f(x|\theta)\pi(\theta)$ or $\pi(\theta|x)m(x)$.
- Since we started with $f(x|\theta)\pi(\theta)$, the exponent above corresponds to $\pi(\theta|x)m(x)$.

Quite simply then, the marginal distribution, $m(x)$, resolves to $X \sim N(\mu, \sigma^2+ \tau^2)$. 

The posterior distribution, $\pi(\theta|x)$, on the other hand resolves to:
$$ \theta | X \sim N\left(\frac{\tau^2}{\sigma^2 + \tau^2}x + \frac{\sigma^2}{\sigma^2+\tau^2}\mu, \, \frac{\sigma^2\tau^2}{\sigma^2+\tau^2}\right) $$

A few things to note about the above posterior distribution:
- The mean is composed of "what is observed" ($x$) and "what is believed ($\mu$)".
- The weights of the observation and belief sum up to 1. In other words, the mean is a **compromise** between the observation and the belief.
- The weights depend solely on the variance of both the observation and the belief. In other words, "how sure are you about $x$?" and "how sure are you about $\mu$?"

Let us rewrite the posterior mean, using $w$ to represent the weights of the observed $x$ and the elicited $\mu$. Note that since it is given that we are dealing with normal distributions, the mean = mode = median.
$$ \frac{\tau^2}{\sigma^2 + \tau^2}x + \frac{\sigma^2}{\sigma^2+\tau^2}\mu = w\cdot x + (1-w)\cdot \mu$$
If:
- $\tau^2 \gg \sigma^2 \rightarrow w \approx 1$, the posterior mean is closer to $x$. Alternatively, since $\sigma^2$ is small, we are more certain about the observations than the prior belief.
- $\sigma^2 \gg \tau^2 \rightarrow w \approx 1$, the posterior mean is closer to $\mu$. Alternatively, since $\tau^2$ is small, we are more certain about the prior belief than the observations from the experiment.

In a very natural way, most of the Bayes rules that we are going to discuss take on the form of a **compromise** between what is observed and what is believed.

#### Defining Conjugate Families

In the last example, we had a Normal likelihood and a Normal prior. Finding the the Posterior was relatively straightforward; the posterior turned out to be Normal as well. This is also called a **conjugate family**. 

More formally, if for a likelihood $f$ and prior $\pi$ the **prior and posterior belong to the same family of distributions**, then the pair $(f, \pi)$ is conjugate.

Now if the pair $(f, \pi$) is conjugate, there is no need to calculate the normalising constant, $m(x)$, in $f(x|\theta)\pi(\theta) \propto \pi(\theta|x)$. 
- Recall from Bayes' Theorem that $\pi(\theta|x) = \frac{f(x|\theta)\pi(\theta)}{m(x)}$.
- Basically, if $(f, \pi)$ is conjugate, there is no need to calculate $m(x)$, which is the marginal distribution of $x$!
- Recall also that $m(x)$ is calculated by integrating the joint distribution of $x$ and $\theta$, $h(x, \theta)$, with respect to the parameter, $\theta$. This integration is usually the "troublemaker" in Bayesian statistics, as it is often not feasible.
- Simply put, if we can conclude the form of the posterior from the product of the likelihood and the prior, $f(x|\theta)\pi(\theta)$, there is no need to perform integration.
#### Binomial Likelihood, Beta Prior

Let us now take a look at another example, this time involving a Binomial likelihood and a Beta prior. 

Suppose we do not know the probability of success, $p$, but we have conducted $n$ experiments and obtained the number of successes, $x$. We model this number of successes with the Binomial distribution:
$$ X|p \sim Bin(n, p); \quad f(x|p) = {n\choose x}p^x(1-p)^{n-x}$$
We model the prior on the probability of success, $p$:
$$ p \sim Be(\alpha, \beta); \quad \pi(p) = \frac{1}{B(\alpha, \beta)}p^{\alpha - 1}(1-p)^{\beta -1} $$
First, understand that the fraction $\frac{1}{B(\alpha, \beta)}$ is a normalising constant. 
- More concretely, the term $B(\alpha, \beta)$ is known as the Beta function, which is equal to $\int_0^1 p^{\alpha-1}(1-p)^{\beta-1} dp$.
- This is a fixed integral, which will integrate into some constant, which in turn normalises the terms $p^{\alpha-1}(1-p)^{\beta-1}$ to 1.

Next, notice that the two distributions are somewhat similar. There is (i) some constant, (ii) $p$ raised to some power, and (iii) $1-p$ raised to some power. (ii) and (iii) together are known as the **kernel** of the distribution. 

Therefore, when calculating the product of the likelihood and the prior, we get:
$$
\begin{aligned}
\pi(p|x) \propto f(x|p)\pi(p) &= C \times p^{x+\alpha-1}(1-p)^{n-x+\beta-1}\\ \\
&\Rightarrow p|X \sim Be(x+\alpha, n-x+\beta)
\end{aligned}
$$
Now what does this tell us?
- Prior to the experiment, we have believe that $p$ followed a Beta distribution with parameters $\alpha$ and $\beta$ $\Rightarrow p \sim Be(\alpha, \beta)$.
- After the experiment, we discover that the $p$ also follows a Beta distribution, this time with updated parameters $\Rightarrow Be(x+\alpha, n-x+\beta)$
- Given a Beta distribution, the mean is calculated simply following this formula: $E(X) = \frac{\alpha}{\alpha + \beta}$.
- Therefore, the mean of our posterior is easily calculated as $E(p|X) = \frac{x+\alpha}{n+\alpha+\beta}$.

Let's focus a bit more on the updated mean. We can represent this as as a weighted average:
$$
\begin{aligned}
E(p|X) &= \frac{x+\alpha}{n+\alpha+\beta} \\ \\
&= \frac{x}{n}\times\frac{n}{n+\alpha+\beta} + \frac{\alpha}{\alpha+\beta}\times\frac{\alpha+\beta}{n+\alpha+\beta} \\ \\
&\Rightarrow \frac{n}{n+\alpha+\beta}\times\frac{x}{n} + \frac{\alpha+\beta}{n+\alpha+\beta}\times\frac{\alpha}{\alpha+\beta}
\end{aligned}
$$
- $x/n$ represents $\hat{p}$, which is the estimate of the probability $p$ given _only_ the data (i.e. relative frequency). Think of this as the estimate obtained from sampling.
- Of course, $\frac{\alpha}{\alpha + \beta}$ is the prior mean. 
- In other words, we are weighing both the observed and the prior belief. This is exactly similar to the case when we considered both a normal likelihood and normal prior -- we weighed both the observed and the belief.
#### Poisson Likelihood, Gamma Prior (Exercise)

This is left as an exercise. Show that a Poisson Likelihood and a Gamma prior form a conjugate pair.

Hint: 

$$\large{
\begin{aligned}
x|\lambda \sim Poi(\lambda); &\quad f(x|\lambda) = \frac{\lambda^x}{x!}e^{-\lambda} \\ \\
\lambda \sim Ga(\alpha, \beta); &\quad \pi(\lambda) = \frac{\lambda^{\alpha-1}\beta^\alpha}{\Gamma(\alpha)}e^{-\beta\lambda} \\ \\
\end{aligned}
}$$
$$\large{
\pi(\lambda|x) \propto \lambda^{x+\alpha-1}e^{-(1+\beta)\lambda}
}$$

### Multiple Observations

#### Normal Likelihoods, Normal Prior

Recall the example with a normal likelihood and normal prior. In that example, only a single observation, $X$ was made. What happens if we have multiple (say $n$) observations, each of which is normal with parameters $\theta$ and $\sigma^2$? In other words, we have:
$$ X_1, \cdots, X_n \sim N(\theta, \sigma^2) $$
Of course, we still have our prior on $\theta$, which is:
$$ \theta \sim N(\mu, \tau^2)$$
Due to the **Sufficiency Principle**, it is sufficient to only incorporate data from all $X_n$ through $\bar{X}$. Paraphrased, we only take the distribution of $\bar{X}$ instead of considering every possible $X$:
$$ \bar{X} \sim N(\theta, \frac{\sigma^2}{n})$$
- This is a characteristic of Normal distributions. The mean is simply retained, $\theta$. The variance is simply divided by $n$. 
- This property of the new variance is true of any IID (Independent and Identically Distributed) distributions. The variance is simply divided by $n$.

Now we have two normal distributions for the likelihood and the prior, and we already know that these are conjugate pairs. In fact, since we have already calculated the parameters for the posterior from above, all we have to do is update the parameters accordingly.

$$\large{
\theta|X_1,\cdots,X_n \equiv \theta|\bar{X} \sim N\left(\frac{\tau^2}{\frac{\sigma^2}{n}+\tau^2}\bar{X} + \frac{\frac{\sigma^2}{n}}{\frac{\sigma^2}{n}+\tau^2}\mu, \frac{\frac{\sigma^2}{n}\tau}{\frac{\sigma^2}{n}+\tau^2}\right)
}$$

#### Poisson Likelihoods, Gamma Prior

As a continuation of the exercise above with Poisson likelihood and Gamma prior, given the following:

$$ X_i, \cdots, X_n \sim Poi(\lambda); \quad \lambda \sim Ga(\alpha, \beta)$$
Show that the posterior is:
$$ \lambda |x \sim Ga\left(\sum_i x_i + \alpha, n+\beta\right)$$
Hint:
$$ \sum_i X_i \sim Poi(n\lambda) $$
### Caveat

Conjugate families are convenient, since both the prior and posteriors turn out to be the same family. In short, we simply perform small updates to the parameters.

Unfortunately, conjugate families are not too common. Some pairs do exist, though they are not the most useful. Therefore, we have limited modelling abilities using conjugate families, although the computation _is_ simple if we are able to use them.

The need to go beyond conjugacy is unavoidable, since some priors cannot be conjugate with the likelihood. In this case, there is no escaping finding the marginal, $m(x)$ by integrating the product of the likelihood and the prior.

### Modelling with Conjugate Families

Here we will consider more concrete scenarios using the three conjugate pairs we have talked about above. 
1. Normal likelihood, Normal prior
2. Binomial likelihood, Beta prior
3. Poisson likelihood, Gamma prior
#### Jeremy's IQ (Normal, Normal)

Jeremy models his IQ as a normal distribution with mean $\theta$ and variance 80, $N(\theta, 80)$. What does this mean?
- If Jeremy takes multiple IQ tests, the results of these IQ tests should be centered about $\theta$, and there should be some scattering represented by the variance.

Jeremy is a student from GT. Given that GT is selective, we may have a prior on the IQ of a GT student elicited as $N(110, 120)$. 

Jeremy takes an IQ test and scores $X=98$.

Find the posterior for $\theta$.

Recall from above that the posterior distribution given normal likelihood and normal prior is:
$$ \theta | X \sim N\left(\frac{\tau^2}{\sigma^2 + \tau^2}x + \frac{\sigma^2}{\sigma^2+\tau^2}\mu, \, \frac{\sigma^2\tau^2}{\sigma^2+\tau^2}\right) $$
Subbing in the relevant values, we get:
$$\begin{aligned}
\theta | X &\sim N\left(\frac{120}{80 + 120}98 + \frac{80}{80+120}110, \, \frac{80\times120}{80+120}\right) \\ \\ 
&\sim N(102.9, 48)
\end{aligned}$$
The mean of the posterior is taken as the **Bayes estimator** of a parameter $\Rightarrow \widehat{\theta_B} = 102.8$.
 > We will cover Bayes estimators later but for now, just take it as the mean of the posterior.
 
The classical statistician will estimate $\theta$ using this single test as $\hat{\theta}_{MLE} = 98$. In our case, the prior opinion on $\theta$ influenced our estimator, and the value 102.8 lies between $98$, the observed score, and the prior mean on $\theta$, $110$.

Imagine now that Jeremy took five tests. In these five tests, the average score turned out to be $98$.

Again, recall from above that the posterior given these multiple observations is:
$$\large{
\theta|X_1,\cdots,X_n \equiv \theta|\bar{X} \sim N\left(\frac{\tau^2}{\frac{\sigma^2}{n}+\tau^2}\bar{X} + \frac{\frac{\sigma^2}{n}}{\frac{\sigma^2}{n}+\tau^2}\mu, \frac{\frac{\sigma^2}{n}\tau}{\frac{\sigma^2}{n}+\tau^2}\right)
}$$
Subbing in the relevant values, we get:
$$\begin{aligned}
\theta | X &\sim N\left(\frac{120}{\frac{80}{5} + 120}98 + \frac{\frac{80}{5}}{\frac{80}{5}+120}110, \, \frac{\frac{80}{5}\times120}{\frac{80}{5}+120}\right) \\ \\ 
&\sim N(99.4118, 14.1176)
\end{aligned}$$
Notice that the estimator ($99.4118$) is valued closer to the observations ($98$) instead of what was believed in the prior ($110$). Why is that?
- The variance of the observations is much smaller now, at $\frac{80}{5}$. Therefore, we put more weight on the more certain observations than the less certain prior.



 