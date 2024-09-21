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



