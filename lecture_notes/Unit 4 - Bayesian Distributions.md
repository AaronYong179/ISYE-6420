## Basic Distributions

### Random Variables

The random variable is the simplest statistical model.  The lecturer claims that "random variable" is a bit of a misnomer; a random "variable" is actually a mapping (function) from the sample space $S$ to real numbers. The figure below illustrates this idea, where the mapping (random variable) is depicted as $X$:

![[Pasted image 20240830194800.png]]

For example, flip a fair coin. If the coin turns up heads, you get $1. If the coin turns up tails, you pay $1. Your gain in one flip is a random variable -- it maps the sample space (heads or tails) to two real numbers (-1 or +1).
#### Discrete vs Continuous

The random variables are divided by their nature of **realisations**: discrete and continuous. These random variables are fully determined by distributions.

Recall that the distribution for discrete random variables are often modelled as a table. We have the random variable $X$, and all possible realisations (values) for $X$ are listed. To each $x_i$, we then assign the relevant probability.

![[Pasted image 20240830195425.png]]

This table is also known as the **probability mass function** (PMF) as the probabilities are atomic -- each probability belongs to a particular discrete variable. Note that the sum of all probabilities will be 1, which means that some $x_i$ must occur.

Continuous random variables on the other hand involve realisations that are points in a continuous interval, which might be finite or infinite in nature. Naturally, the tables used for discrete random variables would no longer suffice -- we require a **density function** instead.

![[Pasted image 20240830195742.png]]

Note that the function cannot return any negative values, since density ultimately represents probability.  Also note that the integral of the density function should be equal to 1. 
#### Important Distributions

##### Bernoulli Distribution (Discrete)

| X    | 0   | 1   |
| ---- | --- | --- |
| Prob | q   | p   |
$q = 1-p, \quad 0 \leq p \leq 1$

- The Bernoulli distribution is the simplest discrete distribution, consisting of only two realisations: 0 and 1 (either something happened or did not happen).
- Although very simple, the Bernoulli distribution is used to model many things, especially events where binary outcomes are involved.
- The parameter of the Bernoulli distribution is $p$. We denote this as $X \sim Ber(p)$.
##### Binomial Distribution (Discrete)

| X    | 0     | 1     | ... | n     |
| ---- | ----- | ----- | --- | ----- |
| Prob | $p_0$ | $p_1$ | ... | $p_n$ |
$p_k = P(X=k) = {n \choose k} p^k q^{n-k}, \quad q = 1-p, \quad k = 0, 1, \cdots, n$

- An intuitive way to think about Binomial distributions would be to consider $n$ independent experiments, in which some event, $A$, might appear in each experiment with some probability $p$. For instance,  we could count the **number of times** a coinflip results in a heads, given that the coinflip happened $n$ times.
- Clearly then, the realisations range from 0 to $n$. The event might never occur given all $n$ experiments, or it might occur in all experiments!
- The name "Binomial" reflects the fact that these probabilities are parts of the Binomial series.
- The parameters for the Binomial distribution are $n$ and $p$. We denote this $X \sim Bin(n, p)$.

##### Poisson Distribution (Discrete)

| X    | 0     | 1     | ... | n     | ... |
| ---- | ----- | ----- | --- | ----- | --- |
| Prob | $p_0$ | $p_1$ | ... | $p_n$ | ... |
$p_k = P(X=k) = \frac{\lambda^k}{k!}e^{-\lambda}, \quad k=0, 1, \cdots, n, \cdots$

- The Poission distribution appears to be similar to the Binomial distribution, but note that the Poission distribution **is not bounded from above**. In other words, $X$ can realise values that grow beyond $n$ -- perhaps even to infinity. 
- The parameter, $\lambda$, is also termed the "rate", and is defined by the product of $n$ and $p$. We denote this $X \sim Pois(\lambda)$.
- The Poisson distribution is useful in modelling phenomena where counts are involved (similar to the Binomial distribution). However, the Poisson distribution is useful as a **limiting distribution**; that is, if we have a Binomial distribution with $n \rightarrow \infty$ and $p \rightarrow 0$, we approach a Poisson distribution with parameter $\lambda$.  
- In that case, the Poisson distribution models events that might happen a large number of times ($n \rightarrow \infty$), but happen rarely ($p \rightarrow 0$). 

I quite liked this Reddit comment explaining why we would need the Poisson distribution:
> The Poisson distribution is the limit of binomial distributions. If you have 10 independent opportunities, each with probability _p_, then you can use a binomial distribution to find the probability of getting exactly 0, or 1, or however many successes. If you have 100 independent opportunities, each with probability _p_, you can again do this. But if you have _infinitely many_ opportunities, you can't do that. You need some sort of limit for this to be well-defined. That's when you use the Poisson distribution.

##### Uniform Distribution (Continuous)

$$ f(x) =
\begin{cases}
\frac{1}{b-a} \quad \text{if} \quad a \leq x \leq b \\
0 \quad\quad \text{else}
\end{cases}
$$
- The uniform distribution is a distribution with flat density -- think of this as a straight horizontal line, or as a square wave.
- In other words, the probability of the random variable falling in any part of some fixed interval $a \leq x \leq b$ is constant. Notice that this probability is proportional to the length of the fixed interval.
- The parameters here are quite simply the boundaries, $a$ and $b$. We denote this distribution as $X \sim U(a, b)$.
##### Exponential Distribution (Continuous)

$$ 
\begin{aligned}
f(x) &= \lambda e^{-\lambda x}, \quad \lambda >0, x>0 \\ \\
&\text{or} \\ \\
f(x) &=\frac{1}{\mu}e^{-x/\mu}, \quad \mu>0, x>0
\end{aligned}
$$

- The exponential distribution can be modelled using the parameter $\lambda$ (rate), but it might also be commonly parameterised by $\mu$ (scale). Both density functions are given above.
- Note that $\lambda$ and $\mu$ are simply reciprocal to each other.
##### Beta Distribution (Continuous)

$$
f(x) = \frac{1}{B(a, b)}x^{a-1}(1-x)^{b-1}, \quad 0 \leq x \leq 1
$$
$$ B(a, b) = \int_0^1 x^{a-1}(1-x)^{b-1} dx$$
- The Beta distribution is going to be interesting for us, as the distribution is bounded between 0 and 1 -- it is often used to model probability.
- The normalising constant involves the Beta function, denoted and defined above as $B(a, b)$. For the Beta distribution, $a$ and $b$ are the parameters, and we denote this distribution $X \sim Be(a, b)$.
- We will see in later sections how the shape of the Beta distribution changes based on parameters $a$ and $b$.
##### Normal Distribution (Continuous)

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}, -\infty < x < +\infty, -\infty < \mu < +\infty, \sigma > 0
$$
- The normal distribution is particularly special, and for good reason. The sum of distribution (be it discrete or continuous), given that many samples are taken, will always approach the normal distribution. Recall that this is the Central Limit Theorem.
- The normal distribution can therefore be called a limiting distribution under very mild conditions. 
- $\mu$ and $\sigma$ are used as parameters. We denote this as $X \sim N(\mu, \sigma^2)$.
- It is also possible to encounter differently parameterised variants of the normal distribution. Suppose we let $X \sim f(x)$:
	- if $f(x-\mu)$, then $\mu$ is used as the shift parameter
	- if $\frac{1}{\sigma}f(\frac{x}{\sigma})$, then $\sigma$ is used as the scale parameter
	- if $f(x)$ depends on parameter $\theta$, we write $f(x|\theta)$. We will see this in later sections.

#### Cumulative Distribution Function (CDF)

Recall that we modelled discrete and continuous random variables differently.
- The discrete random variables were modelled using a table, known more formally as the **probability mass function** (pmf).
- The continuous random variables were modelled as a density function, known more formally as the **probability density function** (pdf).

The cumulative distribution function (cdf) however, is defined for **both** discrete and continuous random variables. The definition is given as:

$$
F(x) = P(X \leq x) = 
\begin{cases}
\sum_{x_i<x}P(X=x_i) \\ \\
\int_{-\infty}^x f(x) dx
\end{cases}
$$
Notice that the cdf is simply the sum of probabilities to the _left_ of $x$. Let us take an example, using $X \sim Exp(\lambda)$.

Recall that the exponential density function is:
$$ f(x) = \lambda e ^{-\lambda x}, x \geq 0$$
The cdf is the integral of this function:
$$
\begin{aligned}
F(x) &= \int_{-\infty}^x f(t) dt\\ \\
&= \int_0^x \lambda e^{-\lambda t} dt \\ \\
&= -e^{-\lambda x} - (-1) \\ \\
&= 1 - e^{-\lambda x}, x \geq 0
\end{aligned}
$$

The cdf plots given various values of $x$ and $\lambda$ are shown below:

![[Pasted image 20240831062823.png]]

Notice that the graph is bounded on the y-axis by 0 and 1. The sum of probabilities can only range from 0 to 1, after all. 
### Numerical Characteristics

In this subsection, we will look at numerical characteristics of distributions. Examples include the moments, expectations, variance, quartiles, and so on. 
#### Expectation and Raw Moments

The expectation is simply a **weighted average** of realisations, where the weights are the probabilities of each realisation. 
$$
E(X) = 
\begin{cases}
x_1p_1 + x_2p_2 + \cdots + x_np_n + \cdots, \quad \text{discrete distribution} \\ \\
\int_\mathbb{R} x\cdot f(x) dx, \quad \text{continuous distribution}
\end{cases}
$$
> Recall that $f(x)$ is the density of $x$. 


The definitions given above extend to the $k$-th moment (also known as $k$-th raw moments) as well, with the following amendments for discrete and continuous distributions respectively:
$$
E(X) = 
\begin{cases}
 x_1^kp_1 + x_2^kp_2 + \cdots + x_n^kp_n + \cdots, \quad \text{discrete distribution} \\ \\
\int_{\mathbb{R}} x^k \cdot f(x) dx, \quad \text{continuous distribution}
\end{cases}
$$

In fact, these definitions extend more generally to any function $\varphi$, where $\varphi$ is simply applied to the realisations $x_i$. More concretely,

$$
E(\varphi(X)) = 
\begin{cases}
 \varphi(x_1)p_1 + \varphi(x_2)p_2 + \cdots + \varphi(x_n)p_n + \cdots, \quad \text{discrete distribution} \\ \\
\int_{\mathbb{R}} \varphi(x) \cdot f(x) dx, \quad \text{continuous distribution}
\end{cases}
$$

#### Variance and Standard Deviation

The variance is a **central moment**. 
$$ \text{Var}(X) = E(X - E(X))^2 = E(X^2) - E(X)^2 $$
Consider what is happening here exactly:
- Take the random variable $X$ and subtract the expectation of $X$. Think of this as centering $X$ around the expectation (hence, "central" moment). You may also think of this as shifting the distribution such that the expectation is centered back at the origin.
- Then, square the above value and calculate the expectation of the squares. This is therefore the **second central moment**. 
- The final expanded version on the right-hand-side shows the expectation of $X^2$ minus the squared expectation of $X$. It is proven that $E(X^2) \geq E(X)^2$, hence variance is always positive, with a minimum value of 0. 

The standard deviation is simply the square root of the variance.
$$ \sigma_x = \sqrt{\text{Var}(X)}$$
#### Quantiles

We say that $\xi_p$ is the $p$-th quartile of distribution $F$ if $F(\xi_p) = p$. Recall that $F$ refers to the CDF. In other words, given the CDF, the if the x-axis value is $\xi_p$, the y-axis value will be $p$. Considering the distribution itself, we can say that the area to the left of $\xi_p$ is equal to $p$. 

For discrete random variables, the $p$-th quantile is calculated as:
$$ \xi_p = \text{inf}\{ x | \sum_{x_i \leq x} p_i \geq p \}$$
 For continuous random variables, the $p$-th quantile is calculated as:
 $$ \int_0^{\xi_p} f(x)dx = p \quad \text{OR}\quad F^{-1}(p) = \xi_p$$
 For example, if we have the exponential distribution $X \sim Exp(\lambda)$, we already know that the corresponding cdf is $F(x) = 1 - e^{-\lambda x}$, where $\lambda > 0, x\geq 0$.

Therefore for an exponential distribution, we have the following equation for $\xi_p$. Note that the natural logarithm is used.
$$ \xi_p = - \frac{1}{\lambda}\log(1-p), \quad 0 \leq p \leq 1 $$

There are a few special quantiles:
- The median is defined as $\xi_{1/2}$. This is more commonly known as the 50% percentile, or 0.5-quantile.
- The first and third quantiles are denoted as $Q_1$ and $Q_3$, and are defined as $\xi_{1/4}$ and $\xi_{3/4}$ respectively.

#### Mode

The mode is simply the most frequent, or the most likely value. 
- For a continuous distribution, the mode is a value that maximises density. 
- For a discrete distribution, the mode is some value $x_i$ for which $p_i = P(X = x_i)$ is maximum. (i.e. maximises probability).

For example, taking the exponential distribution $X \sim Exp(\lambda)$ again:
- The median is $\frac{\log2}{\lambda}$ 
- The mode is 0

The mode, median, and mean are all **location measures**. 

### Joint and Conditional Distributions

This topic is very important for Bayesian inference, as our inference, representations of posteriors, models, etc. are going to be in terms of conditional distributions. Also, our normalising constants are going to involve **marginal distributions**.

In this context, we will be talking about a **vector** of random variables, $X = (X_1, X_2, \cdots, X_n)$.

The joint distribution of this vector is simply given as a PDF where $f(x_1, x_2, \cdots, x_n)$. The CDF is also given as a function on $n$ variables, $F(x_1, x_2, \cdots, x_n)$. Each variable $x_i$ here corresponds to each component.

Perhaps the most important vector for us would be simply two dimensional. That is:
$$ X = (X_1, X_2) $$
We will be going through the marginal distribution and the conditional distribution for this 2D vector, and we will be covering both the discrete and continuous case. This 2D vector is also known as the **bivariate random variable**. 
#### Conditional Distribution (2D)

Let us take a look at the conditional distribution of $X_1$ given $X_2$. 

Recall the following formula with regards to conditional probabilities and **events**:
$$ P(A|B) = \frac{P(AB)}{P(B)} $$
In a similar vein, the **conditional density** can be defined as the **joint density** over the **marginal density**.
$$ f(x_1 | x_2) = \frac{f(x_1, x_2)}{f(x_2)}$$
The marginal density of course comes from the marginal _distribution_. Here, $f(x_2)$ is marginal for $X_2$, defined as $\int_{-\infty}^{+\infty} f(x_1, x_2) dx_1$.
##### Example: Discrete 2D

Firstly, it is important to understand how discrete 2-D random variables are depicted. Recall that with a singular discrete random variable, we have a table as follows:

| $X$  | $x_1$ | $x_2$ | $x_3$ |
| ---- | ----- | ----- | ----- |
| prob | $p_1$ | $p_2$ | $p_3$ |

With two discrete random variables, the header row and column denote the values of $X$ and $Y$ (suppose here that $X$ and $Y$ are two discrete random variables). The table cells themselves represent the probability of the pair occurring. 

![[Pasted image 20240902164738.png]]

Now that we have the 2-D discrete joint distribution, how do we find the **marginal distributions** of $X$ and $Y$? From above, the marginal distribution is defined as an integral. However, for discrete random variables, the "integral" boils down to a simple summation.

Therefore, calculating the marginal distribution of $X$, we take the sum of all realisations of $X$ ($P(X = 0)$, and so on), with respect to all corresponding values of $Y$. In short, we will obtain the following table (think of this as a row sum):

| $X$ | Prob |
| --- | ---- |
| 0   | 0.5  |
| 1   | 0.25 |
| 2   | 0.25 |
The same thing is true for calculating the marginal distribution of $Y$. We take the sum of all realisations of $Y$ ($P(Y=-1)$, and so on), with respect to all corresponding values of $X$. Think of this as a column sum:

| $Y$  | -1   | 0    | 1    |
| ---- | ---- | ---- | ---- |
| Prob | 0.25 | 0.45 | 0.30 |
Let us now determine the **conditional distribution** for $X$ and $Y$. We shall start with the conditional distribution of $X$, given that $Y=0$. As usual, the realisations of $X$ remain unchanged. They are: 0, 1, and 2. Going back to the joint distribution table, notice that column 2 is of interest here (i.e., the realisations of $X$ given $Y=0$.

However, notice also that the sum of column 2 (which is incidentally the marginal density of $Y$ at $Y=0$ is **not equal to 1**! This is not a bona fide probability distribution. To remedy this, we divide the probabilities observed in the joint distribution table over the relevant marginal density.

| $X\|Y=0$ | Prob                              |
| -------- | --------------------------------- |
| 0        | $\frac{0.3}{0.45} = \frac{2}{3}$  |
| 1        | $\frac{0.05}{0.45} = \frac{1}{9}$ |
| 2        | $\frac{0.1}{0.45} = \frac{2}{9}$  |
It should be trivial to extend this to the conditional distribution of $Y|X=2$.

| $Y\|X=2$ | -1  | 0   | 1   |
| -------- | --- | --- | --- |
| Prob     | 0.2 | 0.4 | 0.4 |

##### Example: Continuous 2D

For continuous distributions, there is no avoiding (a little bit of) algebra. Let us consider this problem:

If $\large{f(x, y) = \frac{1}{\pi}e^{-1/2(x^2-2xy+5y^2)}}, x, y \in \mathbb{R}^2$, find marginal distributions for $X$, $Y$, and find conditional distributions for $X|Y = y$ and $Y|X=x$.

Given the joint distribution of the bivariate random variable, we can rearrange the terms to have the following:

$$
\large{
\begin{aligned}
f(x, y) &= \frac{1}{\pi}e^{\frac{1}{2}(x^2-2xy+5y^2)} \\ \\
&= \frac{1}{\pi}e^{-2y^2}e^{\frac{1}{2}(x-y)^2} \\ \\
&= \frac{1}{\pi} \cdot e^{-2y^2} \cdot \sqrt{2\pi} \cdot \frac{1}{\sqrt{2\pi}}e^{\frac{1}{2}(x-y)^2}
\end{aligned}}
$$

This rearrangement is simply for ease of integration later.

$$
\large{
\begin{aligned}
f(y) &= \int_{\mathbb{R}} f(x, y) dx \\ \\
&= \sqrt{\frac{2}{\pi}}e^{-2y^2} \cdot \int_{\mathbb{R}} \sqrt{2\pi} \cdot \frac{1}{\sqrt{2\pi}}e^{\frac{1}{2}(x-y)^2} dx \\ \\ 
\end{aligned}}
$$
Notice the following:
- We are only integrating the expression containing $x$, since we are integrating only with respect to $x$. 
- Notice that the expression containing $x$ (right hand side) is essentially that of a normal distribution involving $x$ with variance = 1 and shift = $y$. Therefore, the integral of the term $\large{\frac{1}{\sqrt{2\pi}}e^{\frac{1}{2}(x-y)^2}}$ is equal to 1.

We now have:

$$
\large{
\begin{aligned}
f(y) &= \sqrt{\frac{2}{\pi}}e^{-2y^2} \cdot  \\ \\ 
\end{aligned}}
$$
Rearranging the terms once more, we get:

$$
\large{
\begin{aligned}
f(y) &= \frac{1}{\sqrt{2\pi\cdot\frac{1}{4}}}e^{-\frac{y^2}{2\cdot 1/4}} \\ \\ 
\end{aligned}}
$$
Again, this is similar to a normal distribution involving $y$ with variance $= 1/4$ and mean $= 0$.  In conclusion, $Y \sim N(0, (\frac{1}{2})^2)$.

The same algebraic trickery occurs when calculating $f(x)$.  The following steps are presented, but do practice the steps again.

$$\large{
\begin{aligned}
f(x, y) &= \frac{1}{\pi}e^{\frac{1}{2}(x^2-2xy+5y^2)} \\ \\
&= \frac{1}{\pi}e^{-\frac{5}{2}(y^2-\frac{2x}{5}y + \frac{x^2}{25}-\frac{x^2}{25}+\frac{x^2}{5})} \\ \\
&= \frac{1}{\pi}e^{-\frac{5}{2}\cdot\frac{4}{25}x^2} \cdot e^{-\frac{5}{2}(y-\frac{x}{5})^2} \\ \\
&= \frac{1}{\pi}\cdot e^{-\frac{2}{5}x^2} \cdot \sqrt{2\pi\cdot\frac{1}{5}} \cdot \frac{1}{\sqrt{2\pi\cdot\frac{1}{5}}} \cdot e^{-\frac{1}{2\cdot\frac{1}{5}}(y-\frac{x}{5})^2} \\ \\ \\
f(x) &= \frac{1}{\sqrt{2\pi\cdot\frac{5}{4}}}e^{-\frac{x^2}{2\cdot\frac{5}{4}}} \cdot \int_{\mathbb{R}} \frac{1}{\sqrt{2\pi\cdot\frac{1}{5}}} \cdot e^{-\frac{1}{2\cdot\frac{1}{5}}(y-\frac{x}{5})^2} \\ \\ 
f(x) &= \frac{1}{\sqrt{2\pi\cdot\frac{5}{4}}}e^{-\frac{x^2}{2\cdot\frac{5}{4}}} \cdot 1 \\ \\ 
&\rightarrow X \sim N(0, (\frac{\sqrt{5}}{2})^2)
\end{aligned}
}$$

Thus, the marginals are normal $X \sim N(0, (\frac{\sqrt{5}}{2})^2)$ and $Y \sim N(0, (\frac{1}{2})^2)$.

We are not yet done! Remember that we still have to find the conditional distributions $f(x|y)$ and $f(y|x)$. 

By definition, $f(x| y) = \frac{f(x, y)}{f(y)}$ (as introduced above). We know the joint distribution, $f(x, y)$, as this was given, and we have already calculated the marginal distributions $f(x)$ and $f(y)$ above. Therefore,

$$\large{
\begin{aligned}
f(x|y) &= \frac{\frac{1}{\pi}e^{-\frac{1}{2}(x^2-2xy+5y^2)}}{\frac{1}{\sqrt{2\pi\cdot\frac{1}{4}}}\cdot e^{-\frac{1}{2}\cdot4y^2}} \\ \\
&= \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(x-y)^2}, \quad x\in \mathbb{R} \\ \\
&\rightarrow X|Y = y \sim N(y, 1)
\end{aligned}
}$$
Similarly for $f(y|x)$, 
$$\large{
\begin{aligned}
f(y|x) &= \frac{\frac{1}{\pi}e^{-\frac{1}{2}(x^2-2xy+5y^2)}}{\frac{1}{\sqrt{2\pi\cdot\frac{5}{4}}}\cdot e^{-\frac{1}{2}\cdot x^2\cdot\frac{4}{5}}} \\ \\
&= \frac{1}{\sqrt{2\pi\cdot\frac{1}{5}}}e^{-\frac{1}{2\cdot\frac{1}{5}}(y - \frac{x}{5})^2}, \quad y\in \mathbb{R} \\ \\
&\rightarrow Y|X = x \sim N(\frac{x}{5}, (\frac{1}{\sqrt{5}})^2)
\end{aligned}
}$$
##### Example: Independent 2D

Let us now take a look at an easier example. If $f(x, y) = 2xe^{-x-2y}$, where $x \geq 0, y \geq 0$, find conditional distributions for $f(x| y)$ and $f(y|x)$.

$x$ and $y$ are deemed "separate variables", as the joint distribution can be represented as a product of two densities.
$$f(x, y) = xe^{-x} \cdot 2e^{-2y} = f(x)\cdot f(y) $$
Therefore, $X$ and $Y$ are independent components. $f(x|y) = f(x)$ and $f(y|x) = f(y)$.

Take a moment to understand why this simple reorganisation works. 
- In order to calculate the marginal distributions, we integrate $f(x, y)$ with respect to $x$ or $y$. More concretely, if we were interested in finding the marginal distribution $f(y)$, we would fix $2e^{-2y}$ as a constant and integrate over $xe^{-x}$ with respect to $x$.
- $f(x|y)$ is defined as $\large{\frac{f(x, y)}{f(y)}}$. From the factorisation above, we get that $f(x|y) = \frac{f(x)f(y)}{f(y)} = f(x)$.

In summary, whenever we have the **independence condition** whereby the joint distribution splits into the product of two densities, the conditional distributions are equal to the marginals.






