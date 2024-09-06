## Historic Overview

### Reverend Thomas Bayes

Before diving right into theory, it is perhaps interesting to learn about the founding father, Reverend Thomas Bayes. 

Bayes was estimated to be born in 1701 (or 1702, depending on sources) to a Presbyterian minister known as Joshua Bayes. In 1719, he enrolled in Edinburgh University to study logic and theology. He was a nonconformist, which meant that entry into Oxford and Cambridge was forbidden. He eventually died in 1791 in Bunhill Fields, London, where his body lies today.

Due to his nonconformist leaning, Bayes was not allowed publish most of his works. Still, he managed to publish a paper in Divinity titled _Principal End of the Divine Providence and Government is the Happiness of His Creatures_ (1731) and another in Mathematics titled _An Introduction to the Doctrine of Fluxions_ (1736). These two works granted him a place in the Royal Society in 1742.
- His Divinity paper argued that the role of God is to provide His creations happiness, which runs contrary to the then-widespread belief that God is primarily the provider of wisdom -- very non-conformist indeed.
- His Mathematics paper involved fluxions, which are an early iteration of differential expressions.
### The Essay

Bayes' seminal work, _An Essay toward solving a Problem in the Doctrine of Chances_ was communicated to the Royal Society by Richard Price on the 23rd of December, 1763. Two years after Bayes had died, Price had helped to finish the introduction and appendix.

In the essay, Bayes tried to answer a question not yet solved by the De Moivre in the _Doctrine of Chances_ (1738). Bayes' approach was geometric rather than analytic, as was the method of choice back then. Briefly, one might think of the geometric approach as more tangible and intuitive (e.g., picturing vectors as a line on a Cartesian plane), while the analytic approach involves abstractions and precise formulations. Unfortunately, the paper's impact was not felt until Laplace's rediscovery (in more formal and complete terms) of inverse probabilities around the mid 1770s.

The problem posed by De Moivre is quoted here:
> *Given* the number of times in which an unknown has happened and failed: *Required* the chance that the probability of its happening in a single trial lies somewhere between any two degrees of probability that can be named.

Recognise firstly that this is essentially invoking the binomial system. There are a set of repeated experiments, and the outcome is binary (something can either happen or fail). The probability of this outcome is unknown. Is it then possible to find the _probability_ that this probability lies between some two fixed bounds?

Why was Bayes even thinking of this in the first place? There are two theories.
- Some speculate that De Moivre was Bayes' private tutor, which motivated him to this specific niche.
- Others point to David Hume, a philosopher who published *Of Miracles*, in which he casts doubt on the existence of miracles as detailed by religious doctrines. Given Bayes' religious background, it is plausible that Bayes was searching for a refutation against Hume's claim.

---
## Bayesian v/s Classical

### The Fundamental Difference

Bayesian and Classical statistics differ primarily in how they treat **parameters**. In short, Classical statistics assumes that parameters are fixed numbers while Bayesian statistics assumes that parameters might be randomly following some distribution.

![[Pasted image 20240821154017.png]]

In Classical statistics, we conduct an experiment with fixed by unknown parameters. The inference drawn from the experiment is essentially concerned with the inference regarding the parameters themselves. 
- This was the Fisherian, parametric approach that started in the 1920s.

For the same task, the Bayesian method expresses the parameters by some distribution of the parameter's possible values (uncertain). 
###### DEFINITION: Prior and Posterior
> The **prior** represents what was originally believed before new evidence. The **posterior** takes the new evidence into account.

Therefore, the Bayesian method first takes the **prior**, which might be some known or assumed distribution of the parameter. The experiment is then run, which returns values that could be modeled as per the classical method. These values are termed "**likelihoods**". The prior is combined with the likelihoods to give a **posterior**, from which an inference can be made.
### 10 Coin Flips

This experiment should clearly demonstrate the differences in Classical and Bayesian statistics. Consider the following:
- A coin is flipped 10 times, and we are interested in the probability that the coin lands with heads facing up. In other words, we are interested in $p=P('H')$
- The result of this experiment gives us 0 heads and 10 tails.

We present the result of this experiment to both Classical and Bayesian statisticians.
#### The Classical Approach

The Classical statistician claims that the _relative frequency_ of the event is a good estimator for the probability of that event. Quite simply, they would calculate $\hat{p}$ by taking the frequency of successes over the number of trials. 

In this case, $\hat{p}$ is equal to 0, as $\frac{0}{10}$ is equal to 0. The probability that the coin flip results in heads would be reported as 0.

Understandably, if this inference was reported, one would wonder if it is at all possible for a two-sided coin to _never_ land with heads up. 
#### The Bayesian Approach

Recall that the Bayesian statistician amends the **prior** with information from the experiment (likelihood) to form the **posterior**. A brief note: we will be going into much more detail in later lectures, so do not worry too much about the gaps in reasoning here.

Suppose that the prior informs the statistician that any value between 0 and 1 is equally acceptable. In other words, the probability that the coin flip will result in heads is any value within the range $[0, 1]$. The prior is combined with the likelihood, which gives us the distribution on the left below:

![[Pasted image 20240821155243.png]]

The distribution is then normalised to give a **posterior distribution**, shown by the density plot on the right. Suppose for now that the normalisation is simply a factor of 11.

The posterior distribution can be thought of as the **ultimate** summary for the Bayesian. The Bayes estimate of $p$ can be either the mean, median, or mode of the posterior distribution. For example, if we were to use mean of the posterior distribution, the probability of obtaining heads would be $\frac{1}{12}$. This is by no means reasonable for a (supposedly) fair two-sided coin, but it is certainly more reasonable than 0!

---
## FDA Guidelines

This lecture concludes with a 2010 guideline put out by the US Food and Drug Administration (FDA), encouraging medical devices and clinical trials to abide by Bayesian statistics. Key points are provided here, and hopefully showcase the benefits of Bayesian statistics over the classical methods.

For medical devices:
1. Valuable prior information is often available for medical devices, as a result of their mechanism of action and evolutionary development.
2. Correctly employed Bayesian approaches may be less burdensome.
3. Often the use of prior information may alleviate the need for a larger sized trial.
4. When an adaptive Bayesian model is applicable, the size of a trial can be reduced by stopping early when conditions warrant.
5. Bayesian approaches to multiplicity problems may be advantageous (multiple endpoints and testing of multiple subgroups)
6. Bayesian methods allow for great flexibility in dealing with missing data.

For clinical trials:
1. An unlimited look at the accumulated data when sampling is of a sequential nature and will not affect the inference. In the frequentist approach, interim data analyses affect type I errors. 
2. The ability to stop a clinical trial early is important from moral and economic viewpoints.
3. Trials should be stopped early due both to **futility** (save resources or stop an ineffective treatment) or **superiority** (provide best possible treatments as fast as possible)
