## A Review of Necessary Probability

### Events and Probabilities

"Events" can be depicted in any manner, though they are often introduced via Set Theory. As such, events are often depicted as Venn diagrams (see below).

![[Pasted image 20240826191942.png]]

The figures below show the many possible interactions between events.

![[Pasted image 20240826192008.png]]

![[Pasted image 20240826192030.png]]

"Probabilities" on the other hand are **normed measures** of events. For example, the probability of a sure event is 1, while the probability of an impossible event is 0. For any event, the probability value ranges between 0 and 1 (inclusive).

A few quick refreshers:
- If $A$ and $B$ are exclusive events, then $P(A \cup B) = P(A) + P(B)$.
- $A^c$ is exclusive with $A$ (the complement is always exclusive). $S = A \cup A^c$ and $P(S) = 1$. Therefore, $P(A^c) = 1 - P(A)$.
- If $A$ and $B$ are arbitrary, then $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.
- Independence is not the same as exclusivity. We say that two events $A$ and $B$ are independent iff $P(A \cap B) = P(A)P(B)$.

### Example: Circuit Problem 

#### Series and Parallel

The figure below shows a circuit of events and their associated probabilities. It might be helpful to think of them quite literally as an electrical circuit. If a component fails and electricity has no way to travel through the circuit, the circuit fails. There are two basic systems of connection: (i) inline or serial, and (ii) parallel. 

![[Pasted image 20240826192129.png]]

In the serial system, the probability that the entire circuit works is the intersection of all probabilities (of the events). We are assuming here that all events are independent, of course. 

In the parallel system, the entire circuit works if any of the components work. Thus, it is much more convenient to find the probability of the system failing, then taking its complement (rather than take the union of each). The probability that the entire circuit fails is intersection of all probabilities of failure.

#### Complex Circuits

Given the following example:

![[Pasted image 20240826192205.png]]

Let
- $S_1 = A_2A_3$
- $S_2 = A_5 \cup A_6$
- $S_3 = A_4S_2$
- $S_4 = S_1 \cup S_3$
- $S = A_1S_4A_7$

It should be trivial to show that the probability of the system working is 0.15084.

## Conditioning

### Conditional Probability and Independence

We previously saw that if two events, $A$ and $B$ are independent, then $P(A \cap B) = P(A)P(B)$.  Note that this is **the definition of independence**. 
$$ A, B \text{ independent } \Leftrightarrow P(A \cap B) = P(A)P(B)$$

Recall that conditional probability is defined as follows:

$$ P(A|B) = \frac{P(A \cap B)}{P(B)} $$
> What does this mean? Recall that conditional probability can be read as "$A$ given $B$". The formula above is essentially asking from the the probability of both events $A$ and $B$ occurring, divided by the probability that $B$ even occurs in the first place.

Note that conditional probability necessitates $P(B)$ from being non-zero. Therefore, $B$ **cannot be an impossible event**.

From the formula above, we get:
$$ P(A \cap B) = P(B|A)P(A) $$
And by symmetry, we get:
$$ P(A \cap B) = P(A|B)P(B) $$
Finally, if $A$ and $B$ are independent, $P(A|B) = P(A)$ or $P(B|A) = P(B)$. This should be easy to prove.
> Intuitively, if $A$ and $B$ are independent, the probability of $A$ does not change given that $B$ has occurred!
### Queen of Spades

This example is often used to illustrate the "volatile" nature of independence, especially when events are derived from **a single sample space**. Consider the following scenario, where we have a standard deck of 52 cards. Naturally, there are 13 cards with the "spades" suit and 4 cards with the "Queen" rank. Suppose now that one card is selected at random.

Let $A$ be the event that a spade is selected, and $B$ be the event that a Queen is selected.

In order to check if $A$ and $B$ are independent, we have to prove that $P(A \cap B) = P(A)P(B)$.
$$ P(A \cap B) = \frac{1}{52} $$
$$ P(A) = \frac{13}{52}, \quad P(B) = \frac{4}{52} $$
$$ P(A)P(B) = \frac{1}{52} = P(A \cap B) $$
These events are clearly independent.

Now consider what happens if any non-spade, non-Queen card is removed from the deck. Suppose we remove the 2 of diamonds:

$$ P(A \cap B) = \frac{1}{51} $$
$$ P(A) = \frac{13}{51}, \quad P(B) = \frac{4}{51} $$
Comparing the two, we get $P(A)P(B) \neq p(A \cap B)$. These two events are now dependent!

### Hypotheses and Total Probability

#### Deriving Total Probability

In this section we will learn how to use conditional probabilities to derive the probability of an event.

Consider a sample space, $S$, and $n$ non-overlapping events (let's call them hypotheses, $H_n$, that split the total sample space. The figure below illustrates this idea. 

![[Pasted image 20240826194404.png]]

Notice that the sample space can be expressed as a union of all hypotheses, such that:
$$ S = H_1 \cup H_2 \cup \cdots \cup H_n$$
where no two hypotheses overlap:
$$ H_i \cap H_j = \emptyset, \quad i \neq j $$


Suppose we are interested in the probability of some event $A$ as depicted below:

![[Pasted image 20240826194708.png]]
$$ 
\begin{aligned}
A &= A \cap S \\ \\
&= A(H_1 \cup H_2 \cup \cdots \cup H_n) \\\\
&= AH_1 \cup AH_2 \cup \cdots \cup AH_n)
\end{aligned}
$$
Taking the probability of both sides, considering that all $AH_i$ are exclusive:
$$
\begin{aligned}
P(A) &= P(A \cap H_1) + P(A \cap H_2) + \cdots + P(A \cap H_n) \\ \\
&=P(A|H_1)P(H_1) + \cdots P(A|H_n)P(H_n)
\end{aligned}
$$

###### DEFINITION: Total Probability
$$ P(A) = \sum^n_{i=1} P(A|H_i)P(H_i) $$
> This formula is essentially taking the "weighted average", where the weights are the probabilities of hypotheses, $P(H_i)$.

#### Example: Manufacturing Bayes

Let us now consider an example. Suppose we have a production line with three machines. The relevant probabilities are given below:

![[Pasted image 20240826195431.png]]

One item is randomly selected from the production. What is the probability that the item is conforming? Let $H_i$ be the hypothesis that the item is from the $i$-th machine.

There are a few things to check before using the total probability formula.
1. Is the sample space completely split by the hypotheses? Yes, $H_1 \cup H_2 \cup H_3 = S$. Alternatively, check that $\sum P(H_i) = 1$.
2. Are the hypotheses exclusive (non-overlapping?). Yes, $H_i \cap H_j = \emptyset$.
3. Do we know the probabilities of these hypotheses? Yes, this is equivalent to the production volume (percentage).

We also know the following conditional probabilities:
- $P(A|H_1) = 0.94$
- $P(A|H_2) = 0.95$
- $P(A|H_3) = 0.97$

By total probability, $P(A)$ is therefore equal to:
$$ (0.94 \times 0.3) + (0.95 \times 0.5) + (0.97 \times 0.2) = 0.951 $$

#### Example: Bridged Circuit

Consider a more challenging problem. Find the probability that the overall circuit shown below works:

![[Pasted image 20240826200233.png]]

The issue with this circuit is element $A_5$ -- it cannot be considered simply serial or parallel. There are many ways to solve this problem, but one way using the total probability formula would be to bisect the total sample space into two possible scenarios: either $A_5$ works, or it fails.

More concretely, let $S$ be the probability that the circuit works. 
- $H_1$ be the hypothesis that $A_5$ works, and
- $H_2$ be the hypothesis that $A_5$ fails.
> Check that the two hypotheses neatly split the sample space, and are not overlapping. 

Now the total probability of (S) can be calculated as:

![[Pasted image 20240826200610.png]]
Finally,

![[Pasted image 20240826200826.png]]

## Bayes Formula

### Prior to Posterior Probabilities

Suppose that we are interested in $P(H_i|A)$ for some hypothesis $H_i$. In other words, we are trying to "update" our hypothesis, given that event $A$ has occurred. Note that this is different from the previous section, where we were concerned with $P(A)$.

By Total Probability, we can determine $P(A)$. Therefore,
$$ \begin{aligned}
P(H_i|A) &= \frac{P(AH_i)}{P(A)} \\ \\
&= \frac{P(A|H_i)P(H_i)}{P(A)}
\end{aligned}
$$
This is also known as Bayes Formula.  In essence, we first know something about the hypothesis -- denote this **the prior**, $P(H_i)$. Some event $A$ happened, which allows us to update our hypothesis -- this is **the posterior**, $P(H_i|A)$.

### Example: Manufacturing Bayes cont.

Let us return to the example scenario described above. For ease of reference, the relevant probabilities will be copied here:

![[Pasted image 20240826195431.png]]

Now, suppose that the selected item was found to be conforming. What is the probability that it was produced on Machine 1?
- Prior to selection, $P(H_1)$ = 0.3. More simply put, before the item was found to be conforming, the probability that an item was from Machine 1 is simply 0.3.
- From the Total Probability formula, we calculated above that $P(A) = 0.951$.
- Therefore: 
$$P(H_i|A) = \frac{P(AH_i)}{P(A)} = \frac{0.94 \times 0.3}{0.951} = 0.2965 $$
Notice that the calculated probability is slightly lower than $0.3$.

### Example: Bridged Circuit cont.

Again, recall the following problem:

![[Pasted image 20240826200233.png]]

Suppose that the circuit $S$ works. What is the probability that the element $A_5$ works as well?
- Recall that we let $H_1$ be the hypothesis that $A_5$ works. Therefore, $P(H_1)$ = $P(A_5 \text{works}) = 0.6$.
- By Bayes Formula:
$$ P(H_1|S) = \frac{P(S|H_1)P(H_1)}{P(S)} = \frac{0.8536 \times 0.6}{0.8315} = 0.6159 $$
Notice that the calculated probability is slightly higher than $0.6$.

### Example: Two-Headed Coin

Let us now consider a particularly interesting problem. 

In a box, there are $N$ coins, $N-1$ coins are fair, while one is two-headed. A coin is selected from the box and flipped $k$ times. In all $k$-flips, the coin came heads-up. What is the probability that the two-headed coin was selected?

> This is an example of a scenario that is Bayesian. First, we have some prior probability -- the probability that the two-headed coin was selected is simply $1/N$. An experiment is conducted (or an event happened), which provides more information for us to revise our prior, forming the posterior probability. 

Let:
- $A$ be the event that the coin lands heads up $k$ times in $k$ flips,
- $H_1$ be the hypothesis that the fair coin is selected, and
- $H_2$ be the hypothesis that the two-headed coin is selected.

 Then, we have the following:
$$ P(H_1) = \frac{N-1}{N}, \quad P(H_2) = \frac{1}{N}$$
$$ $P(A|H_1) = \frac{1}{2^k}, \quad P(A|H_2)=1$$
Still, to apply Bayes Formula, we would need $P(A)$, which we can calculate using the Total Probability formula:
$$ P(A) = (\frac{N-1}{N} \times \frac{1}{2^k}) + (\frac{1}{N} \times 1) = \frac{N-1+2^k}{2^kN} $$
Applying Bayes Formula, we get:
$$ P(H_2|A) =\frac{\frac{1}{N} \times 1}{\frac{N-1+2^k}{2^kN}} = \frac{2^k}{(N-1)+2^k}$$
This is all and well, but let us plug in real numbers to intuit what exactly the above means. Suppose we have $N = 1,000,000$ coins, and we perform the coin flip $k= 20$ times. 
$$ P(H_2) = \frac{1}{1,000,000}, \quad P(H_2|A) = 0.5119 $$
The probability that the two-sided coin was picked increased to $0.5119$. 

Suppose that we perform the coin flip $k=40$ times now.
$$ P(H_2) = \frac{1}{1,000,000}, \quad P(H_2|A) = 0.0.9999999095 $$
Now, we can be almost certain that the coin picked was indeed the two-headed coin, even though there was only a one in a million chance that the coin would be picked randomly. Again, given the results of the experiment, we are able to modify our initial hypothesis and its probability accordingly. This is also known as **Bayes Learning**.

## Basic Bayes Networks

### Brief Overview

Let us now build upon the content in this unit. This will still be concerned with event space, and will not have anything to do with Bayesian inferences for now.

Define a Bayes Network as follows:
- It is an directed (oriented) acyclic graph, consisting of nodes (events, random variables, etc.) and directed edges (causal relations).
- Recall the example with circuits. In a somewhat similar manner, if we were to be concerned with the overall network, this is more formally termed as computing the **joint distribution of all nodes**.

### Joint Distribution; Overview

The joint distribution of all nodes is simply the product of conditional distributions of all nodes -- this is potentially quite complicated. However, it helps that directed acyclic graphs possess the Markovian property, in which a node **only depends on its immediate parents**. In other words, nodes that are not immediately connected to a given node has no effect on it's conditional distribution. 

In Bayes networks, there will be a mixture of observed nodes (where we have evidence), and unobserved nodes (where we will learn). In the event space, the above statement can be rephrased as, "we have a bunch of nodes with known probabilities, and we can learn probabilities of previously unknown nodes".

### ALARM

This is a paradigmatic example in introducing Bayes Networks.

![[Pasted image 20240827205543.png]]

Suppose that you live in Downtown Los Angeles. California is known to have frequent earthquakes -- denote this event with $E$. Unfortunately, L.A. is also prone to burglary -- denote this event $B$. Both of these events lead to the alarm triggering -- denote this event $A$. 

The probabilities of an earthquake or a burglary occurring are given as $P(B)$ and $P(E)$ above.

The probabilities of the alarm going off, given the presence and/or absence of earthquakes or burglaries are also given above. Note that even if there is no burglar or earthquake, the alarm has a 0.001 probability of triggering -- maybe a bird flew by really quickly.

An alarm trigger might in turn lead to two possibilities. Your friend John might call to ask if everything is alright, $J$, or your neighbour Mary might call to ask instead, $M$. John and Mary do not know each other, therefore these events are independent. If they hear the alarm they call you, but this is not guaranteed. They also call you from time to time just to chat.







