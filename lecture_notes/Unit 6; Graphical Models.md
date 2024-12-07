This unit (and all units onwards) will be more programming focused. However, this unit in particular isn't really meaningful or long. Feel free to skip this with little to no repercussions.
## Introduction

Graphical models in the Bayesian context are simply directed acyclic graphs (DAG). Each node represents _not_ an event, rather a random variable. The directed edges correspond to statistical dependencies between nodes.

Why do we use graphical models in the first place?
- These are natural and intuitive representations of relationships between random variables (circuits, social networks, neural networks, etc.)
- It is an easy way to immediately access conditional independence. 
- It provides enhanced efficiency in inference by Markovian property (recall: only the parental nodes matter)
### Markovian Property

##### Reversed Conditional Independence

We have already touched upon this concept, though we will define it more concretely here.

Given the graph with nodes $A \rightarrow B \rightarrow C$, we say that it satisfies the Markovian property if $P(C|A, B) = P(C|B)$.

Now, is this true in the opposite direction? Is $P(A|B, C) = P(A|B)$? It actually is. Let us take a look at the short proof of this.

$$
\begin{aligned}
P(A|B, C) &=\frac{P(B, C|A)P(A)}{P(B, C)} \quad \text{[ Bayes Rule ]} \\ \\
&= \frac{P(B|A)P(C|A,B)P(A)}{P(C|B)P(B)} \quad \text{[ Chain Rule ]} \\ \\
&= \frac{P(B|A)P(C|B)P(A)}{P(C|B)P(B)}\quad \text{[ Markovian Property ]} \\ \\
&= \frac{P(B|A)P(A)}{P(B)} = P(A|B) 
\end{aligned}
$$
Therefore, note that the Markovian property **also works backwards**.

##### Non-Neighbour Conditional Independence

###### Case 1
Now let us consider the graph with nodes $A \rightarrow B \rightarrow C$ once more. Are $A$ and $C$ independent given $B$ ($A \perp \!\!\! \perp C | B$)? Alternatively, we may ask if $P(A, C|B) = P(A|B) \times P(C|B)$? The answer is yes.

Again, here's a short proof:
$$
\begin{aligned}
P(A,C|B) &= \frac{P(A, B, C)}{P(B)} = \frac{P(A)P(B|A)P(C|A,B)}{P(B)} \\ \\
&= \frac{P(A)P(B|A)P(C|B)}{P(B)} \\ \\
&= P(A|B)\times P(C|B)
\end{aligned}
$$
###### Case 2
What about the graph with nodes $A \leftarrow B \rightarrow C$? Given $B$, $A$ and $C$ are also independent.
$$
\begin{aligned}
P(A,C|B) &= \frac{P(A,B,C)}{P(B)} = \frac{P(B)P(A|B)P(C|B)}{P(B)} \\ \\
&= P(A|B)\times P(C|B)
\end{aligned}
$$
###### Case 3
A final alternative: what if $A$ and $B$ are both parents of $C$? In other words, we have the graph with nodes $A \rightarrow B \leftarrow C$. In this case, $A$ and $C$ are **not** independent.
$$
\begin{aligned}
P(A,C|B) &= \frac{P(A,B,C)}{P(B)} = \frac{P(A)P(C)P(B|A,C)}{P(B)} \\ \\
&\neq P(A|B)\times P(C|B)
\end{aligned}
$$

A simple illustration of the above could be as follows:

Suppose we flip a fair coin twice. Let the first flip resulting in "heads" be event $A$. Let the second flip resulting in "heads" be event $C$. Let $B$ represent the event that both flips landed on the same side (either "heads-heads" or "tails-tails").

If we _know_ that event $B$ has occurred (i.e. same side), the probability of $P(A, C|B)$ is $\frac{1}{2}$. 

However, if we look at the probability $P(A|B)$ and $P(C|B)$, the fact that we know that the coin landed on the same side does not tell us anything about $A$ or $C$. In other words, $P(A|B) = P(C|B) = \frac{1}{2}\times\frac{1}{2} = \frac{1}{4}$

Hopefully this example serves to show how in the case of  $A \rightarrow B \leftarrow C$, $P(A,C|B) \neq P(A|B)P(C|B)$

### d-Separation

Looking at the conditional independence of non-neighbour nodes in these different cases brings us to the concept of $d$-separation. This concept is important, especially in the analysis of large Bayesian networks.

We will not cover this concept in this course, but suffice it to note that $A$ and $C$ are separated given $B$ only in these cases:

![[Pasted image 20241017192421.png|250]]

For the final example, $A$ and $C$ are only separated if $B$ and all of its dependents are not observed.

