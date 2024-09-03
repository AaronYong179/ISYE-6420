
All final answers will be given to 5 significant figures (5 s.f.)
### Question 1

##### 1a.

Sensitivity:
$$ 0.98 \cdot 0.92 \cdot 0.93 = 0.83845 $$
Specificity:
$$ 1 - [(1 - 0.91) \cdot (1 - 0.88) \cdot (1 - 0.87)] = 0.99860 $$
##### 1b.

Sensitivity:
$$ 1 - [(1-0.98) \cdot (1-0.92) \cdot (1-0.93)] = 0.99989 $$
Specificity:
$$ 0.91 \cdot 0.88 \cdot 0.87 = 0.69670 $$
##### 1c.

Let:
- sensitivity be $Se$, 
- specificity be $Sp$, and 
- prevalence, $Pre = 50/1000$

Then, PPV is defined as:
$$ PPV = \frac{Se \cdot Pre}{(Se \cdot Pre) + (1 - Sp)\cdot (1-Pre)} $$

If the tests are combined in a serial manner, $PPV = 0.96917$. Conversely, if the tests are combined in a parallel manner, $PPV = 0.14785$.

### Question 2

Let $N_i$ represent the event that the $i$-th neuron fires. Let $H_i$ represent the event that the $i$-th neuron received a stimulus. 

Clearly, if the $i-1$-th neuron fires, the $i$-th neuron will receive a stimulus. Therefore, $N_{i-1} = H_i$.

It is given that neuron 1 is given a stimulus (i.e., $H_1$ has occurred).
##### 2a.

Calculating the probabilities of each neuron firing given neuron 1 is stimulated:

$$
\begin{aligned}
P(N_1) &= 0.9 \\ \\
P(N_2) = P(N_3) &= P(N_2|H_1) \cdot P(H_1) + P(N_2|\neg H_1) \cdot P(\neg H_1) \\
&= 0.9 \cdot 0.9 + 0.05 \cdot 0.1 \\
&= 0.815 \\ \\
P(N_4) = P(N_5) &= P(N_4|H_3) \cdot P(H_3) + P(N_4|\neg H_3) \cdot P(\neg H_3) \\
&= 0.9 \cdot 0.815 + 0.05 \cdot 0.185 \\
&= 0.74275 \\ \\
P(H_6) = P(N_5 \cup N_6) &= 1 - [(1 - 0.74275) \cdot (1 - 0.74275)] \\
&= 0.93382 \\ \\
\therefore P(N_6) &= (0.9) \cdot (0.93382) + (0.05) \cdot (1 - 0.93382) \\
&= 0.84375
\end{aligned}
$$

##### 2b.

Given neuron 4 did not fire, $P(H_6) = P(N_5) = 0.74275$ as calculated above.

Therefore, $P(N_6| \neg N_4)$ can be calculated as:
$$ \begin{aligned}
P(N_6|\neg N_4) &= (0.9\cdot0.74275) + (0.05\cdot(1-0.74275)) \\
&= 0.68134
\end{aligned}$$

##### 2c.

By Bayes Theorem,

$$ P(H_5|\neg N_6) = \frac{P(\neg N_6 | H_5)\cdot P(H_5)}{P(\neg N_6)} $$

We know the following from (2a.):
- $P(\neg N_6) = 1 - 0.84375 = 0.15325$ 
- $P(H_5) = P(N_2) = 0.815$

Solving for $P(\neg N_6 | H_5)$,
$$ 
\begin{aligned}
P(N_5 | H_5) &= 0.9 \\ \\
P(H_6 | H_5) &= 1 - [(1 - P(N_5 | H_5)) \cdot (1 - P(N_4 | H_5)] \\ 
&= 1 - [(1-0.9)\cdot (1-0.74275)] \\
&= 0.97428 \\ \\

P(\neg N_6 |H_5) &= 1 - [(0.9 \cdot 0.94728) + (0.05 \cdot (1 - 0.94728))] \\
&= 0.074439
\end{aligned}
$$
Finally, 
$$
\begin{aligned}
P(H_5|N_6) &= \frac{0.074439 \cdot 0.815}{0.15325} \\\\
&= 0.38827
\end{aligned}
$$
##### 2d.

By Bayes Theorem,
$$
P(\neg N_2 \neg N_3 | N_6) = \frac{P(N_6|\neg N_2 \neg N_3) \cdot P(\neg N_2 \neg N_3)}{P(N_6)}
$$

We know the following from (2a.):
- $P(N_6) = 0.84375$
- $P(\neg N_2 \neg N_3) = (1-0.815)^2 = 0.034225$

Given $N_2$ and $N_3$, $P(N_4) = P(N_5) = 0.05$.

Solving for $P(N_6 | \neg N_2 \neg N_3)$ ,

$$\begin{aligned}
P(H_6|\neg N_2 \neg N_3) &= 1 - [(1-0.05)^2] \\
&= 0.0975 \\ \\

\therefore P(N_6 | \neg N_2 \neg N_3) &= (0.9 \cdot 0.0975) + (0.05 \cdot(1-0.0975)) \\
&=0.13288
\end{aligned}$$
Finally,
$$\begin{aligned}
P(\neg N_2 \neg N_3 | N_6) &= \frac{0.13288\cdot0.034225}{0.84375} \\
&= 5.3898 \times 10^{-3}
\end{aligned}$$

### Question 3

##### 3a.

Let $S$ denote the overall success of the machine.

Let $A$ denote the event that the "component with failure probability of 0.5" works as intended. In other words, $P(A) = P(\neg A) = 0.5$.

Let $p$ be the probability that each of the other three components work.

By Total Probability,

$$ P(S) = P(S|A)P(A) + P(S|\neg A)P(\neg A)$$

Given $\neg A$, all other three components must work for $S$ to occur. Therefore,
$$ P(S|\neg A) = p^3$$

Given $A$, at least two other components need to work for $S$ to occur. Calculating the inverse, i.e.
$$\begin{aligned}
P(\neg S|A) &= 3 \cdot ((1-p)^2 \cdot p) + (1-p)^3 \\
&= 2p^3 - 3p^2 + 1
\end{aligned}$$
Therefore, 
$$ P(S|A) = 1 - [2p^3 - 3p^2 + 1] = 3p^2 - 2p^3  $$

Finally, the probability that the overall machine fails is:
$$\begin{aligned}
P(\neg S) &= 1 - P(S) \\
&= 1 - \frac{3p^2-p^3}{2}\\
\end{aligned}$$

It is given that $p=0.75$. 
$$ P(\neg S) = 0.36719 $$
##### 3b.

By Bayes Theorem,

$$ P(\neg A|\neg S) = \frac{P(\neg S|\neg A)\cdot P(\neg A)}{P(\neg S)}$$
From (3a.):
$$ P(\neg S) = 1 - \frac{3p^2-p^3}{2} $$
It is also trivial to show that:
$$\begin{aligned}
P(\neg S|\neg A) &= 1 - p^3 \\ \\
\end{aligned}$$

Therefore, 
$$\begin{aligned}
P(\neg A | \neg S) = \frac{1-p^3}{2 - 3p^2 + p^3}
\end{aligned}$$

Assuming $p=0.75$, we get $P(\neg A | \neg S) = 0.78725$.

##### 3c.

Let the probability that the test returns a positive result be $P(T)$, and the probability that the test returns a negative result be $P(\neg T)$.

Given the following:
- Specificity, $P(\neg T |\neg A) = 0.9$
- Sensitivity, $P(T|A) = 0.8$

By Bayes Formula and Total Probability,

$$\begin{aligned}
P(\neg A| \neg T) &= \frac{P(\neg T|\neg A)\cdot P(\neg A)}{P(\neg T)} \\ \\
&= \frac{P(\neg T|\neg A)\cdot P(\neg A)}{P(\neg T|\neg A)\cdot P(\neg A) + P(\neg T | A)\cdot P(A)} \\ \\
&= \frac{0.9 \cdot 0.5}{(0.9\cdot0.5)+(0.2\cdot0.5)} \\ \\
&= 0.81818
\end{aligned}$$

