#### Glossary of Dynamic Bayesian Networks and N²DBN

##### 1. Notation and conventions

###### Random variable and realization

An uppercase letter such as $X$ denotes a random variable; lowercase $x$ denotes one possible value. Thus $P(X=x)$ is a probability. The symbol $X$ by itself denotes the variable.

###### Time-index notation

$X_t$ is the instance of $X$ at discrete time $t$, and

$$
X_{a:b}=(X_a,X_{a+1},\ldots,X_b).
$$

Here $a$ is the first included time index, $b$ is the last included time index, the colon means “the complete consecutive range,” and $X_{a:b}$ is the ordered tuple containing $b-a+1$ variables when $a\le b$. For example, $X_{2:4}=(X_2,X_3,X_4)$; neither endpoint is omitted. $T$ usually denotes the final time index or sequence length. A document should state whether indexing begins at 0 or 1.

###### Generic DBN notation

Following Murphy, a slice may be partitioned as $Z_t=(U_t,X_t,Y_t)$, where $U_t$ is an input or control, $X_t$ is a state, and $Y_t$ is an output or observation. Each may be a vector or set of nodes. This partition is optional.

###### N²DBN event-log and array notation

The manuscript uses different symbols. For scale $k$, epoch $t$, and event label $\ell$,

$$
X_t^{(k,\ell)}=\text{number of label-}\ell\text{ episodes starting in the epoch},
$$

$$
Y_t^{(k,\ell)}=\text{total duration of those episodes in finest-scale slots}.
$$

The manuscript writes the scale-$k$ temporal partition as

$$
\mathcal T^{(k)}=\{I_t^{(k)}:t=1,\ldots,T_k\},
$$

where $T_k$ is the number of epochs and $I_t^{(k)}$ is the actual time interval occupied by epoch index $t$. With that distinction made explicit, the indices and values mean:

- $k\in\{1,\ldots,K\}$ selects the temporal resolution, with $k=1$ finest and $k=K$ coarsest;
- $t\in\{1,\ldots,T_k\}$ selects one epoch index at that resolution; $I_t^{(k)}\in\mathcal T^{(k)}$ is its interval;
- $\ell\in\mathcal L$ selects the event label;
- $X_t^{(k,\ell)}\in\mathbb N_0$ counts episode **starts**; ongoing occupancy is a separate quantity; and
- $Y_t^{(k,\ell)}\in\mathbb N_0$ sums the full durations of those starts, measured in units of the finest slot even when $k>1$.

Both are observed non-negative-integer variables. Here $X$ denotes observed counts; the hidden-state symbols are $U$, $S$, and $H$. Write $Z^{(k)}=(X^{(k)},Y^{(k)})$ for the complete count–duration array at scale $k$ and $C_t^{(k)}$ for exogenous context. The learning section writes the same context vector as $\mathcal K_t^{(k)}$; throughout this glossary,

$$
\mathcal K_t^{(k)}\equiv C_t^{(k)},
$$

so both symbols denote the same context vector. The event log is

$$
\mathcal D=\{(\ell_i,s_i,d_i)\}_{i=1}^N,
$$

where $\mathcal D$ denotes the dataset, braces denote the collection of recorded episodes, $i$ indexes episodes, and $N$ is the total number of episodes. The time-slot index is $t$. Within tuple $i$, $\ell_i$ is the label, $s_i$ is the start time, and $d_i\in\mathbb N$ is the duration in finest-slot units. Projecting these tuples into epochs produces $X^{(k)}$ and $Y^{(k)}$; the number of tuples is therefore $N=\sum_t\sum_\ell X_t^{(1,\ell)}$ under atomic finest-scale starts.

###### N²DBN latent-state notation

At scale $k$ and epoch $t$,

$$
U_t^{(k)}=(S_t^{(k)},H_t^{(k)}),
$$

Here $U_t^{(k)}$ is the complete latent state at one scale and epoch. Its first component $S_t^{(k)}\in\{1,\ldots,M_k\}$ is a discrete **regime**, where $M_k$ is the number of possible regimes at scale $k$. Its second component $H_t^{(k)}\in\mathbb R^{r_k}$ is a continuous factor vector, where $r_k$ is its dimension and each coordinate is real-valued. $S$ represents persistent, discrete drift; $H$ represents low-dimensional same-epoch co-movement. Both components are unobserved and inferred from the arrays and context. The manuscript occasionally spells *regime* as *regimen*; the statistical term intended by the model is *regime*.

###### N²DBN typed edges and implemented parent sets

Let $\mathcal Q=\{\mathrm{cnt},\mathrm{dur}\}$ denote the count and duration channels. A learned edge at scale $k$ is

$$
e=((a,q_s),(b,q_t),d), \qquad d\in\{1,\ldots,p_k\},
$$

In this tuple:

- $a,b\in\mathcal L$ are the source and target labels;
- $q_s,q_t\in\mathcal Q$ identify the source and target quantity channels;
- $q=\mathrm{cnt}$ means the $X$ channel and $q=\mathrm{dur}$ means the $Y$ channel;
- $d$ is a strictly positive lag measured in scale-$k$ epochs; and
- $p_k$ is the largest admissible lag at scale $k$.

Thus $e$ says that the value of channel $q_s$ for label $a$ at epoch $t-d$ enters the conditional predictor of channel $q_t$ for label $b$ at epoch $t$. For example, $((a,\mathrm{cnt}),(b,\mathrm{dur}),2)$ is a two-epoch lag from $a$'s count to $b$'s duration. In the compact model, the learned graph $\mathcal G=\{\mathcal G^{(k)}\}_{k=1}^K$ contains these resolution-specific typed edges. The learned-edge count excludes aggregation links, latent-state links, and other construction-fixed links.

The current learning algorithm represents the learned structure by target-specific parent sets

$$
\mathcal G=\{\mathcal P_{k,b}:k=1,\ldots,K,\ b\in\mathcal L\},
\qquad
\mathcal P_{k,b}\subseteq\mathcal C_{k,b}\subseteq
\mathcal L\times\{1,\ldots,p_k\}.
$$

Here $\mathcal C_{k,b}$ is the candidate set for target label $b$, and a selected parent $(a,d)\in\mathcal P_{k,b}$ activates a count coefficient $\beta_{a,d}^{(k,b)}$ and a duration coefficient $\alpha_{a,d}^{(k,b)}$. Thus the implemented unit of selection is one grouped label–lag relation. The current learning equations contain no independent selection for each source-channel/target-channel pairing. An in-degree constraint is written $|\mathcal P_{k,b}|\le D_k$.

**Specification note:** `sec_model.tex` declares the channel-typed edge. `sec_learning.tex` currently searches the grouped parent $(a,d)$ described above. `sec_benchmark.tex` likewise scores $(a,d,b)$ triples and omits channel types. Current fitted and evaluated graphs should therefore be reported using $\mathcal P_{k,b}$ and interpreted as grouped label–lag relations unless the learning and benchmark sections are extended to select all four channel pairings independently.

###### Parent notation

$\operatorname{Pa}(X)$ or $pa(X)$ denotes the parents of node $X$ in the graph under discussion.

###### Sets, indices, and number systems

The following conventions recur throughout the equations:

- $x\in\mathcal X$ means that value $x$ belongs to set $\mathcal X$.
- $\{x_1,\ldots,x_n\}$ denotes a set or collection; $(x_1,\ldots,x_n)$ denotes an ordered tuple or vector.
- $\mathbb N$ denotes positive integers, $\mathbb N_0=\{0,1,2,\ldots\}$ non-negative integers, and $\mathbb R$ real numbers.
- $\mathbb R^r$ is the space of real vectors with $r$ coordinates; $r$ is its dimension.
- $|\mathcal X|$ is the cardinality, or number of elements, of a finite set. For a scalar $x$, $|x|$ means absolute value; context distinguishes the two meanings.
- A subscript usually selects time, sample, or vector position. A superscript in parentheses, such as $(k,\ell)$, is an index label. Exponentiation uses a numerical or algebraic exponent without this index convention.
- $i$, $j$, $s$, and $t$ are commonly dummy indices. Consistent renaming of a dummy index preserves a sum.
- $1{:}T$ and $a{:}b$ denote inclusive consecutive index ranges.
- $\ell$ is the lowercase Greek letter *ell* used for an event label. The numeral one is written $1$.

###### Common mathematical operators

- $\sum_i a_i$ adds terms over the stated index range; $\prod_i a_i$ multiplies them.
- $\int f(x)\,dx$ accumulates a density or function over a continuous domain; the domain is the full support unless bounds are displayed or stated.
- $\exp(x)=e^x$ maps real values to positive values; $\log(x)$ is the natural logarithm unless another base is stated.
- $\mathbb E_q[f(X)]$ is the expectation of $f(X)$ when $X$ follows distribution $q$.
- $\max_x f(x)$ is the largest attained function value. The expression $\arg\max_x f(x)$ gives the value of $x$ at which the maximum is attained. There can be multiple maximizers.
- $=$ denotes equality, $\approx$ approximation, $\propto$ equality up to a positive multiplicative constant, and $\le$ an inequality.
- $\mathbb I\{A\}$ is an indicator: it equals one when statement $A$ is true and zero otherwise.
- $I$ commonly denotes an identity matrix or operator when it appears in linear algebra. In the N²DBN notation, $I_t^{(k)}$ denotes an epoch interval.

###### Decorations on symbols

Decorations convey different kinds of status:

- $\widehat\theta$ is an estimate computed from data.
- $\theta^*$ or $q^*$ denotes a selected optimum or, when explicitly stated, a ground-truth value; the surrounding definition must disambiguate it.
- $\bar x$ usually denotes a mean, centered reference, complement message, or aligned quantity, as defined locally.
- $\widetilde w$ denotes a transformed or normalized version of $w$; in particle filtering it denotes a normalized weight.
- $X'$ is a transformed quantity or proposal. A derivative requires an explicit local definition.
- Bold symbols often denote vectors or matrices. The manuscript also uses plain uppercase letters for vector-valued states; dimensions should therefore be read from the local definition.

###### Probability mass, density, and conditioning notation

$P(A)$ denotes probability of an event. For discrete variables, $P(X=x)$ is probability mass; for continuous variables, a lowercase $p(x)$ normally denotes a density and $P(X=x)=0$ at any single exact value. Integrals of a density over regions give probabilities. Individual density heights can exceed one. In expressions such as $P(A\mid B,C)$, the vertical bar means “conditional on” and the comma means both $B$ and $C$ are included in the conditioning information. A semicolon, as in $L(\theta;D)$ or $W(x;\theta)$, conventionally separates a function's variable arguments from fixed data or parameters. It has no conditional-probability meaning.

##### 2. Bayesian-network foundations

###### Bayes' theorem

Bayes' theorem reverses a conditional probability:

$$
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)},
\qquad P(B)>0.
$$

Here $A$ is a hypothesis or unknown event, $B$ is observed evidence, the vertical bar means “conditional on,” and $P(B)>0$ is required because division by zero is undefined. The numerator is the joint probability $P(A,B)=P(B\mid A)P(A)$; division by $P(B)$ restricts and renormalizes that joint mass to cases where $B$ occurred. The four terms have distinct roles:

| Term | Name | Meaning after observing $B$ |
|---|---|---|
| $P(A)$ | Prior | Belief about $A$ before using $B$ |
| $P(B\mid A)$ | Likelihood | How compatible the observation $B$ is with $A$ |
| $P(B)$ | Evidence or marginal probability | Overall probability of observing $B$ under all possible cases |
| $P(A\mid B)$ | Posterior | Updated belief about $A$ after using $B$ |

The same identity is often written

$$
\text{posterior}
=\frac{\text{likelihood}\times\text{prior}}{\text{evidence}}
\propto \text{likelihood}\times\text{prior}.
$$

The symbol $\propto$ means “is proportional to.” Equality requires the omitted constant $1/P(B)$. The proportional form is useful when comparing hypotheses because the evidence is the same normalizing constant for all values of $A$; it must be restored when normalized posterior probabilities are required.

###### Marginalization and the law of total probability

Marginalization removes an unobserved variable by summing over its possible values or integrating over a continuous domain. If $A$ and its complement $\bar A$ exhaust the possibilities, then

$$
P(B)=P(B\mid A)P(A)+P(B\mid\bar A)P(\bar A).
$$

Here $\bar A$ is the complement of $A$, so $P(\bar A)=1-P(A)$. The first product is the probability that $A$ holds **and** $B$ is observed; the second is the probability that $\bar A$ holds and $B$ is observed. These mutually exclusive cases are added to obtain the total probability of $B$. More generally,

$$
P(B)=\sum_a P(B\mid A=a)P(A=a)
$$

for discrete $A$. The lowercase $a$ ranges over every mutually exclusive value in the support of random variable $A$, $P(A=a)$ weights that case, and $\sum_a$ adds the weighted conditional probabilities. No value with positive prior mass may be silently omitted. For continuous $A$,

$$
p(B)=\int p(B\mid a)p(a)\,da
$$

where $p$ denotes a probability density, $a$ is the integration variable, $p(a)\,da$ is the infinitesimal prior mass near $a$, and the integral covers the full support of $A$. Thus $P(B)$ or $p(B)$ in Bayes' theorem is the **result** of marginalization; *marginalization* is the summing or integration operation used to obtain it.

###### Simple Bayes example: a positive test

Suppose $A$ means that a person has a condition and $B$ means that their test is positive. Assume:

- prior prevalence: $P(A)=0.01$;
- sensitivity: $P(B\mid A)=0.95$; and
- false-positive rate: $P(B\mid\bar A)=0.05$.

First marginalize over the two ways to obtain a positive result:

$$
\begin{aligned}
P(B)
&=P(B\mid A)P(A)+P(B\mid\bar A)P(\bar A)\\
&=(0.95)(0.01)+(0.05)(0.99)\\
&=0.059.
\end{aligned}
$$

The first product, $0.95\times0.01=0.0095$, is the probability of having the condition and testing positive. The second, $0.05\times0.99=0.0495$, is the probability of being healthy and testing positive. Their sum $0.059$ is the marginal probability of any positive result. Bayes' theorem then gives

$$
P(A\mid B)=\frac{(0.95)(0.01)}{0.059}\approx0.161.
$$

The numerator $0.0095$ is the joint probability of a true positive; the denominator $0.059$ is the probability of every kind of positive. Their ratio is therefore the fraction of positive results that are true positives. The posterior probability of the condition after a positive test is about $16.1\%$. The value $95\%$ is the sensitivity $P(B\mid A)$. The requested posterior is the reversed conditional $P(A\mid B)$. The difference occurs because the condition is rare and false positives among the much larger healthy group contribute substantially to all positive results.

Equivalently, among 10,000 comparable people, about 100 have the condition and 95 of them test positive; among 9,900 without it, about 495 test positive. Of the $95+495=590$ positive tests, only 95 correspond to the condition, giving $95/590\approx16.1\%$.

###### Bayes' theorem for parameters and data

In statistical learning, the hypothesis is usually a parameter $\theta$ or graph $G$, and the evidence is a dataset $D$:

$$
p(\theta\mid D)
=\frac{p(D\mid\theta)p(\theta)}{p(D)},
\qquad
p(D)=\int p(D\mid\theta)p(\theta)\,d\theta.
$$

In this parameter form:

- $\theta$ denotes the unknown model parameter or parameter vector;
- $D$ denotes the fixed observed dataset;
- $p(\theta)$ is the prior density over parameter values;
- $p(D\mid\theta)$ is the likelihood of the observed dataset at a proposed parameter value;
- $p(D)$ is the marginal likelihood or model evidence obtained by averaging that likelihood over the prior; and
- $p(\theta\mid D)$ is the normalized posterior density.

The differential $d\theta$ indicates integration over the full parameter space and may represent a multidimensional integral. In a discrete model comparison, the integral becomes a sum over hypotheses.

The likelihood is viewed as a function of the unknown hypothesis or parameter with the observed data held fixed. Its integral over $\theta$ can differ from one. Multiplying it by the prior and dividing by the evidence produces the normalized posterior.

###### Bayesian Network, BN

A BN consists of a directed acyclic graph (DAG) and local conditional probability distributions (CPDs). For nodes $Z^1,\ldots,Z^N$,

$$
P(Z^1,\ldots,Z^N)=\prod_{i=1}^{N}P\!\left(Z^i\mid\operatorname{Pa}(Z^i)\right).
$$

Here $N$ is the number of nodes, $Z^i$ is node $i$, $\operatorname{Pa}(Z^i)$ is the collection of its graph parents, and $P(Z^i\mid\operatorname{Pa}(Z^i))$ is its local CPD. The product symbol $\prod_{i=1}^N$ multiplies one local conditional term per node. Root nodes have an empty parent set, so their term is simply $P(Z^i)$. The equality states the BN factorization property: the graph's local CPDs reconstruct the global joint distribution. Marginal independence requires a separate graphical or probabilistic condition. A causal interpretation of an arrow also requires additional assumptions.

###### Directed Acyclic Graph, DAG

A DAG is a directed graph with no directed cycle. Every finite unrolling of a standard DBN must be a DAG. Forward-pointing inter-slice edges preserve temporal acyclicity. Intra-slice edges must also form an acyclic graph.

###### Node or vertex

A node represents a random variable. An observed value is a realization assigned to that node. In an unrolled DBN, $X_t$ and $X_{t+1}$ are distinct nodes instantiated from the same variable template.

###### Edge or arc

$A\rightarrow B$ makes $A$ a parent of $B$ and allows $B$'s CPD to depend directly on $A$. In a probabilistic BN this is a local model relation; it denotes a causal effect only under additional causal assumptions.

###### Parent, child, ancestor, and descendant

If $A\rightarrow B$, $A$ is a parent of $B$ and $B$ is a child of $A$. An ancestor has a directed path to a node; a descendant is reachable from it by a directed path.

###### Conditional Probability Distribution, CPD

A CPD specifies $P(X\mid\operatorname{Pa}(X))$. It may be a discrete table, conditional density, regression, deterministic function, neural parameterization, or another probability kernel.

###### Conditional Probability Table, CPT

A CPT is a tabular CPD for discrete nodes and discrete parents. CPD is the general term and also covers continuous or hybrid models.

###### Deterministic node or CPD

A deterministic node is an exact function of its parents, such as $C=A+B$. Its CPD puts all mass on that function's value. N²DBN's coarser aggregates are deterministic quantities derived from finer-scale variables.

###### Conditional independence

$A$ and $B$ are conditionally independent given $C$, written $A\perp B\mid C$, when learning $B$ adds no information about $A$ after $C$ is known. Where defined,

$$
P(A\mid B,C)=P(A\mid C).
$$

In this equality, commas inside the conditioning set mean that both variables are known: $P(A\mid B,C)=P(A\mid B\cap C)$ in event notation. The equality must hold for all relevant values with positive conditioning probability. After $C$ is known, adding $B$ leaves the conditional distribution of $A$ unchanged. Marginal independence of $A$ and $B$ is a separate condition.

###### d-separation

d-separation is the graphical criterion for reading conditional independence from a DAG. If $A$ and $B$ are d-separated by $C$, every distribution that factorizes according to the DAG satisfies $A\perp B\mid C$.

###### Collider and v-structure

In $A\rightarrow B\leftarrow C$, $B$ is a collider. If no edge joins $A$ and $C$, this is an unshielded collider or v-structure. Conditioning on a collider, or one of its descendants, can open an otherwise blocked path.

###### Markov blanket

A node's Markov blanket consists of its parents, its children, and its children's other parents. Given this set, the node is conditionally independent of all remaining nodes.

###### Markov equivalence class, MEC

Two DAGs are Markov equivalent if they encode the same d-separation relations. This holds exactly when they have the same skeleton and v-structures. Observational data generally identifies an equivalence class. Identification of a unique causal DAG requires additional information or assumptions.

###### Completed Partially Directed Acyclic Graph, CPDAG

A CPDAG represents an MEC. Directed edges have the same orientation in every DAG in the class; undirected edges can be oriented differently. Structure evaluation should account for these observationally indistinguishable orientations.

###### Faithfulness

A distribution is faithful to a graph if all and only its conditional independences are implied by d-separation. Constraint-based learning commonly requires faithfulness or a weaker variant.

###### Causal sufficiency

Causal sufficiency means there is no unobserved common cause of two or more included variables. Without it, learned edges can reflect latent confounding. Causal sufficiency, faithfulness, temporal ordering, and suitable data are among the assumptions needed for causal discovery.

##### 3. Dynamic Bayesian networks

###### Dynamic Bayesian Network, DBN

A DBN is a directed probabilistic graphical model for a stochastic process. In Murphy's standard first-order, time-homogeneous formulation it is a pair $(B_1,B_{\rightarrow})$: $B_1$ defines $P(Z_1)$ and the two-slice template $B_{\rightarrow}$ defines $P(Z_t\mid Z_{t-1})$ for $t>1$.

*Dynamic* means that the modeled system evolves. Time variation in the graph or parameters is an additional model property.

###### Time slice

A time slice is the set $Z_t=\{Z_t^1,\ldots,Z_t^N\}$ associated with one discrete time point or interval. Its wall-clock duration is the model's time granularity.

###### Initial Bayes net, $B_1$

$B_1$ defines the initial distribution $P(Z_1)$, including dependencies within the first slice. Its structure and parameters may differ from the within-slice portion of later slices.

###### Two-Slice Temporal Bayes Net, 2TBN

A 2TBN defines the transition from slice $t-1$ to $t$:

$$
P(Z_t\mid Z_{t-1})=
\prod_{i=1}^{N}P\!\left(Z_t^i\mid\operatorname{Pa}(Z_t^i)\right).
$$

Here $Z_t=(Z_t^1,\ldots,Z_t^N)$ is the complete current slice, $Z_{t-1}$ is the preceding slice, and $Z_t^i$ is current-slice node $i$. The product contains one CPD for each of the $N$ current-slice nodes. $\operatorname{Pa}(Z_t^i)$ may contain nodes from $t-1$ and earlier-ordered nodes from $t$; any same-slice parents must be ordered acyclically. The left side is one conditional distribution over the whole current slice. The right side factorizes that distribution into local conditionals. First-slice nodes in the template are conditioning variables and receive no CPD from the transition template.

The familiar expression

$$
P(X_{1:T},Y_{1:T})=P(X_1)
\prod_{t=2}^{T}P(X_t\mid X_{t-1})
\prod_{t=1}^{T}P(Y_t\mid X_t)
$$

has three parts:

- $P(X_1)$ initializes the latent state at the first time point;
- $\prod_{t=2}^T P(X_t\mid X_{t-1})$ multiplies the state-transition probability for every later time; and
- $\prod_{t=1}^T P(Y_t\mid X_t)$ multiplies one observation or emission probability at every time.

$X_{1:T}$ denotes the entire state trajectory and $Y_{1:T}$ the entire observation trajectory. The factorization assumes first-order state dynamics and that each $Y_t$ is conditionally independent of the other variables given $X_t$. It is a common state-observation special case. General DBNs allow other factorizations.

###### Unrolling

Unrolling instantiates the template over a finite horizon. For a homogeneous DBN,

$$
P(Z_{1:T})=P(Z_1)
\prod_{t=2}^{T}\prod_{i=1}^{N}
P\!\left(Z_t^i\mid\operatorname{Pa}(Z_t^i)\right).
$$

$Z_{1:T}$ is the collection of all variables in all $T$ slices. $P(Z_1)$ is the joint distribution of the initial slice. The outer product iterates over times $2,\ldots,T$; the inner product iterates over the $N$ variables in each new slice. Each parent set refers to the appropriate instantiated nodes in the unrolled graph. The same template and parameters may be reused over time. The result is an ordinary, usually large, BN whose acyclicity can be checked on the instantiated nodes.

###### Intra-slice or contemporaneous edge

$A_t\rightarrow B_t$ joins nodes in one slice and represents a contemporaneous dependency at the chosen resolution. The sampling resolution prevents a conclusion about literal physical simultaneity.

###### Inter-slice or temporal edge

$A_{t-r}\rightarrow B_t$ is a lag-$r$ dependency from an earlier slice. Standard first-order 2TBNs use $r=1$; higher-order DBNs can use longer lags.

###### Persistent node and forward interface

Murphy calls a variable persistent when its instance has a self-transition edge $Z_{t-1}^i\rightarrow Z_t^i$. The forward interface is broader: it contains every node in slice $t$ with any child in $t+1$. The interface d-separates the earlier past from the future and supports efficient exact online inference.

###### State variable

A state variable is part of a representation sufficient to predict the future, given future inputs. A state can be factored into several variables. State variables are often latent. A state can also be observed, and a latent variable may serve another role.

###### Observation or output variable

An observation variable represents a measurement or output. $P(Y_t\mid X_t)$ is common. A general DBN may also give it inputs, earlier observations, or other same-slice parents.

###### Hidden or latent variable

A latent variable is modeled and inferred from observed variables. It has no direct observation. DBNs can be fully observed; such DBNs are common in structure learning.

###### Input, control, or exogenous variable

An input $U_t$ is supplied from outside the modeled state dynamics. A controlled model can use $P(X_t\mid X_{t-1},U_t)$. Exogeneity is a modeling assumption that requires substantive justification.

###### Static or time-invariant covariate

A static covariate, such as a participant attribute, has one value across the sequence. It can be represented once and connected to multiple slices. A dynamic variable has a value at each time step, even when its CPD is time-homogeneous.

###### Factored state representation

A DBN may represent $X_t=(X_t^1,\ldots,X_t^n)$ and factor the transition according to the graph. An HMM uses one monolithic state $S_t$. This factored representation is the principal sense in which DBNs generalize HMMs.

##### 4. Temporal assumptions and models

###### Markov assumption and Markov order

An $m$th-order Markov process assumes

$$
P(Z_t\mid Z_{1:t-1})=P(Z_t\mid Z_{t-m:t-1}).
$$

The left side conditions on the entire past $Z_1,\ldots,Z_{t-1}$. The right side retains only the most recent $m$ slices, $Z_{t-m},\ldots,Z_{t-1}$. Equality means that older history provides no additional predictive information once those $m$ slices are known. $m$ is the Markov order; the common first-order case has $m=1$. Near the beginning of a sequence, only the available history is used. A higher-order process can be made first-order by augmenting its state, potentially at substantial cost.

###### Lag and lag order

The lag of $A_{t-r}\rightarrow B_t$ is $r$. Maximum permitted lag is sometimes called lag order. Lag is counted in slices, so its wall-clock meaning changes with resolution.

###### Time-homogeneous DBN

A DBN is time-homogeneous when the same transition structure and CPDs are reused at all applicable steps. *Homogeneous* or *time-invariant* is clearer than *stationary* for this template-level property.

###### Stationarity

DBN papers sometimes use *stationary* to mean time-homogeneous. Strict statistical stationarity is stronger: joint distributions must be invariant under shifts of all time indices. A homogeneous transition kernel alone is insufficient; its initial distribution must also be invariant. This glossary distinguishes these meanings unless reporting a source's usage.

###### Initial, transition, and observation models

The initial distribution supplies the sequence boundary condition. The transition model describes evolution between slices. The observation or emission model describes measurements from state. In a simple state-space model these are $P(X_1)$, $P(X_t\mid X_{t-1})$, and $P(Y_t\mid X_t)$. A general 2TBN permits a less restricted graph.

###### Trajectory

A trajectory is a time-ordered realization such as $x_{1:T}$. A dataset can contain one long trajectory or multiple independent or conditionally independent trajectories. Time points within one trajectory usually depend on earlier time points.

###### Hidden Markov Model, HMM

An HMM is a DBN with a single discrete, usually latent, state per slice, a first-order transition, and conditionally independent emissions. A factored DBN represents the state with multiple interacting variables.

###### Linear Dynamical System and Kalman filter

A linear dynamical system (called a Kalman filter model by Murphy) uses linear-Gaussian transitions and observations:

$$
X_t=AX_{t-1}+\epsilon_t,\qquad Y_t=CX_t+\eta_t.
$$

In the transition equation, $X_{t-1}$ and $X_t$ are the previous and current state vectors, $A$ is the state-transition matrix, and $\epsilon_t$ is process noise, normally assumed Gaussian with zero mean and covariance $Q$. In the observation equation, $Y_t$ is the measurement vector, $C$ is the observation or loading matrix, and $\eta_t$ is Gaussian measurement noise with covariance $R$. Dimensions must conform: if $X_t\in\mathbb R^r$ and $Y_t\in\mathbb R^p$, then $A$ is $r\times r$ and $C$ is $p\times r$. Initial-state mean and covariance complete the model. Under these linear-Gaussian assumptions, the Kalman filter gives exact filtering. Other state-space models can require other algorithms.

###### State-Space Model, SSM

An SSM combines an initial state distribution, transition, and observation model. DBNs are a graphical language for structured SSMs with discrete, continuous, or hybrid variables and general CPDs.

###### Hierarchical Hidden Markov Model, HHMM

An HHMM organizes latent states hierarchically so high-level states invoke lower-level submodels. Murphy represents HHMMs as DBNs. This state hierarchy is unrelated to N²DBN's hierarchy of temporal aggregation levels.

###### Continuous-Time Bayesian Network, CTBN

A CTBN uses conditional intensity matrices to model finite-state variables in continuous time. Standard DBNs use fixed-width slices. The supplied time-granularity paper adapts **inference resolution** within a CTBN. Its construction contains no deterministic aggregates across calendar scales.

###### Vector Autoregression, VAR

A VAR models each component of a multivariate series as a linear function of lagged components. Sparse coefficient patterns define lagged-dependence networks. VAR-based dynamic networks overlap with linear-Gaussian DBNs and emphasize regression structure.

###### Granger causality

$A$ Granger-causes $B$ relative to an information set if $A$'s past improves prediction of $B$. This definition concerns prediction. A structural causal interpretation requires assumptions about omitted variables, sampling resolution, and the data-generating process.

###### Copula

A copula couples marginal distributions into a joint distribution and can represent nonlinear dependence separately from marginals. The supplied Copula-DBN paper uses copulas to build conditional runoff-state probabilities. Copulas are optional CPD tools. DBN and N²DBN definitions allow other CPD families.

##### 5. Inference queries and algorithms

###### Evidence and marginalization

Evidence is an observed assignment, such as $Y_{1:t}=y_{1:t}$. Marginalization removes variables by summing or integrating, e.g. $P(A)=\sum_bP(A,b)$. Temporal aggregation combines values across time and is a different operation.

###### Belief state

The filtering belief state is $\operatorname{Bel}_t(X_t)=P(X_t\mid y_{1:t})$. If the state is sufficient, this summarizes the past evidence needed for future prediction.

###### Filtering

Filtering estimates the current state from evidence through the current time: $P(X_t\mid y_{1:t})$.

###### Prediction

Prediction estimates a future state or observation, such as $P(X_{t+h}\mid y_{1:t})$ or $P(Y_{t+h}\mid y_{1:t})$ for $h>0$.

###### Smoothing

Fixed-interval smoothing computes $P(X_t\mid y_{1:T})$ over a completed sequence. Fixed-lag smoothing computes $P(X_{t-\ell}\mid y_{1:t})$ for a chosen lag $\ell$.

###### Online, offline, and roll-up

Online inference processes observations as they arrive; offline inference can use the full sequence. Roll-up summarizes past evidence in the current belief or interface distribution and discards earlier slices.

###### Most Probable Explanation, MPE

MPE jointly assigns **all** non-evidence variables: $z^*=\arg\max_zP(z\mid e)$. Separate maximization of each marginal can give a different assignment.

###### Maximum a posteriori query, MAP

A MAP query maximizes selected variables $Q$ after marginalizing nuisance variables $H$:

$$
q^*=\arg\max_q\sum_hP(q,h\mid e).
$$

Here $Q$ is the set of variables whose value is requested, $q$ is one joint assignment to $Q$, $H$ is the nuisance-variable set, $h$ ranges over all joint assignments to $H$, and $e$ is the observed evidence. The sum marginalizes $H$, and $P(q,h\mid e)$ is the posterior probability of the combined assignment. The operator $\arg\max_q$ returns the assignment; $q^*$ denotes the winning assignment. The corresponding probability is the maximized function value. For continuous nuisance variables the sum becomes an integral. MPE is the special case where $Q$ contains every non-evidence variable, so no nuisance variables remain to marginalize.

###### Viterbi and posterior decoding

Viterbi decoding finds the joint latent path $x_{1:T}^*=\arg\max_{x_{1:T}}P(x_{1:T}\mid y_{1:T})$. It is MPE if that path contains all unobserved variables. Posterior decoding chooses each state from its marginal $P(X_t\mid y_{1:T})$ and can produce a different path.

###### Exact inference, junction tree, and treewidth

Exact inference returns model-implied probabilities up to numerical precision. The junction-tree algorithm performs exact cluster message passing after triangulation. Its cost is exponential in treewidth. Treewidth equals the largest cluster size in an optimal tree decomposition minus one. Graph connectivity and elimination order determine this cost in addition to node count.

###### Forward and forward-backward algorithms

For a discrete first-order model,

$$
P(X_t\mid y_{1:t})\propto P(y_t\mid X_t)
\sum_{x_{t-1}}P(X_t\mid x_{t-1})P(x_{t-1}\mid y_{1:t-1}).
$$

The terms perform two operations:

- **Prediction:** $\sum_{x_{t-1}}P(X_t\mid x_{t-1})P(x_{t-1}\mid y_{1:t-1})$ averages the transition model over the previous filtering distribution to obtain the prior predictive distribution of $X_t$.
- **Correction:** multiplying by $P(y_t\mid X_t)$ weights each current state by how well it explains the new observation $y_t$.

Lowercase $x_{t-1}$ is the value being summed out. Uppercase $X_t$ remains the variable whose posterior is computed. The symbol $\propto$ means the right side must be divided by its sum over $X_t$ so that probabilities total one. $y_{1:t}$ denotes all observations through time $t$. Continuous-state models replace the sum by an integral. Forward-backward combines these forward messages with future-to-past messages for smoothing.

###### Interface algorithm

The interface algorithm applies junction-tree reasoning slice by slice and passes a distribution over interface nodes. Murphy presents it as a simple exact approach to online DBN inference.

###### Approximate inference

Approximate inference trades exactness for computation or memory. Approximation error, Monte Carlo error, and numerical optimization error are distinct.

###### Factored Frontier, FF

FF uses a product of node marginals, $P(X_t\mid y_{1:t})\approx\prod_iq_t^i(X_t^i)$. It is a deterministic approximation.

###### Boyen–Koller approximation, BK

BK projects the belief onto distributions over chosen clusters: $P(X_t\mid y_{1:t})\approx\prod_cq_t^c(X_t^c)$. Cluster size controls cost and fidelity; FF is the singleton-cluster extreme.

###### Belief propagation and Loopy Belief Propagation, LBP

Belief propagation is exact on suitable trees. Applying its updates to a graph with loops gives LBP. LBP is approximate and can fail to converge.

###### Monte Carlo method

A Monte Carlo method approximates a probability or expectation with random samples, for example

$$
\mathbb E[f(X)]\approx\frac{1}{N}\sum_{i=1}^{N}f(x^{(i)}).
$$

$\mathbb E[f(X)]$ is the expectation of a function $f$ under the target distribution of random variable $X$. $x^{(i)}$ is sample $i$; the superscript is a sample index. An exponent would denote a power. $N$ is the number of samples, and $1/N$ makes the sum an arithmetic average. The approximation improves under suitable laws of large numbers; its random error is commonly of order $N^{-1/2}$ for independent finite-variance samples. Some Monte Carlo methods use dependent samples. Dependence reduces the effective information relative to $N$ independent draws.

###### Particle and importance weight

A particle is a sampled hypothesis about the current latent state or its trajectory. Its importance weight measures its contribution to the target distribution relative to the proposal that generated it. Normalized weights satisfy $\sum_i\widetilde w_t^{(i)}=1$.

###### Sequential Monte Carlo and particle filtering

SMC represents an evolving posterior with weighted samples. Particle filtering gives

$$
P(X_t\mid y_{1:t})\approx
\sum_{i=1}^{N}\widetilde w_t^{(i)}\delta_{x_t^{(i)}}(X_t).
$$

Here $x_t^{(i)}$ is particle $i$'s current state, $\widetilde w_t^{(i)}\ge0$ is its normalized importance weight, and $\sum_i\widetilde w_t^{(i)}=1$. $\delta_{x_t^{(i)}}(X_t)$ is a Dirac point mass located at that particle. It is a discrete measure with all mass at one point. The weighted sum is therefore an empirical discrete approximation to the filtering distribution: the approximate probability of a region is the total weight of particles inside it. The simple unnormalized update $w_t^{(i)}\propto P(y_t\mid x_t^{(i)})$ assumes a bootstrap transition proposal only after the previous weight is also carried forward; a general proposal requires a target-to-proposal correction.

###### Effective sample size, resampling, and degeneracy

A common diagnostic is $N_{\mathrm{eff}}=1/\sum_i(\widetilde w_t^{(i)})^2$. Resampling replaces low-weight particles with draws from the weighted empirical distribution. It combats weight degeneracy and can cause sample impoverishment.

###### Rao–Blackwellized Particle Filtering, RBPF

RBPF samples some state components and analytically integrates tractable conditional components. This reduces Monte Carlo variance when the DBN contains suitable substructure.

###### Markov chain

A Markov chain has the property that its next state is conditionally independent of earlier states given its current state. In MCMC, the chain is an artificial computational process designed to have the desired posterior as its invariant distribution. Its transitions can differ completely from the modeled DBN dynamics.

###### Markov Chain Monte Carlo, MCMC

MCMC constructs a Markov chain whose invariant distribution is a posterior such as $P(X\mid Y)$ or $P(G\mid D)$. Burn-in, mixing, and convergence diagnostics concern this computational chain. Stationarity of the modeled temporal process is a separate property.

###### Gibbs sampling

Gibbs sampling is MCMC that updates a variable or block from its full conditional distribution, such as $X_i\sim P(X_i\mid X_{-i},Y)$. It avoids an accept/reject step when the conditional can be sampled exactly.

###### Metropolis–Hastings

Metropolis–Hastings proposes $x'\sim q(x'\mid x)$ and accepts it with probability

$$
\alpha=\min\!\left(1,\frac{\pi(x')q(x\mid x')}{\pi(x)q(x'\mid x)}\right).
$$

$x$ is the current state of the computational chain, $x'$ is the proposed state, $q(x'\mid x)$ is the forward proposal density, and $q(x\mid x')$ is the probability of proposing the reverse move. $\pi$ is the target density, possibly known only up to a normalizing constant, which cancels in the ratio. $\alpha\in[0,1]$ is the acceptance probability: accept $x'$ with probability $\alpha$, otherwise repeat $x$. For a symmetric proposal, $q(x'\mid x)=q(x\mid x')$, the proposal terms cancel and only the target-density ratio remains.

###### Burn-in, mixing, and convergence

Burn-in is an initial portion often discarded to reduce initialization effects. Mixing describes how effectively the chain explores the target. Evidence of convergence requires diagnostics and, where practical, multiple chains; discarding burn-in alone is insufficient.

##### 6. Parameter and structure learning

###### Parameter or weight learning

Parameter learning estimates CPDs for a fixed graph. Parameters can be probabilities, regression coefficients, variances, or neural-network weights.

###### Likelihood, MLE, prior, and posterior

The likelihood and log-likelihood are

$$
L(\theta;D)=P(D\mid\theta,G),\qquad \ell(\theta;D)=\log P(D\mid\theta,G).
$$

Here $D$ is the observed dataset, $G$ is the graph treated as fixed in this parameter-learning step, and $\theta$ is the unknown CPD-parameter vector. The semicolon in $L(\theta;D)$ emphasizes that the data are held fixed and the likelihood is viewed as a function of $\theta$. Conditional probability uses a vertical bar. $L$ is the likelihood and $\ell$ its natural logarithm. The logarithm changes products of data contributions into sums and preserves the maximizing $\theta$ because $\log$ is strictly increasing. MLE chooses $\widehat\theta_{\mathrm{MLE}}=\arg\max_\theta L(\theta;D)$. A MAP parameter estimate additionally uses the prior $P(\theta\mid G)$. Bayes' rule gives

$$
P(\theta,G\mid D)\propto P(D\mid\theta,G)P(\theta\mid G)P(G).
$$

The left side is the joint posterior over parameters and graph. On the right, $P(D\mid\theta,G)$ is the data likelihood, $P(\theta\mid G)$ is the graph-conditional parameter prior, and $P(G)$ is the graph prior. The missing normalizer is $P(D)$, obtained by summing over $G$ and integrating over $\theta$. The factorization also encodes the prior assumption that parameters depend on the chosen graph. Temporal likelihoods must preserve within-trajectory dependence; rows from one trajectory generally form dependent cases.

###### Marginal likelihood or model evidence

For a graph $G$, the Bayesian marginal likelihood integrates parameters out:

$$
P(D\mid G)=\int P(D\mid\theta,G)P(\theta\mid G)\,d\theta.
$$

The integral ranges over every admissible parameter vector $\theta$. Each likelihood value is weighted by its prior density $P(\theta\mid G)$, so $P(D\mid G)$ is an average fit under the prior. A maximized likelihood evaluates one best parameter value. The marginal likelihood incorporates an Occam effect: a graph receives high evidence only when a non-negligible region of its prior parameter space explains the data well. Bayesian structure scores use this quantity. Parameter integration also differs from latent-variable marginalization with fixed parameters. Both operations use integration and integrate different variables.

###### Sufficient statistic

A statistic is sufficient for a parameter when it retains all information in the data relevant to that parameter under the model. Counts of node-parent configurations are sufficient statistics for discrete CPTs; expected sufficient statistics replace them in EM when values are unobserved.

###### Multiple trajectories

A multiple-trajectory dataset contains several sequences, often from different subjects or runs. Trajectories may be independent conditional on shared parameters or covariates. Observations within each trajectory remain temporally dependent. Subject-level heterogeneity can violate an unconditional independence assumption.

###### Structure learning

Structure learning estimates a graph or posterior over graphs. For DBNs this can include intra-slice edges, inter-slice edges, lag order, and time variation in non-stationary models.

###### Candidate parent set

A candidate parent set is an allowed subset of potential parents for a node. Temporal order, maximum lag, domain restrictions, sparsity limits, and resolution rules can reduce the search space.

###### Acyclicity constraint

Structure search must prevent directed cycles in each finite unrolling. With forward-only temporal edges, it is enough to enforce acyclicity on contemporaneous structure; continuous optimization methods often encode this as a differentiable constraint on an adjacency matrix.

###### Decomposable or local score

A decomposable score has $S(G;D)=\sum_iS_i(Z^i,\operatorname{Pa}(Z^i);D)$. Local graph edits then require recomputing only affected node-family scores.

###### Score equivalence

A score is score equivalent if it gives the same score to Markov-equivalent DAGs. Standard BDe/BDeu and BGe scores are designed to have this property under their assumptions.

###### Bayesian Information Criterion, BIC

In a log-score convention,

$$
\operatorname{BIC}(G)=\log P(D\mid\widehat\theta_G,G)-\frac{d_G}{2}\log n.
$$

The first term is the maximized log-likelihood under graph $G$, where $\widehat\theta_G$ is its MLE. The second term penalizes complexity: $d_G$ is the number of independently estimated free parameters, $n$ is the effective sample size under the likelihood's asymptotic argument, and $\tfrac12\log n$ is the price per parameter. Larger values are better in this convention because fit is added and complexity subtracted. Some software reports $-2$ times this expression and minimizes it. For dependent temporal data, $n$ must follow the likelihood factorization. The row count can give a different value.

**N²DBN usage:** the implemented parent adds two lag coefficients, one to the count channel and one to the duration channel. Its unscaled BIC price is therefore $\log T_k$ at a scale with $T_k$ epochs, and the search uses $\lambda_k=\lambda\log T_k$ with $\lambda$ selected on validation evidence.

###### BDe, BDeu, and BGe scores

BDe is a score-equivalent marginal likelihood for discrete BNs with Dirichlet priors. BDeu is its uniform equivalent-sample-size variant and can be hyperparameter-sensitive. BGe is the linear-Gaussian counterpart under normal-Wishart assumptions. Non-stationary BGe extensions can use segment-specific parameters with shared structure.

###### Regularization and sparsity

Regularization discourages excessive complexity. Examples include graph-size penalties, sparse-graph priors, and $\ell_1$ penalties. In coefficient-defined graphs, a zero coefficient can remove an edge.

###### Constraint-based, score-based, and hybrid learning

Constraint-based methods use conditional-independence tests. Score-based methods optimize over DAGs, equivalence classes, orderings, or parent sets. Hybrid methods first restrict candidates using independence information and then optimize a score.

###### Hill climbing, tabu search, and exact search

Hill climbing greedily adds, deletes, or reverses legal edges and can stop at local optima. Tabu search temporarily forbids recent moves and can accept short-term decreases. Exhaustive search checks every structure; more sophisticated exact algorithms prune and guarantee the optimum of the **chosen score**. Score optimality provides no guarantee of recovery of the true graph.

###### Structure MCMC and Bayesian model averaging

Structure MCMC samples a posterior over graphs. Posterior edge probabilities and predictions can be averaged across graphs. This average represents structural uncertainty across several graphs.

###### Complete and incomplete data

Complete data supply every required modeled value. Incomplete data contain unobserved entries due to missingness, censoring, or latent variables. These mechanisms have different scientific meanings and likelihood assumptions.

###### Expectation–Maximization, EM

EM alternates an E-step, computing a posterior or expected sufficient statistics for unobserved quantities, and an M-step, maximizing the expected complete-data log-likelihood. It can converge to a local optimum or saddle point.

###### Structural Expectation–Maximization, Structural EM

Structural EM optimizes graph structure as well as parameters, scoring candidates with expected sufficient statistics under the current model. **SEM** can also mean structural equation model, so the expansion should always be stated.

###### Structural Equation Model, SEM

A structural equation model expresses every variable as a function of its parents and a disturbance. Linear SEM coefficient matrices can encode BN or DBN structures. Structural EM is a different concept: it is an algorithm for learning structure with incomplete data.

###### Variational inference and ELBO

Variational inference approximates $p(z\mid D)$ with $q_\phi(z)$. It commonly maximizes

$$
\mathcal L(q,\theta)=
\mathbb E_q[\log p(D,z\mid\theta)]-\mathbb E_q[\log q(z)]
\leq\log p(D\mid\theta),
$$

the evidence lower bound (ELBO). In this expression:

- $D$ is observed data and $z$ collects latent variables;
- $p(D,z\mid\theta)$ is the model's complete-data joint density;
- $q(z)$ is a tractable approximation to the exact posterior $p(z\mid D,\theta)$;
- $\mathbb E_q$ means that the enclosed quantity is averaged with respect to $q$;
- the first term rewards explaining the observations and latent configurations under the model; and
- $-\mathbb E_q[\log q(z)]=\mathbb H[q]$ is the entropy of $q$, which discourages unjustified collapse.

The inequality means the ELBO never exceeds the log evidence. More exactly,

$$
\log p(D\mid\theta)-\mathcal L(q,\theta)
=\mathrm{KL}\!\left(q(z)\,\|\,p(z\mid D,\theta)\right)\ge0.
$$

The left side is the gap between the true log evidence and the lower bound. On the right, $\mathrm{KL}(q\|p)$ is the Kullback–Leibler divergence from the variational distribution to the exact posterior; it is non-negative and equals zero only when the two distributions agree almost everywhere. The bound is therefore tight exactly when $q$ equals the exact posterior.

###### Variational Structural EM

Variational Structural EM replaces Structural EM's exact latent posterior with a tractable variational approximation and alternates latent-state, parameter, and graph updates against a lower bound or an approximation to it.

**N²DBN usage:** the three coordinates are an E-step for $q$, an M-step for $\Theta$, and an S-step for the parent sets. An interior scale's variational coordinate receives both its own factor $\Psi_k$ and the factor $\Psi_{k-1}$ through which it generates the next finer scale. One outer sweep visits fine-to-coarse and then coarse-to-fine so that within-block detail statistics and coarser block-level information move in both directions.

###### Structured variational family

N²DBN factorizes the approximate posterior top-down by scale and, within a scale, separates the discrete and continuous trajectories. In the notation of the learning section,

$$
q(\mathcal S,\mathcal H)
=q^{(K)}(\mathcal S^{(K)},\mathcal H^{(K)})
\prod_{k=1}^{K-1}
q^{(k)}(\mathcal S^{(k)},\mathcal H^{(k)}\mid
\bar{\mathcal Z}^{(k+1)}),
$$

where:

- $\mathcal S=\{S_t^{(k)}\}_{k,t}$ and $\mathcal H=\{H_t^{(k)}\}_{k,t}$ collect all latent variables at all scales and epochs; equivalently, they are the components of $U^{(1:K)}$;
- $q^{(K)}$ is the approximate posterior at the coarsest scale, which has no parent scale;
- the product multiplies the conditional approximate-posterior factors for scales $k=1,\ldots,K-1$;
- $\mathcal S^{(k)}$ and $\mathcal H^{(k)}$ denote the **entire temporal trajectories** at scale $k$; single-epoch states carry the subscript $t$;
- $\bar{\mathcal Z}^{(k+1)}$ is the parent-scale summary or message supplied to scale $k$; and
- the direction of conditioning follows coarse-to-fine generation and retains separate inference factors at every scale.

Within a scale,

$$
q^{(k)}=q_S^{(k)}(\mathcal S^{(k)})\,
q_H^{(k)}(\mathcal H^{(k)}\mid\mathcal S^{(k)}).
$$

The factor $q_S^{(k)}$ represents the discrete regime-path distribution and $q_H^{(k)}$ the conditional continuous factor-path approximation. Parent messages include the mapped parent-regime marginals $\gamma_{\pi_k(t)}^{(k+1)}(\cdot)$ and observed parent totals $X_{\pi_k(t)}^{(k+1)}$ and $Y_{\pi_k(t)}^{(k+1)}$. The current learning equations omit the available parent factor mean from this message. This structured mean-field approximation preserves temporal dependence inside each regime chain and Gaussian factor trajectory. It breaks selected cross-scale and discrete–continuous dependencies to make inference tractable.

###### Regime marginal and pair marginal

$\gamma_t(m)=q(S_t=m)$ is the posterior probability of regime $m$ at epoch $t$, and $\xi_t(m,m')=q(S_{t-1}=m',S_t=m)$ is the adjacent-state posterior. N²DBN obtains both with log-space forward-backward recursion for a time-inhomogeneous Markov chain. They are sufficient for expected regime occupancies and transition counts.

###### Expected local evidence

The expected local evidence $\psi_t(m)$ is the expected log emission or allocation likelihood at epoch $t$ conditional on regime $m$. N²DBN evaluates nonlinear factor-dependent terms at the current posterior factor mean. This first-order plug-in approximation avoids numerical quadrature. It supplies the time-varying emission score used by forward-backward inference.

###### Regime-averaged Kalman and RTS smoother

Conditional on a fixed regime path, the continuous state is a switching linear-Gaussian system. N²DBN replaces regime-specific dynamics by posterior-regime-weighted dynamics, runs a Kalman filter, and applies the Rauch–Tung–Striebel (RTS) backward smoother. A **Jensen-gap covariance correction** adds the variation among regime-specific transition means that would be lost by using only their average. The coarsest non-Gaussian emission is inserted as a Laplace pseudo-observation; at finer scales the current factor update uses only its dynamic prior.

###### Laplace pseudo-observation

A Laplace approximation replaces a non-Gaussian log likelihood locally by a quadratic at a mode or current expansion point, producing Gaussian-like information for a Kalman update. N²DBN forms this term from the gradient and Hessian of the coarsest emission and clips Hessian eigenvalues so the local quadratic is concave. The generative observation family remains ZINB/NB; the approximation applies only during inference.

###### Profiled local structure score

For target label $\ell$ at scale $k$, a profiled local likelihood first refits the coefficients available under a proposed parent set $\mathcal P$:

$$
Q_{k,\ell}(\mathcal P;q,\Theta)
=\max_{\boldsymbol\beta_{\mathcal P}}
\sum_{t\in\mathcal T^{(k)}}\sum_{m=1}^{M_k}
\gamma_t(m)
\log p_{\boldsymbol\beta_{\mathcal P},\Theta_{-\mathcal P}}
\!\left(
X_t^{(k,\ell)},Y_t^{(k,\ell)}
\mid S_t^{(k)}=m,\bar H_t^{(k)},
\mathcal P^X(t),\mathcal P^Y(t)
\right).
$$

Here $\mathcal P\subseteq\mathcal C_{k,\ell}$ is one proposed parent set; $\boldsymbol\beta_{\mathcal P}$ collects both the count coefficients $\beta$ and duration coefficients $\alpha$ enabled by that set; and the maximization means that these coefficients are re-optimized separately for every candidate. The sum over $t$ covers the scale-$k$ epochs. The sum over $m$ averages the local log likelihood using the regime probabilities $\gamma_t(m)=q(S_t^{(k)}=m)$. The density is evaluated for the observed count–duration pair, conditional on regime $m$, the current factor mean $\bar H_t^{(k)}$, and the selected lagged count and duration values $\mathcal P^X(t)$ and $\mathcal P^Y(t)$. All remaining parameters $\Theta_{-\mathcal P}$ are held at their current values during this local comparison.

The definition of $Q_{k,\ell}$ contains only the profiled expected local likelihood. The implemented S-step applies the structure penalty separately and selects

$$
\widehat{\mathcal P}_{k,\ell}
=\arg\max_{\mathcal P\subseteq\mathcal C_{k,\ell},\ |\mathcal P|\le D_k}
\left\{Q_{k,\ell}(\mathcal P;q,\Theta)-\lambda_k|\mathcal P|\right\}.
$$

Here $D_k$ is the maximum allowed in-degree, $|\mathcal P|$ is the number of grouped label–lag parents, and $\lambda_k$ is the per-parent penalty. Keeping the penalty outside $Q_{k,\ell}$ agrees with `sec_learning.tex` and avoids using the same symbol for two different scores.

Evaluating an added edge at its incumbent coefficient of zero leaves the likelihood unchanged and incurs the penalty, so such a search can never add an edge. Profiling is therefore essential: a candidate is judged by the improvement from its best fitted coefficients.

###### Normalized local likelihood and block normalizer

A child-allocation score must include the probability normalizer over siblings in the same parent block. Omitting it leaves a predictor linear in unrestricted coefficients and can make the local objective unbounded. The normalized multinomial or Dirichlet–multinomial likelihood couples siblings, bounds the score, and makes parent-set comparisons meaningful.

###### Leave-one-out mean-field normalizer

In the finer-scale variational likelihood, N²DBN evaluates the focal child's score under a candidate regime and replaces block-mates' scores by their regime averages. This leave-one-out approximation avoids summing jointly over all sibling regime assignments. The local M-step acceptance score and the next E-step evidence therefore differ slightly, which motivates the outer monotonicity guard.

###### Per-target decomposition

Because learned observed-variable edges have strictly positive lag, every admissible parent-set assignment is acyclic. With parameters and $q$ fixed, the profiled objective separates by scale and target label, so structure learning becomes $K|\mathcal L|$ independent subset-selection problems. Under in-degree bound $D_k$, exhaustive search for one target checks

$$
\sum_{j=0}^{D_k}\binom{|\mathcal C_{k,\ell}|}{j}
=O(|\mathcal C_{k,\ell}|^{D_k})
$$

parent sets for fixed $D_k$. Here $\mathcal C_{k,\ell}$ is the candidate-parent set for target label $\ell$ at scale $k$, $|\mathcal C_{k,\ell}|$ is its number of candidates, $D_k$ is the maximum allowed in-degree, and $j$ is one possible parent-set size. The binomial coefficient $\binom{|\mathcal C|}{j}$ counts the distinct subsets of exactly size $j$; summing from zero includes the empty set through the largest admissible set. The big-$O$ statement describes growth for fixed $D_k$: it is polynomial in the candidate count. The exponent and constant can still make enumeration expensive when $D_k$ grows.

###### Search cascade and search alternatives

N²DBN may rank scale-target tasks using coarse correlation or regime information before running greedy forward-backward, beam, exhaustive, or genetic parent-set search. The cascade changes the search order. Every candidate remains eligible under a full traversal. Parent-set scores are memoized because different search paths revisit the same subsets.

###### Monotonicity guard and best-state restoration

The monitored sweep objective is summed scale-local log evidence minus the edge penalty. If an approximate sweep lowers the best value, N²DBN restores the best parameters, graph, and posterior summaries, halves the M-step learning rate, and retries; a third consecutive rejection stops at the best state. Thus the returned state is the best accepted sweep. The approximate coordinate updates alone provide no guarantee of monotonic ascent.

###### Shift confirmation and shifted-feature null

After structure search, an accepted edge is compared with surrogate versions of its own source feature. At the coarsest scale the null uses whole-series circular shifts, preserving marginal distribution and autocorrelation; below it, the feature is rolled within each parent block, also preserving block membership. An edge is retained only when its profiled gain exceeds a prespecified upper surrogate quantile, and retained edges are rechecked after removals.

This is a confirmation screen. It provides neither a causal test nor a family-wise multiple-testing correction. It can remove false or true edges. It cannot recover missed edges or eliminate dependence from an epoch-aligned common cause.

###### Declared search space and selected hyperparameter

N²DBN selects regime count $M_k$, factor dimension $r_k$, and penalty multiplier $\lambda$ using held-out log evidence. Maximum lag, in-degree bound, hierarchy, and resolution are scientific declarations that determine which questions the search can ask; tuning them on the same benchmark target would change the estimand and can bias comparison.

##### 7. Non-stationary and time-varying DBNs

###### Non-stationarity

A process is non-stationary when its distribution changes with time. In a DBN this can mean changing CPD parameters, graph structure, noise, trend, seasonality, or latent regime. A paper should state exactly what varies.

###### Non-homogeneous DBN, NH-DBN

NH-DBN is an umbrella term for a time-dependent transition mechanism. In the supplied NH-DBN work, change points divide the data into segments and interaction parameters can be segment-specific. Some variants keep structure fixed; other variants also vary it.

###### Time-varying DBN, TV-DBN

In Song, Kolar, and Xing, a TV-DBN allows directed structure to vary smoothly with time. Kernel-weighted observations and sparse autoregression share information across neighboring time points. This differs from fitting unrelated graphs independently at each time.

###### Parameter and structural non-stationarity

Parameter non-stationarity changes relationship strength or form with a fixed edge set. Structural non-stationarity changes which edges exist. A coefficient crossing zero blurs the distinction unless edge presence is defined explicitly.

###### Change point, segment, and piecewise stationarity

A change point is a time where the regime changes. A segment is the contiguous interval between change points. A piecewise-stationary model is constant within each segment and may change at segment boundaries.

###### Regime

A regime is a mode of system behavior with its own parameters or structure. A hidden regime variable can select among transition models. Regimes can recur, so several disjoint segments may share one regime.

###### Local stationarity

A locally stationary process changes globally and is approximately stationary over a sufficiently short neighborhood. The validity of this approximation depends on window width and rate of change.

###### Temporal smoothness and information sharing

Time-varying models are data-hungry. Kernel weighting, smoothness penalties, hierarchical priors, or parameter coupling let nearby times or segments share strength.

###### Edge-wise parameter coupling

Edge-wise coupling lets the model determine which interaction parameters remain similar across segments and which can change. Each edge can therefore have its own degree of sharing.

###### Trend and seasonality

A trend is systematic long-term change. Seasonality is systematic variation tied to a repeating cycle. Explicit time or calendar covariates can make a conditional mechanism homogeneous even when the unconditioned series varies over time.

##### 8. Temporal resolution and the N²DBN model

###### Time granularity, temporal resolution, and time scale

These terms describe the wall-clock width or detail of a temporal unit. Fine resolution uses shorter epochs; coarse resolution uses longer ones. *Time scale* can also mean a characteristic process rate, so a paper should state whether it means bin width, frequency band, lag horizon, or physical mechanism.

###### Multi-scale or multi-resolution model

This label is overloaded in the literature. It can mean temporal aggregation, recursive change-point partitions, fast/slow latent processes, multiple physical rates, or neural down-sampling and feature fusion. N²DBN specifically uses a **known nested temporal aggregation hierarchy** together with scale-specific latent dynamics and learned lagged relations.

###### Episode, event, and event log

An event log records timestamped occurrences. In N²DBN an episode is the marked tuple $(\ell_i,s_i,d_i)$: label, start time, and integer duration. It is a discrete marked temporal process after time is quantized into finest-resolution slots.

###### Epoch, slot, and atomic resolution

An epoch at scale $k$ is a contiguous calendar interval $I_t^{(k)}$. A training epoch is an unrelated optimization term. A **slot** is an epoch at the finest scale, $k=1$, of wall-clock width $\Delta_1$. Atomic resolution assumes

$$
X_t^{(1,\ell)}\in\{0,1\},
$$

where superscript $1$ selects the finest scale, $t$ selects one finest slot, and $\ell$ selects a label. Membership in $\{0,1\}$ means $X_t^{(1,\ell)}$ is Bernoulli-valued: zero means no label-$\ell$ episode starts in slot $t$, and one means exactly one starts. The restriction applies **per label**, so different labels may each start an episode in the same slot. If a value above one is possible for a single label, exact start-boundary reconstruction requires a finer slot or an additional within-slot allocation model.

**Specification note:** the benchmark section both enforces this binary condition and later says that two episodes of one label may begin in the same finest epoch. Those statements cannot both hold. The formal model, reconstruction map, identifiability section, and stated generator constraint support the binary interpretation.

###### Slot-normalized duration and start-epoch booking

$Y_t^{(k,\ell)}$ is measured in finest-slot units at **every** scale. An episode's entire duration is booked to the epoch containing its start, even if the episode extends beyond that epoch. Consequently, durations are additive across the hierarchy and $Y_t$ can exceed the width of epoch $t$. The variable $Y$ summarizes booked duration by start epoch. Occupancy time is a separate quantity.

###### Finest and coarsest resolution

N²DBN indexes the finest level by $k=1$ and coarsest by $k=K$. The convention is project-specific; some multiresolution papers reverse it.

###### Parent map, child block, and nested hierarchy

The map

$$
\pi_k:\mathcal T^{(k)}\to\mathcal T^{(k+1)}
$$

is shorthand for the map between the indices of the interval collections $\mathcal T^{(k)}=\{I_t^{(k)}\}_{t=1}^{T_k}$ and $\mathcal T^{(k+1)}=\{I_c^{(k+1)}\}_{c=1}^{T_{k+1}}$. More explicitly, $\pi_k(t)=c$ means $I_t^{(k)}\subseteq I_c^{(k+1)}$. For $k\in\{1,\ldots,K-1\}$, every fine epoch has exactly one parent; surjectivity, when assumed, ensures every coarse epoch has at least one child. The sibling or child block of $c$ is

$$
\operatorname{ch}_k(c)=\{t:\pi_k(t)=c\}.
$$

The braces define a set, the colon means “such that,” $t$ ranges over $\mathcal T^{(k)}$, and $c\in\mathcal T^{(k+1)}$ is fixed. Thus $\operatorname{ch}_k(c)$ contains exactly the fine epochs whose parent is $c$. These child sets are disjoint and together partition $\mathcal T^{(k)}$. Composed parent maps connect non-adjacent resolutions. A regular hierarchy has the same number of children per parent; the map formulation can also represent irregular calendar blocks and partial boundary periods.

###### Fold or aggregation operator, $\mathcal A_k$

The deterministic fold from scale $k$ to $k+1$ sums children componentwise:

$$
(\mathcal A_kZ^{(k)})_c
=\sum_{t:\pi_k(t)=c}Z_t^{(k)},
\qquad Z^{(k+1)}=\mathcal A_kZ^{(k)}.
$$

Here $\mathcal A_k$ is the deterministic aggregation operator, $Z^{(k)}=(X^{(k)},Y^{(k)})$ is the whole fine-scale array, and $(\mathcal A_kZ^{(k)})_c$ is its aggregate for one parent epoch $c$. The constrained summation index $t:\pi_k(t)=c$ means “sum over every child of $c$.” Because $Z_t^{(k)}$ has count and duration components, the addition is componentwise: counts are summed with counts and durations with durations, separately for every label. The second equality applies this operation to every parent $c$ and names the resulting complete array $Z^{(k+1)}$. Observed data are constructed bottom-up with this map. Folding performs temporal aggregation. Probabilistic marginalization integrates or sums out random variables.

###### Fiber of an aggregate

For a parent value $v$, the fiber

$$
\mathcal F_k(v)=\{u:\mathcal A_ku=v\}
$$

where $v$ is a fixed coarse parent total or coarse array, $u$ is a candidate fine allocation with the matching child dimensions, and $\mathcal A_ku=v$ is the exact aggregation constraint. $\mathcal F_k(v)$ is a **set** of inverse images: many different $u$ can share the same aggregate $v$. For example, if a parent count is three over two children, $(0,3)$, $(1,2)$, $(2,1)$, and $(3,0)$ all lie in the unconstrained integer fiber. Capacity, duration-floor, and active-child rules may remove some candidates from the model's admissible support.

###### Unfold or child kernel, $W_k$

An unfold is the stochastic inverse problem: draw one fine allocation conditional on a parent total. N²DBN's kernel

$$
K_k(Z^{(k)}\mid Z^{(k+1)},U^{(k)},C^{(k)})
=W_k(\eta^{(k)},Z^{(k+1)};\Theta^{(k)})
$$

uses the following notation:

- $K_k(\cdot\mid\cdot)$ is the full conditional probability kernel for the scale-$k$ child array;
- $Z^{(k)}$ is the random finer array being generated;
- $Z^{(k+1)}$ is the already generated parent array whose totals must be preserved;
- $U^{(k)}$ is the scale-$k$ latent regime/factor trajectory and $C^{(k)}$ is observed context;
- $\eta^{(k)}$ collects the predictor scores computed from parents, latents, context, depletion, and lagged values;
- $\Theta^{(k)}$ is the scale-$k$ parameter block; and
- the semicolon separates model parameters from the stochastic arguments of $W_k$.

$W_k$ is the implemented normalized allocation map that evaluates this kernel. Its support is exactly $\mathcal F_k(Z^{(k+1)})$, meaning it assigns zero probability to every finer array that fails to aggregate to the parent. Because many children fold to the same parent, the inverse image is a set. The kernel defines a conditional distribution over that set.

###### Top-down generative factorization

N²DBN first draws the coarsest array with $E_K$, then recursively applies $W_{K-1},\ldots,W_1$. Data representation folds fine-to-coarse. Simulation and forecasting unfold coarse-to-fine. Fitting separate likelihoods to every deterministic view as if they were independent would count the same event evidence repeatedly. The joint factorization treats finer arrays as conditional allocations of their parents and counts the event evidence once.

###### Exact multiresolution coherence

Because every child kernel is fiber-supported,

$$
\sum_{t:\pi_k(t)=c}X_t^{(k,\ell)}=X_c^{(k+1,\ell)},\qquad
\sum_{t:\pi_k(t)=c}Y_t^{(k,\ell)}=Y_c^{(k+1,\ell)}
$$

for every adjacent-scale pair $k,k+1$, coarse epoch $c$, and label $\ell$. The left sides add all child count or duration values; the right sides are the corresponding stored parent values. The two equalities hold **almost surely**, meaning with probability one under the generative model for each complete draw. No averaging across draws is involved. The identity composes across non-adjacent scales because applying successive aggregation operators preserves the equality. Almost-sure equality implies equality in expectation under integrability and gives exact numerical reconciliation for every draw.

###### Reconciliation

Forecast reconciliation projects or adjusts independently produced forecasts so their aggregates agree. N²DBN restricts predictive support to the coherent integer lattice. This construction preserves integer values and the point mass at zero and requires no post-hoc projection. Coherence alone provides no guarantee of calibration or forecast accuracy.

###### Temporal and cross-sectional hierarchy

A temporal hierarchy aggregates one stream across time bins, such as minutes into hours. A cross-sectional hierarchy aggregates different component series, such as stores into regions. N²DBN's principal hierarchy is temporal. Its labels form multiple parallel series.

###### Coarsest emission map, $E_K$

The coarsest scale has no parent total to allocate. Its count branch uses a zero-inflated negative binomial (ZINB) and its positive-count duration branch uses a negative binomial (NB). Count and duration predictors can share latent and contextual drivers; the duration predictor also conditions on the contemporaneous count. In the current implementation, learned lag terms enter the count and duration means and are absent from the zero-inflation logit.

###### Negative binomial distribution and overdispersion

The negative binomial is a count distribution whose variance may exceed its mean. Its dispersion parameter controls extra-Poisson variation; parameterizations differ, so software must state whether large dispersion approaches the Poisson limit or produces heavier tails. N²DBN estimates separate coarsest count and duration dispersions and imposes a positive floor for numerical regularity.

###### Zero-inflated negative binomial, ZINB

A ZINB mixes a structural-zero component with an NB count component:

$$
P(X=0)=\varpi+(1-\varpi)P_{\mathrm{NB}}(0),\qquad
P(X=x>0)=(1-\varpi)P_{\mathrm{NB}}(x).
$$

Here $X\in\mathbb N_0$ is the count, $\varpi\in[0,1]$ is the probability of selecting the structural-zero component, and $1-\varpi$ is the probability of selecting the NB component. $P_{\mathrm{NB}}(x)$ is the NB probability mass at count $x$, determined by its mean and dispersion. For $x>0$, only the NB component can produce the observation, hence the factor $1-\varpi$. For $x=0$, either the structural component contributes $\varpi$ or the NB component contributes $(1-\varpi)P_{\mathrm{NB}}(0)$. These cases are mutually exclusive mixture components and are added. M-step gradients assign each observed zero its posterior responsibility for the NB component and its posterior responsibility for the structural-zero component.

###### Duration floor and duration surplus

Every started episode occupies at least its start slot, so $Y_t^{(k,\ell)}\ge X_t^{(k,\ell)}$. At a finer scale, each active child first receives a floor equal to its allocated count. The remaining parent duration units form the **duration surplus** relative to that floor and are split stochastically. The paper's displayed coarsest learning likelihood is an NB model for positive $Y$ itself. The term *surplus* refers most literally to the residual allocated below the coarsest scale.

###### Dirichlet–multinomial allocation, DM

The DM is an overdispersed distribution over an integer composition with a fixed total. One representation draws shares $P\sim\operatorname{Dirichlet}(\alpha)$ and then draws allocations conditional on $P$ with a multinomial. N²DBN sets concentration coordinates to a total concentration times block-softmax weights, then constrains or repairs the draw to satisfy model support.

###### Concentration and multinomial limit

The count and duration allocation concentrations, $\kappa_k$ and $\tau_k$, control variation around the softmax shares. As concentration tends to infinity the DM approaches its multinomial limit; finite concentration admits extra allocation variation. N²DBN estimates concentration using the complete block likelihood because the block-constant term omitted from regime comparisons is needed to identify it. Assuming the multinomial limit can overstate evidence and favor spurious lagged edges.

###### Block softmax and allocation share

For the count-allocation scores $\eta_t^{(k,\ell)}$ within one parent block,

$$
\omega_t^{(k,\ell)}
=\frac{\exp(\eta_t^{(k,\ell)})}
{\sum_{s\in\operatorname{ch}_k(c)}\exp(\eta_s^{(k,\ell)})}.
$$

Here $c=\pi_k(t)$ is the parent of child $t$, $\ell$ is the event label, $\eta_t^{(k,\ell)}\in\mathbb R$ is child $t$'s unconstrained **count** score, and $s$ is a dummy index ranging over every sibling in $\operatorname{ch}_k(c)$. Exponentiation converts real scores into positive numbers and the denominator normalizes them, so $0<\omega_t^{(k,\ell)}<1$ and the count shares sum to one within each label-specific block. Adding the same constant to every sibling score cancels between numerator and denominator.

The duration branch uses a distinct score $\zeta_t^{(k,\ell)}$ and share $\rho_t^{(k,\ell)}$:

$$
\rho_t^{(k,\ell)}
=\frac{\exp(\zeta_t^{(k,\ell)})}
{\sum_{s\in\operatorname{ch}_k(c):\,X_s^{(k,\ell)}>0}
\exp(\zeta_s^{(k,\ell)})},
\qquad X_t^{(k,\ell)}>0.
$$

Thus $\eta$ and $\omega$ belong to count allocation. The symbols $\zeta$ and $\rho$ belong to duration allocation. The duration denominator ranges only over **active children**, those with positive count. Siblings compete for fixed parent count and duration budgets, so their allocations remain dependent even conditional on predictors. In the variational E-step, the manuscript replaces the ordinary sibling scores in these denominators by the leave-one-out regime-averaged scores described below.

###### Active-child set

The duration surplus is allocated only among children with positive allocated count. This prevents duration being assigned to an epoch in which no episode starts and implements the start-epoch booking convention.

###### Capacity cap and redistribution

At scale $k$, a child epoch contains $n_k$ finest slots and can start at most $n_k$ episodes of one label. A raw allocation above this cap is clipped, and excess units are reassigned one at a time among siblings with remaining capacity, in proportion to their allocation weights. Redistribution stays inside the block and preserves its total exactly.

###### Sequential depletion feature

The depletion variable records how much of a parent block's fixed budget earlier children have already consumed. It is prefix-measurable and varies within the block, so it represents competition among siblings without using future children and remains identifiable after block-softmax normalization.

###### Prefix-measurable feature

A feature at epoch $t$ is prefix-measurable if it is computable from information available no later than $t$ under the model's ordering. Strict-past lags, recency, geometric lag summaries, and sequential depletion satisfy this requirement. It prevents future leakage and preserves the DAG ordering.

###### Recency and geometric-lag summary

Recency is the elapsed number of epochs since a label's most recent positive count. A geometric-lag summary is

$$
M_t(\rho)=(1-\rho)\sum_{d\ge1}\rho^{d-1}Z_{t-d}.
$$

Here $M_t(\rho)$ is the summary available at current epoch $t$, $Z_{t-d}$ is the observed value $d$ epochs in the strict past, $d=1,2,\ldots$ is the lag index, and $0\le\rho<1$ is the decay parameter. The lag-$d$ weight is $(1-\rho)\rho^{d-1}$; over an infinite history these weights sum to one. Small $\rho$ emphasizes the most recent lag. A value near one spreads weight far into the past. At a finite sequence boundary the sum is truncated and must either be renormalized or initialized by an explicit convention. Recency and this geometric summary compress potentially many lag alternatives into strict-past, within-block-varying features. The same Greek letter $\rho$ also denotes the duration allocation share elsewhere in the manuscript; local subscripts and definitions distinguish the two quantities.

###### Discrete regime, $S_t^{(k)}$

The regime is a persistent latent Markov state intended to carry slow, discrete changes in local dynamics. Its transition logits can depend on the previous same-scale regime, the mapped parent-scale regime, and context. Regimes may recur across several disjoint segments. Change points mark transitions between regimes.

###### Continuous latent factor, $H_t^{(k)}$

The factor is a shared low-dimensional continuous state intended to explain same-epoch co-movement across labels. It represents contemporaneous association without an observed-to-observed directed edge. Residual hidden confounding may remain, and an overly flexible factor can attribute genuine lagged signal to the factor.

###### Separation of explanatory roles

N²DBN assigns parent totals to cross-scale budget transfer, lagged edges to directed within-scale predictive dependence, regimes to persistent drift, and factors to contemporaneous co-movement. This separation is partly structural and partly an assumption. Omitting one mechanism can redirect its variation into self-lags, extra regimes, or latent factors.

###### Regime–feature interaction at finer scales

Below the coarsest scale, a regime that is constant throughout a child block cannot be identified as a purely additive shift because block softmax removes block constants. Its effect must interact with a feature that varies inside the block, such as depletion, recency, or a lagged predictor. A purely additive regime level is allowed at the coarsest emission.

###### Context and deterministic calendar node

$C_t^{(k)}$ contains known context such as interventions or participant-independent conditions. The learning section denotes this same vector by $\mathcal K_t^{(k)}$; hence $\mathcal K_t^{(k)}\equiv C_t^{(k)}$ in this glossary. The paper also treats hour-of-day, weekday, and similar clock functions as deterministic graph nodes with no parents. Such nodes may enter an observed target at lag zero without creating a cycle because the clock is exogenous. Learned observed-to-observed edges retain the positive-lag restriction.

###### Shadow, detail, and block-average projector

Within a child block, any predictor vector decomposes as

$$
\eta=\Pi_k\eta+(I-\Pi_k)\eta,
$$

where $\eta$ is the vector of allocation scores across child epochs, $\Pi_k$ is the linear projector that replaces every coordinate in a parent block by that block's mean, $I$ is the identity operator, and $I-\Pi_k$ removes the block mean. The first term $\Pi_k\eta$ is constant within each block and is called the **shadow**. The second term $(I-\Pi_k)\eta$ sums to zero within every block and is called **detail**. Because $\Pi_k^2=\Pi_k$, applying the projector twice changes nothing; because the two components lie in complementary subspaces, their sum reconstructs the original score exactly. The child softmax likelihood identifies detail and is invariant to shadow. Coarser parent totals contain the information about shadow.

###### Block-constant invariance

Adding the same constant to all scores in one child block leaves their softmax and hence their DM allocation law unchanged. This is an exact observational invariance. Raw allocation intercepts and other block-constant coefficient levels therefore cannot be recovered from the child likelihood alone, even with unlimited data.

###### Low-pass fold and band-pass ladder

Block summation acts as a low-pass operation. N²DBN interprets scale-$k$ detail as the temporal-frequency band approximately between $1/n_{k+1}$ and $1/n_k$, where $n_k$ is the number of finest slots per scale-$k$ epoch. This interpretation is an idealization tied to the chosen nested blocks. Arbitrary irregular calendars generally fail to form an exact orthogonal Fourier filter bank.

###### Fourier harmonic and harmonic order

A periodic effect of period $P$ can be represented by sine/cosine pairs. Using $H$ harmonics costs $2H$ coefficients and makes $H$ a smoothness/complexity dial; a full free periodic profile costs roughly $P-1$ contrasts. Each harmonic belongs at a resolution capable of resolving its frequency.

###### Haar detail and localized change

Haar wavelets are localized, sum-zero contrasts on a nested block tree. They represent abrupt level changes more directly than Fourier terms. A change point produces a large Haar detail coefficient near the scale matching its persistence.

###### Home scale

The home scale of a periodic, calendar, or localized feature is the resolution where it varies within parent blocks and therefore enters the identifiable detail subspace. Placing a feature at a coarser scale where it is constant within every relevant block adds no allocation information.

**Specification note:** the inequality in the current calendar-node corollary conflicts with its following verbal condition that the parent span at least one full period. Until the inequality is corrected, use the invariant definition above: nonzero variation within a parent block. This definition also covers equality and irregular-calendar boundaries.

###### Cross-scale structural gate

For a typed relation $r$, $z_{r,k}$ records whether some representation is active at scale $k$. Keep and appear probabilities model $P(z_{r,k}=1\mid z_{r,k+1})$. A Bernoulli tie gate with Beta-conjugate updates allows adjacent scales to share structure without forcing equality when a relation genuinely appears or disappears with resolution.

**Specification note:** `sec_model.tex` defines this gate. The current `sec_learning.tex` contains no derivation of its promised Beta/Bernoulli update. It is therefore part of the compact model specification and currently lacks a documented learning update.

###### Broadcast operator and shadow-only parameter coupling

The broadcast operator maps a parent-scale effect into child epochs and rescales it for feature standard deviation and lag units. Cross-scale Gaussian coupling is applied only to the child predictor's shadow, the component shared with and identified by the parent. Child detail receives its own prior because it has no parent-scale counterpart.

As with the structural gate, the compact model gives this prior mathematically. The learning section represents adjacent-scale coupling only abstractly through $\Lambda_k$ and omits the stated broadcast/tie parameter updates.

###### Scientific relation and representation

A scientific relation is an ordered label pair $r=(a,b)$. Scale, lag, source/target quantity channel, and basis choice are alternative **representations** of that relation. N²DBN uses a nested penalty: a relatively expensive relation price is paid once, and extra representations have a lower refinement price. This avoids making one relation compete repeatedly against its own encodings.

**Specification note:** the compact model proposes this nested penalty. The current learning derivation and benchmark use the simpler $\lambda_k|P|$ penalty on label–lag parents. Until the S-step is updated to optimize relation groups and their representations, describe reported fits as per-parent penalized. Implementation of the nested relation price requires a corresponding S-step update.

###### Self-lag stratum and multiplicity price

Self-lags represent autocorrelation and are exempt from the cross-label relation price in the current construction, then reported separately so they cannot consume the same interpretive budget as cross-label relations. The effective price for the $j$th accepted relation adds a combinatorial term $\log((C_k-j+1)/j)$, accounting for different candidate-field sizes across scales.

These are compact-model rules attached to the nested relation penalty. The current S-step equations based on $\lambda_k|P|$ omit them.

###### Population of event logs and participant random intercept

For repeated participants, N²DBN shares the graph and fixed effects and gives each participant a Gaussian random intercept and participant-specific latent paths. The shared parameters use data from all participants; the random intercepts represent person-level baseline differences. Lag histories never cross participant boundaries; invalid history-dependent emissions are masked without dropping the epoch from aggregation identities.

###### Event-log reconstruction

At the finest scale, every $X_t^{(1,\ell)}=1$ creates an episode beginning in slot $t$ with inclusive end $t+\max(Y_t^{(1,\ell)},1)-1$, clipped at the record horizon. Reconstruction is deterministic once $(X^{(1)},Y^{(1)})$ is known.

###### Occupancy and mutual-exclusivity projection

Occupancy counts how many reconstructed episodes of a label cover each slot. For a label declared mutually exclusive with itself, an episode overlapping the next start is truncated before that next start. This preserves starts and episode counts and shortens duration. Consequently, the safeguard can break exact **trace-level** duration reconciliation. The generated arrays remain exactly coherent.

###### Coherent ancestral forecast

Forecasting filters latent states to the prediction origin, propagates their transitions without reading future observations, samples $E_K$ at the coarsest scale, and descends through $W_{K-1},\ldots,W_1$. Each sampled path uses its own earlier simulated values for lagged predictors. Every predictive draw is coherent. Independently computed marginal point forecasts can violate coherence; summaries derived from the same joint draws preserve it.

###### Nested Non-Stationary Dynamic Bayesian Network, N²DBN

N²DBN is the project's top-down, multiresolution latent-variable DBN for labeled episode starts and durations. It combines exact fiber-supported allocation, resolution-specific positive-lag graphs, persistent discrete regimes, continuous co-movement factors, cross-scale shadow coupling, and variational Structural EM. Its learned edges are conditional predictive temporal relations inside the declared admissible class. A causal interpretation requires additional assumptions.

##### 9. Identifiability and equivalence in N²DBN

###### Identifiability and estimability

A quantity is identifiable if different values imply different distributions of the observables, subject to the stated model class. Estimability concerns finite-data precision and computation. Exact invariance is an identifiability failure that no sample size fixes; weak edge effects, collinearity, and rare regimes are power or estimability problems.

###### Observational equivalence orbit

An invariance transformation maps one parameterization to another with exactly the same observed-data distribution. The set reached by all such transformations is an equivalence orbit. Recovery metrics should compare functions constant on these orbits. Arbitrary raw coordinates can change within an orbit.

###### Regime label switching

Permuting regime labels and every regime-indexed parameter leaves the likelihood unchanged. Regime paths and parameters must therefore be aligned by a permutation before comparison. Accuracy, adjusted Rand index, and normalized mutual information address different aspects of regime recovery; the latter two are label-permutation invariant by construction.

###### Factor rotation, sign, and loading subspace

For any invertible change of factor coordinates $H'_t=RH_t$, transforming dynamics and loadings consistently leaves their inner products and observed law unchanged. Individual factor signs, order, and coordinates remain unidentified. The loading row space or another rotation-invariant subspace summary is the appropriate target.

###### Coarsest intercept–regime trade

At the coarsest scale, a constant may move between an emission intercept and all regime effects without changing their sum. The regime-specific combined level is identified. The two raw blocks are unidentified separately.

###### Destination-free transition-logit shift

A softmax over destination regimes is unchanged when every destination logit receives the same quantity for a fixed conditioning state or context. The resulting row-stochastic transition probability table at a reference context is identified. An unconstrained absolute logit level is unidentified.

###### Identified allocation contrast

At finer scales, block constants vanish. Regime effects are therefore compared after subtracting their per-label mean across regimes, or through the child shares implied by their centered softmax. Lag coefficients, depletion effects, count–duration coupling, dispersions, and allocation concentrations remain identifiable when their regressors vary within blocks and are non-collinear.

###### Restricted graph identifiability

N²DBN claims graph recovery only relative to a finite candidate set, positive-lag restriction, in-degree bound, latent dimension, penalty, and chosen resolutions. An edge is recoverable when its inclusion changes the conditional law enough over the observed support. Nearly collinear candidates can remain indistinguishable, and predictive performance can stay stable. These conditions establish restricted predictive graph recovery; unrestricted causal-graph identification requires stronger assumptions.

###### Role-separation assumption

Interpretation assumes regimes, factors, lagged edges, context, and parent totals play their assigned roles. Ablation tests this assumption; fitting alone provides insufficient evidence. A low-dimensional factor may leave some common causes unmodeled, so learned observed edges have a predictive interpretation. Causal interpretation requires additional assumptions.

###### Identifiability screen

Before using recovery metrics, the benchmark runs inference at the true graph and parameters and profiles every candidate edge's attainable likelihood gain. A scenario is informative only when true-edge gains exceed both the best non-edge gains and the edge penalty, and when oracle latent recovery is nontrivial. This check separates learner failures from generated scenarios with unrecoverable targets.

##### 10. Structure-recovery evaluation

###### True positive, false positive, and false negative edge

A true positive (TP) is a learned edge present in the reference. A false positive (FP) is a learned edge absent from the reference. A false negative (FN) is a reference edge absent from the learned graph. N²DBN's current benchmark compares scale-specific $(\text{source label},\text{lag},\text{target label})$ triples. Source, target, lag, and resolution must therefore match; the current metric collapses count/duration channel types. A future channel-typed metric would be a different, finer recovery target.

###### Precision, recall, and F1

$$
\operatorname{Precision}=\frac{TP}{TP+FP},\qquad
\operatorname{Recall}=\frac{TP}{TP+FN},
$$

$$
F_1=2\frac{\operatorname{Precision}\operatorname{Recall}}
{\operatorname{Precision}+\operatorname{Recall}}.
$$

The counts and denominators have different meanings:

- $TP$ is the number of correctly recovered reference edges;
- $FP$ is the number of reported edges absent from the reference;
- $FN$ is the number of reference edges the learner failed to report;
- $TP+FP$ is the number of reported edges, so precision asks “what fraction of reports were correct?”; and
- $TP+FN$ is the number of true reference edges, so recall asks “what fraction of the truth was recovered?”

Both precision and recall lie between zero and one when their denominators are nonzero. $F_1$ is their harmonic mean: it is one only when both equal one, is zero if either is zero, and is pulled toward the smaller of the two. The leading factor two makes $F_1=P=R$ when precision and recall are equal. If there are no reported edges, precision has a zero denominator; if there are no true edges, recall has a zero denominator. Those cases require an explicit convention. Silent arithmetic leaves the result undefined or misleading. All three metrics inherit the chosen edge-matching convention.

###### Empty-set convention and staying silent

When both learned and true edge sets are empty, precision and recall are formally $0/0$. N²DBN declares precision, recall, and $F_1$ to be one for that case because the empty scale was recovered perfectly. The convention must accompany macro averages. It rewards a method for correctly staying silent and gives an always-empty method the same credit on every truly empty scale.

The **empty-graph baseline** is consequently substantive. If a learner's SHD is worse than that baseline, its reported structure is less accurate than reporting no edges.

###### Per-scale, macro, and pooled evaluation

Per-scale metrics report false positives at a deliberately empty resolution. A pooled edge set can conceal those errors among true positives at another scale. A macro average gives each scale equal weight and depends on the empty-set convention; a pooled micro metric weights scales by their number of candidate or true edges. N²DBN reports the per-scale quantities before any summary.

###### Structural Hamming Distance, SHD

SHD counts edge edits between graphs. Conventions differ on whether reversal costs one or two and how partially directed edges are handled. N²DBN uses symmetric difference on the directed, positive-lag edge tuples: every missing and every spurious edge costs one, and reversal is outside its admissible lagged search space.

###### Structural Intervention Distance, SID

SID counts disagreements in the intervention distributions that two DAGs would imply under graphical adjustment. SHD counts edge edits. Two graphs can have similar SHD and very different SID. SID requires causal DAG semantics and therefore applies to N²DBN's predictive lagged graph only under stronger causal assumptions.

###### Balanced Scoring Function, BSF

BSF is a confusion-matrix score designed to balance edge and non-edge recovery when graphs are sparse, reducing the built-in advantage that some conventional accuracy-like scores give the empty graph. Report BSF together with precision, recall, false positives, and an explicit empty-graph baseline.

###### AUROC and AUPRC

AUROC measures whether edge scores rank positives above negatives across thresholds. AUPRC summarizes the precision–recall trade-off and is usually more informative for sparse graphs because false positives directly lower precision; its chance baseline is the positive-edge prevalence. N²DBN credits a tied score block at the precision after the entire block is included, counts an unscored true edge as missed, and treats AUPRC as undefined on a scale with no true edge.

###### Regime accuracy, adjusted Rand index, and normalized mutual information

Regime accuracy is computed after the best label permutation. It can be misleading when one regime dominates, so adjusted Rand index (ARI), which corrects pairwise clustering agreement for chance, and normalized mutual information (NMI), which measures shared partition information, are reported beside it. A true path that never switches has no regime-segmentation task and should receive an N/A label. Assigning a perfect score would add evidence from an absent task.

###### Factor recovery

Pointwise comparison of factor coordinates is invalid under rotation and sign invariance. N²DBN uses an absolute-correlation or loading-subspace summary after alignment. A componentwise best-correlation score is exactly rotation-invariant only in special cases, so a principal-angle or projection-matrix distance is the cleaner general subspace metric.

###### Aggregation consistency error

Aggregation error sums the absolute residual between each parent and the sum of its children, separately for $X$ and $Y$. It certifies the generator and data split when computed on model-generated arrays, where it must be zero. It becomes a method comparison only when evaluated on arrays or forecasts reconstructed by the method itself.

###### Held-out log evidence

Held-out log evidence measures predictive fit after integrating or approximating latent states. In N²DBN, scale-local allocation likelihoods may omit a data-only constant. Values therefore compare methods on the **same** dataset. Raw magnitudes across datasets with different lengths or count levels are incomparable. Paired within-replicate differences are preferable to averaging unnormalized evidence across scenarios.

###### Invariance-aware parameter recovery

Parameter recovery must first align regime labels and then compare identified quantities: transition tables, combined coarsest regime levels, centered finer-scale regime contrasts or implied shares, loading subspaces, dispersions, and on/off-support lag mass. Entrywise error on raw logits, factor coordinates, or allocation intercepts penalizes equivalent parameterizations and therefore gives an invalid recovery metric.

###### Oracle graph and oracle parameters

An oracle-graph fit fixes the true graph and estimates latent states and parameters, isolating estimation error from structure-search error. An oracle-parameter run uses the true graph and parameters and performs only inference, giving the attainable latent-recovery ceiling for that realized scenario. Both are diagnostic reference procedures and are unavailable in deployment.

###### Ablation

An ablation removes or freezes one mechanism and keeps the rest of the protocol fixed. N²DBN uses single-scale, no-factor, no-regime, empty-graph, and oracle-graph variants to isolate cross-scale messages, contemporaneous factors, persistent regimes, learned edges, and search error. An ablation supports a role-separation claim only when all other search spaces, initialization, data splits, and tuning opportunities remain comparable.

###### Coherence-preserving chronological split

Training, validation, and test partitions should be chronological for forward prediction and cut only at coarsest-block boundaries. Cutting inside a parent block gives one partition only part of its children and breaks the aggregation identity within that split. Random row splitting additionally leaks temporal dependence.

###### Deliberate misspecification

A misspecified benchmark family violates one modeling assumption on purpose, such as omitting a true context variable or changing the noise family, and measures degradation relative to a paired well-specified scenario. The current paper calls its large-dispersion NB variant *heavier-tail*. Under the paper's own dispersion convention, a large value approaches the Poisson limit and is **less** overdispersed. That variant should be renamed (for example, *near-Poisson*) or parameterized in the opposite direction.

###### Varsortability and $R^2$-sortability

Varsortability measures how closely increasing marginal variance follows causal order in synthetic DAG data. $R^2$-sortability measures a related ordering in the explainable fraction of variance and may persist after standardization. High values let an algorithm exploit a simulator artifact without learning structure. Benchmarks should standardize where appropriate and report both diagnostics. Standardization may leave some ordering leakage.

###### Synthetic, semi-synthetic, and real-ground-truth benchmark

A synthetic benchmark supplies complete graph, parameter, and latent truth and primarily tests self-consistency with its generator. A semi-synthetic benchmark inserts known structure or interventions into realistic covariates or learned dynamics. Some real benchmarks offer interventional or physical ground truth. Many observed datasets supply only predictive validation and lack a complete true graph. Robust evaluation uses several families and deliberate assumption violations.

###### Self-compatibility without ground truth

Self-compatibility asks whether causal conclusions remain coherent when the same method is applied to different variable subsets or related views of one dataset. It is a falsification or stability check available on observed data. Ground-truth precision and recall require a reference graph.

###### Modeling fidelity and structure recovery

A model can reproduce observed distributions with the wrong graph. It can also recover important edges and fit some distributions poorly. Event-log simulation comparisons and edge-recovery metrics therefore test different claims.

##### 11. Related structure-learning and time-series discovery terms

###### Greedy Equivalence Search, GES

GES is a score-based search over Markov equivalence classes. It greedily adds and then removes equivalence-class edges and is consistent under suitable score, faithfulness, and large-sample assumptions. Static GES on lag-expanded data provides none of the multiscale latent or coherence semantics of N²DBN unless those properties are added explicitly.

###### Max–Min Hill Climbing, MMHC

MMHC is a hybrid learner: a max–min conditional-independence phase proposes each node's candidate neighborhood, then greedy score-based hill climbing orients and selects among candidates. N²DBN's default local forward–backward search resembles its score-search stage and uses a model-specific normalized allocation score.

###### NOTEARS and DYNOTEARS

NOTEARS converts DAG learning into continuous optimization using a smooth exact acyclicity constraint. DYNOTEARS extends this idea to contemporaneous and lagged linear time-series relations. Every learned observed edge in N²DBN has positive lag, which makes each unrolled learned graph acyclic without a global acyclicity constraint.

###### PC, PCMCI, PCMCI+, LPCMCI, and RPCMCI

PC is a constraint-based causal discovery algorithm. PCMCI adapts conditional-independence testing to autocorrelated time series using parent selection and momentary conditional independence tests. PCMCI+ adds contemporaneous discovery under causal sufficiency; LPCMCI allows latent confounding and represents partial ancestral information; RPCMCI alternates regime reconstruction with regime-dependent causal discovery. Their edge semantics and assumptions differ from N²DBN's latent-factor-mediated, positive-lag predictive graph.

###### VAR-LiNGAM

VAR-LiNGAM combines a vector autoregression for lagged effects with a linear non-Gaussian acyclic model for instantaneous effects. Non-Gaussian independent disturbances can identify orientations unavailable under Gaussian observational equivalence. N²DBN excludes observed contemporaneous edges and represents same-epoch co-movement with latent state.

###### Granger test and neural Granger causality

A classical Granger test asks whether past values of one series improve prediction of another, often in a VAR. Neural Granger methods use structured sparsity on MLP or RNN inputs to recover nonlinear predictive relations and longer histories. Both remain predictive notions unless assumptions justify a structural causal reading.

###### Change-point graph and regime-dependent graph

Non-stationary DBN, TV-DBN, and causal-regime methods may let the edge set itself switch at change points or evolve smoothly. N²DBN's current learned graph is resolution-specific and shared through time. Its discrete regime changes local dynamics; graph support remains fixed across epochs. These are different forms of non-stationarity.

###### Benchmark suite and external gold standard

DREAM/GeneNetWeaver and SERGIO provide simulated gene-regulatory ground truth; TimeGraph, CausalTime, CausalDynamics, Causal Chambers, and CausalRivers cover different combinations of realism, intervention, lag, confounding, and non-stationarity. A benchmark name alone leaves identifiability unspecified. Evaluation must document how edges, lags, interventions, sampling, and hidden variables define its gold standard.

###### Software toolbox and statistical method

pgmpy, bnlearn, causal-learn, gCastle, PyTetrad, and Benchpress are software libraries or benchmarking infrastructure. Each provides multiple discovery algorithms or evaluation components. Results should name the exact algorithm, score or test, preprocessing, hyperparameters, version, and toolbox.

##### 12. Important distinctions and cautions

###### Hidden variable and missing value

A hidden variable is unobserved by design. A missing value is an absent realization of a variable observed in at least some cases. Their scientific meanings and missingness assumptions differ even if both are integrated out computationally.

###### Filtering and smoothing

Filtering uses evidence only through $t$; smoothing uses later evidence. Smoothing cannot support a strict real-time decision at $t$ without information that was unavailable then.

###### Particle filtering and MCMC

Particle filtering advances through time and approximates successive filtering distributions. MCMC iterates toward a fixed target distribution, often a trajectory or graph posterior. Particle MCMC combines the families.

###### Dynamic graph and time-varying graph

A standard DBN has a repeated template for a dynamic process. A time-varying DBN lets parameters, structure, or both change. The word “dynamic” alone therefore gives no information about time variation in the graph.

###### Association, prediction, Granger causality, and structural causality

Conditional dependence is association; predictive improvement is a forecasting property; Granger causality is lagged predictive content; structural causality supports interventions or counterfactuals under stronger assumptions. Each term has a distinct definition.

###### Aggregation and marginalization

Aggregation maps fine values to a coarse value, such as a sum. Marginalization removes variables from a distribution by integration or summation. Constructing a coarse-resolution DBN generally requires aggregation rules and new conditional distributions beyond marginalization of a fine-resolution DBN.

###### Statistical significance and practical equivalence

Failure to reject a difference provides insufficient evidence for equivalence. A “no significant difference” claim should report the test, estimand, sample size, effect size, and uncertainty; equivalence requires a predefined tolerance and an equivalence-oriented procedure.

##### 13. Sources checked and audit scope

The current project definitions were checked directly against:

- [`n2dbn.tex`](docs/paper/n2dbn.tex): paper-level claims, model positioning, and benchmark declarations;
- [`sec_synopsis.tex`](docs/paper/sec_synopsis.tex): fold/unfold ladder, fibers, shadow/detail, bands, cascade, and bidirectional messages;
- [`sec_model.tex`](docs/paper/sec_model.tex): event projection, typed edges, top-down factorization, exact coherence, allocation support, cross-scale coupling, participant effects, and forecasting;
- [`sec_learning.tex`](docs/paper/sec_learning.tex): variational family, forward–backward/Kalman inference, M-step blocks, profiled structure scores, search, shift confirmation, and monotonicity guard;
- [`sec_identifiability.tex`](docs/paper/sec_identifiability.tex): slot-duration semantics, reconstruction safeguards, exact invariances, identified recovery targets, assumptions, and failure modes; and
- [`sec_benchmark.tex`](docs/paper/sec_benchmark.tex): identifiability screening, baselines, per-scale metrics, oracle rows, sortability, and evidence-comparison protocol.

Every one of the 107 files in [`docs/paper/pdfs/`](docs/paper/pdfs/) was text-extracted and inspected individually. Relevant concepts were incorporated by topic. The glossary remains organized by concepts and includes a separate source list. The following files contain works that differ from their citation-style filenames and were excluded from the evidence base: `cook2013casas.pdf`, `franzin2017bnstruct.pdf`, `klasnja2019heartsteps.pdf`, `marbach2010revealing.pdf`, `pratapa2020beeline.pdf`, `ramsey2023pytetrad.pdf`, `runge2018pitfalls.pdf`, `runge2019pcmci.pdf`, `schaffter2011gnw.pdf`, `teixeira2023yeastract.pdf`, `vandenbulcke2006syntren.pdf`, and `zhang2023improvedmcmc.pdf`. These files should be replaced before they are cited as the works named by their filenames.

Especially relevant papers in that folder include:

- [Friedman, Murphy, and Russell on Structural EM](docs/paper/pdfs/friedman1998dbnstructure.pdf), [Chickering on GES](docs/paper/pdfs/chickering2002ges.pdf), and [Tsamardinos et al. on MMHC](docs/paper/pdfs/tsamardinos2006mmhc.pdf): expected-score structure learning, equivalence-class search, and hybrid search;
- [Robinson and Hartemink on non-stationary DBNs](docs/paper/pdfs/robinson2010nsdbn.pdf), [Song et al. on TV-DBNs](docs/paper/pdfs/song2009tvdbn.pdf), [Grzegorczyk and Husmeier on continuous NH-DBNs](docs/paper/pdfs/grzegorczyk2009nscontinuous.pdf), and [Kamalabad and Grzegorczyk on edge-wise coupling](docs/paper/pdfs/kamalabad2020edgewise.pdf): regimes, change points, smooth temporal variation, and parameter sharing;
- [Pamfil et al. on DYNOTEARS](docs/paper/pdfs/pamfil2020dynotears.pdf), [Runge on PCMCI+](docs/paper/pdfs/runge2020pcmciplus.pdf), [Gerhardus and Runge on LPCMCI](docs/paper/pdfs/gerhardus2020lpcmci.pdf), [Saggioro et al. on RPCMCI](docs/paper/pdfs/saggioro2020rpcmci.pdf), and [Shojaie and Fox on Granger causality](docs/paper/pdfs/shojaie2022granger.pdf): alternative time-series discovery assumptions and edge semantics;
- [Wickramasuriya et al. on MinT reconciliation](docs/paper/pdfs/wickramasuriya2019mint.pdf): coherent hierarchical forecasts and the distinction between coherent generation and post-hoc reconciliation;
- [Peters and Bühlmann on SID](docs/paper/pdfs/peters2015sid.pdf), [Constantinou on BSF](docs/paper/pdfs/constantinou2019bsf.pdf), [Davis and Goadrich on PR versus ROC](docs/paper/pdfs/davis2006prroc.pdf), and [Reisach et al. on varsortability](docs/paper/pdfs/reisach2021varsortability.pdf) and [$R^2$-sortability](docs/paper/pdfs/reisach2023r2sortability.pdf): graph metrics and benchmark artifacts; and
- [CausalTime](docs/paper/pdfs/cheng2024causaltime.pdf), [CausalDynamics](docs/paper/pdfs/herdeanu2025causaldynamics.pdf), [Causal Chambers](docs/paper/pdfs/gamella2025causalchambers.pdf), [CausalRivers](docs/paper/pdfs/stein2025causalrivers.pdf), and [CausalCompass](docs/paper/pdfs/yi2026causalcompass.pdf): complementary synthetic, semi-synthetic, physical, real-world, and robustness-oriented benchmark designs.

- [Murphy (2002), *Dynamic Bayesian Networks: Representation, Inference and Learning*](docs/Dynamic_Bayesian_Networks_Representation.pdf): DBNs, 2TBNs, state-space queries, exact/approximate inference, particle methods, and Structural EM.
- [Kitson et al. (2023), *A survey of Bayesian Network structure learning*](docs/A%20survey%20of%20Bayesian%20Network%20structure%20learning.pdf): graph semantics, equivalence, learning families, scores, and evaluation.
- [Kungurtsev et al., *Learning Dynamic Bayesian Networks from Data*](docs/Learning%20Dynamic%20Bayesian%20Networks%20from%20Data.pdf): parameterization, trajectories, scoring, acyclicity, and joint learning.
- [Leão et al. (2021), DBNs with time-dependent and time-independent data](docs/Learning%20dynamic%20Bayesian%20networks%20from%20time-dependent.pdf): static covariates, Markov order, stationary/non-stationary transition networks, and restricted structure search.
- [Grzegorczyk and Husmeier (2009)](docs/NIPS-2009-non-stationary-continuous-dynamic-bayesian-networks-Paper.pdf), [Song, Kolar, and Xing (2009)](docs/NIPS-2009-time-varying-dynamic-bayesian-networks-Paper.pdf), and [Shafiee Kamalabad and Grzegorczyk (2020)](docs/Non-homogeneous%20dynamic%20Bayesian%20networks.pdf): change-point, time-varying, and coupled NH-DBNs.
- [Saria, Nodelman, and Koller (2007)](docs/Reasoning%20at%20the%20Right%20Time%20Granularity.pdf) and [Kang, Ganguly, and Kolaczyk (2022)](docs/Dynamic%20Networks%20with%20Multi-scale%20Temporal%20Structure.pdf): distinct meanings of multi-scale temporal modeling.
- [Groß et al. (2019)](docs/Representing%20dynamic%20biological%20networks%20with%20multi-scale%20probabilistic%20models.pdf) and [Chen et al. (2026), DynaMamba](docs/Multi-scale%20dynamic%20interacting%20Mamba%20network.pdf): mechanism-level and neural multi-scale meanings used only for comparison.
- [Das et al. (2023), coherent probabilistic forecasting](docs/A%20Top-Down%20Approach%20to%20Hierarchically%20Coherent%20Probabilistic%20Forecasting.pdf): hierarchy, coherence, and reconciliation.
- [Chen et al. (2026), Copula-DBN](docs/A%20Copula-based%20Dynamic%20Bayesian%20Network.pdf) and [Wang and Kim (2026), long-term structural monitoring](docs/Dynamic%20Bayesian%20networks%20for%20long-term.pdf): examples of copula CPDs, particle filtering, and time-varying state tracking.
- [Bøttcher and Dethlefsen, *deal: A Package for Learning Bayesian Networks*](docs/Learning%20Bayesian%20Networks.pdf) and [Korre (2021), mixed static/dynamic Bayesian networks](docs/new%20dbns.pdf): mixed-variable BNs, Bayesian scores, and contemporaneous versus lagged structure.
- [Current N²DBN manuscript](docs/paper/n2dbn.tex): project notation, exact top-down construction, learning, identifiability, forecasting, and evaluation protocol.
