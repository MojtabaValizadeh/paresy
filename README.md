<div class="markdown-google-sans">
<h1><strong>PaRESy</strong></h1>
</div>

This repo contains the code and other artifacts for the  paper.

> **Search-Based Regular Expression Inference on a GPU**

by Mojtaba Valizadeh and Martin Berger. The paper  is available at 
- https://dl.acm.org/doi/10.1145/3591274 (publisher)
- https://arxiv.org/abs/2305.18575 (draft)

The original code, submitted to PLDI 2023, is available using the tag [`v0.1`](https://github.com/MojtabaValizadeh/paresy/releases/tag/v0.1), or using commit `6aa87ae`.
See also 
- https://github.com/MojtabaValizadeh/ltl-learning-on-gpus

or follow-up work that trades off scalability against minimality.

## Introduction

In this work, the goal is to find a `precise` and `minimal` regular expression (RE) that accepts a given set of positive strings and rejects a given set of negative ones. To accomplish this, we developed an algorithm called `PaRESy`, Parallel Regular Expression Synthesiser, which is implemented in two codes: one for CPU using C++, and another for GPU using Cuda C++. By doing this, we could measure the speed-up for the most challenging examples.

In this version of the work, we use a simple grammar for the REs:

```
R ::= Φ|ε|a|R?|R*|R.R|R+R
```
Regarding the minimality, we need to use a cost function that maps every constructor in the RE to a positive integer. By summing up the costs of all the constructors, we can obtain the overall cost of the RE. This cost function helps us to avoid overfitting and returning the trivial RE that is the union of the positive strings.

**Example:**
- Positive: [`""`, `"010"`, `"1000"`, `"0101"`, `"1010"`, `"0010"`]
- Negative: [`"001"`, `"1011"`, `"111"`]

**Note:** In this work, we use `""` to represent the `epsilon (ε)` as the empty string.

We aim to find a regular expression (RE) that accepts all strings in the positive set and rejects all strings in the negative set. However, there could be an infinite number of such REs, and we want to identify the minimal one based on a given cost function. This cost function assigns positive integers to each constructor, including the `alphabet`, `question mark`, `star`, `concatenation`, and `union`. Let us assume that the costs of these constructors are given as [`c1`, `c2`, `c3`, `c4`, `c5`]. Based on the different costs, we can generate various REs, as follows.

-   [`1`, `1`, `1`, `1`, `1`]   --->    `(0+101?)*`
-   [`1`, `5`, `1`, `1`, `5`]   --->    `(01)*(1*0)*`

**Note:** In this work, we use the `+` symbol to represent the `union` constructor, which is commonly denoted by `|` in standard libraries.

In this simple example, both REs are `precise` (i.e., accepts all positive and rejects all negative examples) and `minimal` w.r.t their own cost functions. We can observe that by increasing the costs of `question mark` and `union` in the second cost function, the resulting RE contains fewer instances of these constructors. However, to compensate for this, the regular expression tends to use more stars, which are cheaper in this particular cost function.


## Colab Notebook
This work is presented on a Google Colab notebook platform, which automatically clones this GitHub repository and comes fully equipped with all necessary dependencies and libraries, eliminating the need for additional installations. You can run the scripts by clicking on the designated buttons and adjusting inputs as needed.

To access the notebook, please use the link below and follow the instructions provided.

- [https://colab.research.google.com/github/MojtabaValizadeh/paresy/blob/master/paresy.ipynb](https://colab.research.google.com/github/MojtabaValizadeh/paresy/blob/master/paresy.ipynb)
