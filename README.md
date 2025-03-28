#  Parts Of Speech Tagging Algorithms 
Author : Jimmy Zhang (jimmyjz1127)  
Date : March 6, 2024

## 1. Description
This project implements three Hidden Markov Model algorithms for the task of Parts-Of-Speech tagging using the Universal Dependencies corpus for the English, Swedish, Chinese, and Korean languages. The algorithms are as follows 
1) Eager Algorithm
2) Viterbi Algorithm
3) Posterior Decoding Algorithm


All algorithms are based on first order HMMs with the following assumptions:

1) The probability of an emission (word token) only depends on its corresponding POS tag :
   
$$\begin{align}
    P(w_{1}^{n}|t_{1}^n)=\prod_{i=1}^{n}P(w_{i}|t_{i})
\end{align}$$

3) The probability of a given state (tag) only depends on its immediate previous state
   
$$\begin{align}
    \large
    P(t_{1}^n) = (\prod_{i=1}^n P(t_i | t_{i-1}))P(t_{n+1} | t_n)
\end{align}$$


To account for the sparse data problem when capturing frequency distributions of emissions and transitions, a Witten-Bell probability distribution is to smooth and adjust estimations to account for events not yet observed through a principled redistribution of probablity mass from observed to unseen events.

## 2. To Install Necessary Dependencies 
`pip install nltk pandas pyarrow seaborn matplotlib conllu`

## 3. Execution Instructions 
1. To execute all algorithms on english language :  
    `python p1.py`
2. To specify a language run algorithms on :  
    `python p1.py [en | sv | ko | ...]`
3. To generate confusion matrix :  
    `python p1.py conf`   
    OR  
    `python p1.py [en | sv | ko | ...] conf` 


## 4. Files & Directories 
.  
├── `Figures/`            : evaluation visuals & plots  
├── `treebanks/`          : Universal Dependency treebanks  
├── `StarterCode/`        : provided code examples  
├── `p1.py`               : implementation of algorithms    
└── `treebanks.py`        : code for accessing treebanks 
