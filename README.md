#  Parts Of Speech Tagging Algorithms 
## Author : Jimmy Zhang (jimmyjz1127)
## Date : March 6, 2024

Implements 3 different Hidden Markov Model based POS Tagging Algorithms:
1) Naive Eager Algorithm
2) Viterbi Algorithm
3) Individual Most Probable Tag (Posterior Decoding) Algorithm

POS Tagging executed on Universal Dependency Corpus (for English, Swedish, Korean, and Chinese).

### To Install Necessary Dependencies 
`pip install nltk pandas pyarrow seaborn matplotlib conllu`

### Execution Instructions 
1. To execute all algorithms on english language :  
    `python p1.py`
2. To specify a language run algorithms on :  
    `python p1.py [en | sv | ko | ...]`
3. To generate confusion matrix :  
    `python p1.py conf`   
    OR  
    `python p1.py [en | sv | ko | ...] conf` 


### Files & Directories 
.  
├── `Figures/`            : evaluation visuals & plots  
├── `treebanks/`          : Universal Dependency treebanks  
├── `StarterCode/`        : provided code examples  
├── `p1.py`               : implementation of algorithms    
└── `treebanks.py`        : code for accessing treebanks 
