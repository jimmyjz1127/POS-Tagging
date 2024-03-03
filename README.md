#  CS5012 Deliverable 1 : POS Tagging 
## Matriculation ID : 190015412 
## Date : March 6, 2024

### To Install Necessary Dependencies 
pip install nltk pandas pyarrow seaborn matplotlib conllu 

### Execution Instrucitons 
1. To execute all algorithms on english language :  
    python p1.py 
2. To specify a language run algorithms on :  
    python p1.py [en | sv | ko | ...]
3. To generate confusion matrix :  
    python p1.py conf   
    OR  
    python p1.py en conf 


### Files & Directories 
.  
├── Figures/            : evaluation visuals & plots  
├── treebanks/          : Universal Dependency treebanks  
├── StarterCode/        : provided code examples  
├── p1.py               : implementation of algorithms    
└── treebanks.py        : code for accessing treebanks 