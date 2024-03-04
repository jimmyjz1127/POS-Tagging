import matplotlib.pyplot as plt 
import pandas as pd 
import numpy 



def graph_precision(path, save_path, title, metric):
    df = pd.read_csv(path)

    df.drop(columns=df.columns[0], inplace=True)


    plt.bar(df.columns, df.iloc[0], color='blue')

    plt.title(title)
    plt.xlabel('Tags')
    plt.ylabel(metric)
    plt.xticks(rotation=90)

    plt.savefig(save_path)


graph_precision('./Data/precision_eager_en.csv', './Figures/bar/precision_eager_en.png','Eager Algorithm Precision for English', 'Precision')
graph_precision('./Data/recall_eager_en.csv', './Figures/bar/recall_eager_en.png', 'Eager Algorithm Recall for English', 'Recall')

graph_precision('./Data/precision_viterbi_en.csv', './Figures/bar/precision_viterbi_en.png', 'Viterbi Algorithm Precision for English', 'Precision')
graph_precision('./Data/recall_viterbi_en.csv', './Figures/bar/recall_viterbi_en.png', 'Viterbi Algorithm Recall for English', 'Recall')

graph_precision('./Data/precision_impt_en.csv', './Figures/bar/precision_impt_en.png', 'IMPT Algorithm Precision for English', 'Precision')
graph_precision('./Data/recall_impt_en.csv', './Figures/bar/recall_impt_en.png', 'IMPT Algorithm Recall for English', 'Recall')