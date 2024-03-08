import matplotlib.pyplot as plt 
import pandas as pd 
import numpy 
from treebanks import conllu_corpus, train_corpus, test_corpus


def graph_precision(path, save_path, title, metric):
    df = pd.read_csv(path)

    df.drop(columns=df.columns[0], inplace=True)

    plt.figure(figsize=(10,8))

    plt.bar(df.columns, df.iloc[0], color='#42a7f5')

    plt.title(title)
    plt.xlabel('Tags')
    plt.ylabel(metric)
    plt.xticks(rotation=90)

    plt.savefig(save_path)

def graph_corpus_lengths():
    languages = ['en','sv','ko','ch']

    train_lengths = {}
    test_lengths = {} 

    for lang in languages:
        train_lengths[lang] = len(conllu_corpus(train_corpus(lang)))
        test_lengths[lang] = len(conllu_corpus(test_corpus(lang)))


    bins = ['English','Swedish','Korean','Chinese']
    values1 = list(train_lengths.values())
    values2 = list(test_lengths.values())

    bars1 = plt.bar(bins, values1, label='Training Data')
    bars2 = plt.bar(bins, values2, bottom=values1, label='Testing Data')

    # Adding labels to each bar
    for bar1, bar2, val1, val2 in zip(bars1, bars2, values1, values2):
        plt.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() / 2, str(val1), ha='center', va='center', color='white', fontsize=12)
        plt.text(bar2.get_x() + bar2.get_width() / 2, bar1.get_height() + bar2.get_height() / 2, str(val2), ha='center', va='center', color='black', fontsize=12)


    # Adding labels and title
    plt.xlabel('Languages', fontsize=12)
    plt.ylabel('Number of Sentences', fontsize=12)
    plt.title('Test & Training Data Size for Each Language', fontsize=12)
    plt.xticks(fontsize=14)

    # Adding legend
    plt.legend(fontsize=12)

    # Show plot
    plt.savefig('./Figures/Corpus/corpus_lengths.png')


def tag_freq(lang, lang_label):
    train_sents = conllu_corpus(train_corpus(lang))

    freqs = {}

    for sent in train_sents:
        for token in sent:
            if token['upos'] in freqs:
                freqs[token['upos']] += 1
            else:
                freqs[token['upos']] = 1

    plt.bar(freqs.keys(), freqs.values(), color='blue')
    plt.title(f'Tag Frequencies For {lang_label}')
    plt.xlabel('Tags')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, fontsize=10)
    plt.savefig(f'./Figures/TagFreqs/tag_freq_{lang}.png')


# graph_corpus_lengths()   
    
tag_freq('ko', 'Korean')
tag_freq('en', 'English')
tag_freq('ch', 'Chinese')
tag_freq('sv', 'Swedish')

# ENGLISH
# graph_precision('./Data/precision_eager_en.csv', './Figures/PrecisionRecall/precision_eager_en.png','Eager Algorithm Precision for English', 'Precision')
# graph_precision('./Data/recall_eager_en.csv', './Figures/PrecisionRecall/recall_eager_en.png', 'Eager Algorithm Recall for English', 'Recall')

# graph_precision('./Data/precision_viterbi_en.csv', './Figures/PrecisionRecall/precision_viterbi_en.png', 'Viterbi Algorithm Precision for English', 'Precision')
# graph_precision('./Data/recall_viterbi_en.csv', './Figures/PrecisionRecall/recall_viterbi_en.png', 'Viterbi Algorithm Recall for English', 'Recall')

# graph_precision('./Data/precision_impt_en.csv', './Figures/PrecisionRecall/precision_impt_en.png', 'IMPT Algorithm Precision for English', 'Precision')
# graph_precision('./Data/recall_impt_en.csv', './Figures/PrecisionRecall/recall_impt_en.png', 'IMPT Algorithm Recall for English', 'Recall')

# KOREAN
# graph_precision('./Data/precision_eager_ko.csv', './Figures/PrecisionRecall/precision_eager_ko.png','Eager Algorithm Precision for Korean', 'Precision')
# graph_precision('./Data/recall_eager_ko.csv', './Figures/PrecisionRecall/recall_eager_ko.png', 'Eager Algorithm Recall for Korean', 'Recall')

# graph_precision('./Data/precision_viterbi_ko.csv', './Figures/PrecisionRecall/precision_viterbi_ko.png', 'Viterbi Algorithm Precision for Korean', 'Precision')
# graph_precision('./Data/recall_viterbi_ko.csv', './Figures/PrecisionRecall/recall_viterbi_ko.png', 'Viterbi Algorithm Recall for Korean', 'Recall')

# graph_precision('./Data/precision_impt_ko.csv', './Figures/PrecisionRecall/precision_impt_ko.png', 'IMPT Algorithm Precision for Korean', 'Precision')
# graph_precision('./Data/recall_impt_ko.csv', './Figures/PrecisionRecall/recall_impt_ko.png', 'IMPT Algorithm Recall for Korean', 'Recall')