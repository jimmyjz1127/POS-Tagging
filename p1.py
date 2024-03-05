from nltk import FreqDist, WittenBellProbDist, LaplaceProbDist, KneserNeyProbDist
import os
import numpy as np 
import pandas 
import sys 
from treebanks import conllu_corpus, train_corpus, test_corpus
from math import exp, log
import re
import pprint
import json
from nltk.util import ngrams
from sys import float_info
from math import log, exp
import time 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 


class Tagger():
   
    def __init__(self, lang):
        self.min_log_prob = -float_info.max
        self.lang = lang

        # Import treebanks 
        self.train_sents = conllu_corpus(train_corpus(lang))
        self.test_sents = conllu_corpus(test_corpus(lang))

        # pre-process train and test sentences 
        self.train_sents = self.preprocess_sentences(self.train_sents)
        self.test_sents =  self.preprocess_sentences(self.test_sents)

        # hard code list of tags in case of sparse corpus 
        self.tags =  [ 'X', 'SYM', 'AUX', 'PROPN', 'DET', 'INTJ', 'NUM', 'PUNCT', 'CCONJ', 'SCONJ', 'ADV', 'ADP', 'ADJ', 'PART', 'NOUN', 'VERB', 'PRON', 'START', 'END']
        
        # All tags excluding start of sentence and end of sentence tags
        self.tags_none =  ['AUX', 'PROPN', 'SYM', 'DET', 'INTJ', 'NUM', 'PUNCT', 'X', 'CCONJ', 'SCONJ', 'ADV', 'ADP', 'ADJ', 'PART', 'NOUN', 'VERB', 'PRON']

        # get smoothed emission and transisions (bigram)
        self.emissions = self.init_smoothed_emission_dist(self.train_sents, self.tags)
        self.transitions = self.init_smoothed_transition_dist(self.train_sents)



    def preprocess_sentences(self, sentences):
        '''
            Takes corpus sentence in conllu form and adds start and end of sentence markers to each sentence
            Modifies each token into form  : (word, tag)

            @param sentences : list of sentences in conllu form 
            @returns list of sentences of form : [[(word, tag)]]
        '''

        sents = [] # [(form, upos)]
        
        # Convert conllu format to tuples of form (id, word, tag)
        for sentence in sentences:  
            sent = [('<s>', 'START')]

            for token in sentence:
                sent.append((token['form'].lower(), token['upos']))

            sent.append(('</s>', 'END'))
            sents.append(sent)

        return sents

    def init_smoothed_emission_dist(self, sentences, tags):
        """
            Calculates smoothed distribution of emission probabilities P(word | tag)

            @param sentences : list of sentences [[(word, tag)]]
            @returns : emission probability distribution (with Witten-Bell smoothing)
        """

        distribution = {}

        for tag in self.tags_none:
            words = [w for sentence in sentences for (w, t) in sentence if t == tag]
            distribution[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

        return distribution
    
    
    def init_smoothed_transition_dist(self, sentences):
        '''
            Calculates smoothed distribution of transition probabilities P( tag[i] | tag[i-1] )

            @param sentences : list of sentences [[(word, tag)]]
            @return : transition probability trellis (with witten-bell smoothing)
        '''
        bigrams = [] 

        for sentence in sentences:
            tags = [obj[1] for obj in sentence]
            bigrams += ngrams(tags, 2)

        distribution = {}

        for tag in self.tags:
            succeeding_tags = [tag2 for tag1,tag2 in bigrams if tag1 == tag and tag2 != 'START' and tag1 != 'END']
            distribution[tag] = WittenBellProbDist(FreqDist(succeeding_tags), bins=1e5) 

        return distribution 


        

    def eager_tag(self, sentence):
        """ 
            Tags a sentence using eager algorithm

            @param sentence : sentence to tag in form of [(word, tag)]
            @returns : a new sentence list with predicted tags [(word, predicted tag)]
        """

        pred_sent = [sentence[0]] # initialize with start-of-sentence
        prev_tag='START'
        for token in sentence[1:-1]:
            word = token[0] # the word to predict tag for 

            # list of all possible (tag, emission_prob * transistion_prob) for the given word
            # probs = [(tag, self.emissions[tag].logprob(word) + self.transitions.logprob((prev_tag, tag))) for tag in self.tags_none]
            probs = [(tag, self.emissions[tag].logprob(word) + self.transitions[prev_tag].logprob(tag)) for tag in self.tags_none]

            # tag with highest probability 
            max_prob_tag = max(probs, key=lambda obj:obj[1])[0]

            pred_sent.append((word,max_prob_tag))
            prev_tag = max_prob_tag
            # NOTE : we are leaving out end-of-sentence marker
        pred_sent.append(sentence[-1])

        return pred_sent
    
    def viterbi_tag(self, sentence):
        """
            Tags a sentence using the viterbi algorithm 

            @param sentence : the sentence to tag in form [(word, tag)]
            @return : a new sentence list with predicted tags [(word, predicted tag)]
        """
       
        viterbi = []
        backpointer = []

        # Initialize "viterbi[q,1] for all q"
        initial = {}
        backpointer_initial = {}
        for tag in self.tags_none:
            initial[tag] = self.transitions["START"].logprob(tag) + self.emissions[tag].logprob(sentence[1][0])
            backpointer_initial[tag] = 'START'
        viterbi.append(initial)
        backpointer.append(backpointer_initial)

        # Intermediary "viterbi[q,i] for i=2,...,n"
        for i in range(2, len(sentence)):
            token = sentence[i][0]
            probs = {}
            backpointers = {}

            for tag in self.tags_none:
                max_prob, best_prev_tag = max(
                    [(viterbi[-1][prev_tag] +  self.transitions[prev_tag].logprob(tag) + self.emissions[tag].logprob(token), prev_tag) for prev_tag in self.tags_none],
                    key=lambda x: x[0]
                )
                probs[tag] = max_prob
                backpointers[tag] = best_prev_tag

            viterbi.append(probs)
            backpointer.append(backpointers)

        # Finish "viterbi[qf, n+1]"
        _, final_tag = max([(viterbi[-1][tag] +  self.transitions[tag].logprob('END'), tag) for tag in self.tags_none], key=lambda x: x[0])


        # Backtrack
        pred_tags = ['END']
        current_tag = final_tag
        for i in range(len(sentence) - 2, 0, -1):
            current_tag = backpointer[i][current_tag]
            pred_tags.insert(0, current_tag)
        pred_tags.insert(0, 'START')

        # Pair words with predicted tags, excluding START and END
        pred_sent = [(sentence[i][0], pred_tags[i]) for i in range(len(sentence))]

        return pred_sent

    
    def IMPT_tag(self, sentence):
        """
            Tags a sentence using the "Individually Most Probable Tag" method 

            @param sentence : the sentence to tag of form [(word, tag)] 
            @return : a new sentence list with predicted tags [(word, predicted tag)]
        """
        forward = []
        backward = []

        # INITIAL 
        # forward[q,1] for all q
        # backward[q,n] for all q
        initial_f = {}
        initial_b = {}
        
        for tag in self.tags_none:
            initial_f[tag] = self.transitions['START'].logprob(tag) + self.emissions[tag].logprob(sentence[1][0])
            initial_b[tag] = self.transitions[tag].logprob('END')
        forward.append(initial_f)
        backward.append(initial_b)

        # INTERMEDIARY 
        # forward[q,i] for i = 2,...,n and all q
        # backward[q,i] for i = n-1,...,1 and all q
        for i in range(2, len(sentence) - 1):
            intermed_f = {}
            intermed_b = {}
            token_f = sentence[i][0]
            token_b = sentence[len(sentence) - i][0]

            for tag in self.tags_none:
                inner_f = [forward[-1][prev_tag] + self.transitions[prev_tag].logprob(tag) + self.emissions[tag].logprob(token_f) for prev_tag in self.tags_none]
                inner_b = [backward[-1][next_tag] + self.transitions[tag].logprob(next_tag) + self.emissions[next_tag].logprob(token_b) for next_tag in self.tags_none]
                intermed_f[tag] = self.logsumexp(inner_f)
                intermed_b[tag] = self.logsumexp(inner_b)

            forward.append(intermed_f)
            backward.append(intermed_b)

        # FINAL
        # forward[qf, n+1] 
        # backward[q0, 0]
        final_f = {}
        final_b = {}
        final_f['END'] = self.logsumexp([forward[-1][prev_tag] + self.transitions[prev_tag].logprob('END') for prev_tag in self.tags_none])
        final_b['START'] = self.logsumexp([backward[-1][next_tag] + self.transitions['START'].logprob(next_tag) + self.emissions[next_tag].logprob(sentence[1][0]) for next_tag in self.tags_none])
        forward.append(final_f)
        backward.append(final_b)

        # reverse backwards matrix to align with forward matrix 
        backward.reverse() 

        # cut off columns for start and end of sentence markers 
        backward = backward[1:]
        forward = forward[:-1]  

        # Finish
        pred_sent = [("<s>", "START")]
        for i in range(0, len(sentence) - 2):
            f_col = forward[i]
            b_col = backward[i]
            combined = self.combine_dicts(f_col, b_col)
            max_tag = max(combined.items(), key = lambda obj:obj[1])[0]
            pred_sent.append((sentence[i + 1][0], max_tag))
        pred_sent.append(("</s>", "END"))
                
        return pred_sent

    def forward_tag(self, sentence):
        """
            Tags a sentence using the "Individually Most Probable Tag" method 

            @param sentence : the sentence to tag of form [(word, tag)] 
            @return : a new sentence list with predicted tags [(word, predicted tag)]
        """
        forward = []

        # INITIAL 
        # forward[q,1] for all q
        # backward[q,n] for all q
        initial_f = {}
        initial_b = {}
        
        for tag in self.tags_none:
            initial_f[tag] = self.transitions['START'].logprob(tag) + self.emissions[tag].logprob(sentence[1][0])
        forward.append(initial_f)

        # INTERMEDIARY 
        # forward[q,i] for i = 2,...,n and all q
        # backward[q,i] for i = n-1,...,1 and all q
        for i in range(2, len(sentence) - 1):
            intermed_f = {}
            token_f = sentence[i][0]

            for tag in self.tags_none:
                inner_f = [forward[-1][prev_tag] + self.transitions[prev_tag].logprob(tag) + self.emissions[tag].logprob(token_f) for prev_tag in self.tags_none]
                intermed_f[tag] = self.logsumexp(inner_f)

            forward.append(intermed_f)

        # FINAL
        # forward[qf, n+1] 
        # backward[q0, 0]
        final_f = {}
        final_f['END'] = self.logsumexp([forward[-1][prev_tag] + self.transitions[prev_tag].logprob('END') for prev_tag in self.tags_none])
        forward.append(final_f)

        forward = forward[:-1]  

        # Finish
        pred_sent = [("<s>", "START")]
        for i in range(0, len(sentence) - 2):
            f_col = forward[i]
            max_tag = max(f_col.items(), key = lambda obj:obj[1])[0]
            pred_sent.append((sentence[i + 1][0], max_tag))
        pred_sent.append(("</s>", "END"))
                
        return pred_sent

    def run(self, algo):
        """
            For applying an HMM tagging algorithm to all sentences of a the test corpus 

            @param algo : integer specifying which algorithm to use 
                1 : eager algorithm
                2 : viterbi algorithm
                3 : forward-back (individually most probable tag) algorithm

            @returns : the new list of sentences with predicted tags 
        """

        sentences = self.test_sents
        result = []
    
        if algo == 1:
            start = time.time()
            for sentence in sentences:
                result.append(self.eager_tag(sentence))
            duration = time.time() - start
            return result, duration
        elif algo == 2:
            start = time.time()
            for sentence in sentences:
                result.append(self.viterbi_tag(sentence))
            duration = time.time() - start
            return result, duration
        elif algo == 3:
            start = time.time()
            for sentence in sentences:
                result.append(self.IMPT_tag(sentence))
            duration = time.time() - start
            return result, duration
        else :
            return None


    def combine_dicts(self, dict1, dict2):
        """
            For combining the values of two dictionaries into a single dictionary 
            (for combining forward and backward tables of forward-backward algorithm)

            @param dict1 : dictionary {String : Double}
            @param dict2 : dictionary {String : Double}
            @returns : combined dictionary {String : Double}
        """
        result = {}
        for key in dict1.keys():
            result[key] = dict1[key] + dict2[key]
        return result


    def logsumexp(self, vals):
        """
            For summing a list of log probabilities 

            @param vals : list of log probabilities [double]
            @returns : the log sum (double)
        """
        if len(vals) == 0:
            return self.min_log_prob
        m = max(vals)
        if m == self.min_log_prob:
            return self.min_log_prob
        else:
            return m + log(sum([exp(val - m) for val in vals]))

    def calc_accuracy(self, predictions, algType):
        """
            Calculates the accuracy of an algorithm by comparing its predicted tags against actual tags 

            @param predictions : the sentences with predicted tags of form [[(word, tag)]]
            @returns : double representing (# of correct tags)/total
        """

        num_correct = 0.0
        total = 0.0

        true_pos = {k:0 for k in self.tags}
        false_pos = {k:0 for k in self.tags}
        false_neg = {k:0 for k in self.tags}

        for i in range(0, len(self.test_sents)):
            pred_sent = predictions[i]
            label_sent = self.test_sents[i]

            for j in range(0, len(pred_sent)):
                pred_tag = pred_sent[j][1]
                label_tag = label_sent[j][1]

                if pred_tag == label_tag:
                    num_correct += 1.0
                    true_pos[pred_tag] += 1
                else:
                    false_pos[pred_tag] += 1
                    false_neg[label_tag] += 1

                total += 1.0

        precision = {}
        recall = {}
        for tag in self.tags:
            if ((true_pos[tag] + false_pos[tag]) > 0) : precision[tag] = true_pos[tag] / (true_pos[tag] + false_pos[tag])
            else : precision[tag] = 0
            if ((true_pos[tag] + false_neg[tag]) > 0) : recall[tag] = true_pos[tag] / (true_pos[tag] + false_neg[tag])
            else : recall[tag] = 0

        precision_df = pd.DataFrame([precision])
        recall_df = pd.DataFrame([recall])

        precision_df.to_csv(f'./Data/precision_{algType}_{self.lang}.csv')
        recall_df.to_csv(f'./Data/recall_{algType}_{self.lang}.csv')

        accuracy = num_correct/total
        return accuracy

    def calc_confusion_matrix(self, predictions, title):
        """
            For generating confusion matrix data and heatmap 

            @param predicitons : the predicted tags for test corpus 
            @param title : string title to display on visual
        """

        freq_matrix = pd.DataFrame(0, index=self.tags, columns=self.tags)

        for i in range(0, len(self.test_sents)):
            pred_sent = predictions[i]
            label_sent = self.test_sents[i]

            for j in range(0, len(label_sent)):
                pred_tag = pred_sent[j][1]
                label_tag = label_sent[j][1]

                freq_matrix.loc[pred_tag, label_tag] += 1

        sum_row = freq_matrix.sum(axis=1)
        freq_matrix = freq_matrix.div(sum_row, axis=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(freq_matrix, fmt='.2f', cmap='Reds', annot=True)
        plt.title(f'{title} Confusion Matrix [{self.lang}]')
        image_name = re.sub(r"\s", "", title)
        plt.savefig(f'./Figures/{image_name}_{self.lang}.png')



def main(lang, flag):
    tagger = Tagger(lang)

    print('\n======================================\n')

    # Run Eager Algorithm
    result, duration = tagger.run(1)
    print('EAGER Algorithm')
    print('  Accuracy     :', tagger.calc_accuracy(result, "eager"))
    print('  Time Elapsed :', str(duration) + 's')
    if (flag) : tagger.calc_confusion_matrix(result, 'Eager Algorithm')

    print('\n======================================\n')

    # Run Viterbi Algorithm
    result, duration = tagger.run(2)
    print('VITERBI Algorithm')
    print('  Accuracy     :', tagger.calc_accuracy(result, 'viterbi'))
    print('  Time Elapsed :', str(duration) + 's')
    if (flag) : tagger.calc_confusion_matrix(result, 'Viterbi Algorithm')

    print('\n======================================\n')

    # Run Individual Most Probable Tag Algorithm
    result, duration = tagger.run(3)
    print("IMPT Algorithm")
    print('  Accuracy     :', tagger.calc_accuracy(result, 'impt'))
    print('  Time Elapsed :', str(duration) + 's')
    if (flag) : tagger.calc_confusion_matrix(result, 'Individual Most Probable Tag Algorithm')

    print('\n======================================\n')
    
    
if __name__ == '__main__':
    flag = False
    lang = 'en'
    if (len(sys.argv) >= 2):
        if sys.argv[1] in ['en', 'ko', 'sv', 'ch', 'es']:
            lang = sys.argv[1]
    if ("conf" in sys.argv):
        flag = True

    main(lang, flag)

    