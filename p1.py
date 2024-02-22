from nltk import FreqDist, WittenBellProbDist
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


class Bijection():
    '''
        Custom bijection dictionary for mapping string words and tags to integer indices 
    '''

    def __init__(self, elems):
        self.to_id = {}
        self.to_text = {}
        self.index = 0

        self.setup(elems)

    def setup(self, elems):
        for elem in elems:
            self.to_id[elem] = self.index
            self.to_text[self.index] = elem
            self.index += 1

    def __getitem__(self, key):
        if isinstance(key, int):
            try:
                return self.to_text[key]
            except KeyError:
                return -1
        else:
            try:
                return self.to_id[key]
            except KeyError:
                return 'UNK'

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.to_text[key] = value
            self.to_id[value] = key
        else:
            self.to_text[value] = key
            self.to_id[key] = value

    def __len__(self):
        return len(self.to_text.keys())

class Tagger():
   
    def __init__(self, lang):
        self.min_log_prob = -float_info.max

        self.train_sents = conllu_corpus(train_corpus(lang))
        self.test_sents = conllu_corpus(test_corpus(lang))

        # Setup bijection between text to integer index for tags and types 
        # self.word_vocab, self.tag_vocab = self.setup_vocab(self.train_sents)

        # pre-process train and test sentences 
        self.train_sents = self.preprocess_sentences(self.train_sents)
        self.test_sents =  self.preprocess_sentences(self.test_sents)

        # get set of all unique tags 
        self.tags = set([ tag for sentence in self.train_sents for ( _, tag) in sentence])
        self.words = set([w for sentence in self.train_sents for (w,_) in sentence])

        # get smoothed emission and transisions (bigram)
        self.emissions = self.init_smoothed_emission_dist(self.train_sents, self.tags)
        self.transitions = self.init_smoothed_transition_dist(self.train_sents)


    def setup_vocab(self, sentences):
        '''
            For setting up vocabulary for tags and words 
            Creates bijection between words and tags to integer indices
        '''

        words = [token['form'] for sentence in sentences for token in sentence]
        words.insert(0, '<s>')
        words.append('</s>')

        tags = [token['upos'] for sentence in sentences for token in sentence] 
        tags.insert(0, 'START')
        tags.append('END')

        return Bijection(set(words)), Bijection(set(tags))

    def preprocess_sentences(self, sentences):
        '''
            Takes corpus sentence in conllu form and adds start and end of sentence markers to each sentence
            Modifies each token into form  : (word, tag)

            Arguments 
                sentences : list of sentences in conllu form 
            Returns
                list of sentences of form : [[(word, tag)]]
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
        '''
            Calculates smoothed distribution of emission probabilities P(word | tag)

            Arguments:
                sentences : list of sentences [[(word, tag)]]

            Returns : emission probability distribution (with Witten-Bell smoothing)
        '''
        distribution = {}

        for tag in tags:
            words = [w for sentence in sentences for (w, t) in sentence if t == tag]
            distribution[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)
    
        return distribution
    
    
    def init_smoothed_transition_dist(self, sentences):
        '''
            Calculates smoothed distribution of transition probabilities P( tag[i] | tag[i-1] )

            Arguments:
                sentences : list of sentences [[(word, tag)]]

            Returns : transition probability trellis (with witten-bell smoothing)
        '''
        bigrams = [] 

        for sentence in sentences:
            tags = [obj[1] for obj in sentence]
            bigrams += ngrams(tags, 2)
 
        transition_trellis = WittenBellProbDist(FreqDist(bigrams), bins=1e5)

        return transition_trellis

    def eager_tag(self):
        start = time.time()
        sentences=self.test_sents
        result = [] # will contain list of sentences, tagged using eager algorithm

        for sentence in sentences:
            pred_sent = [sentence[0]] # initialize with start-of-sentence
            prev_tag='START'
            for token in sentence[1:]:
                word = token[0] # the word to predict tag for 

                # list of all possible (tag, emission_prob * transistion_prob) for the given word
                probs = [(tag, self.emissions[tag].logprob(word) + self.transitions.logprob((prev_tag, tag))) for tag in self.tags]

                # tag with highest probability 
                max_prob_tag = max(probs, key=lambda obj:obj[1])[0]

                pred_sent.append((word,max_prob_tag))
                prev_tag = max_prob_tag
                # NOTE : we are leaving out end-of-sentence marker
            pred_sent.append(sentence[-1])
            result.append(pred_sent)

        end = time.time()
        # print(end - start)

        return result
        

    def viterbi_tag(self):
        sentences=self.test_sents

        result = []
         
        for sentence in sentences:
            sentence = sentence[1:-1]
            viterbi = []

            # Initliaize 
            initial = {} 
            for tag in self.tags:
                initial[tag] = self.transitions.logprob(('START',tag)) + self.emissions[tag].logprob(sentence[0][0])
            viterbi.append(initial)

            # Intermediary
            i = 1
            while (i < len(sentence)):
                token = sentence[i][0]
                probs = {}

                for tag in self.tags:
                    probs[tag] = max([viterbi[i - 1][prev_tag] + self.transitions.logprob((prev_tag,tag)) + self.emissions[tag].logprob(token) for prev_tag in self.tags])

                viterbi.append(probs)
                i += 1
            
            # Finish
            final = {}
            final['END'] = max([viterbi[i-1][prev_tag] + self.transitions.logprob((prev_tag,'END')) for prev_tag in self.tags])
            viterbi.append(final)

            # Backtrack
            sen_result = []
            sen_result.append(("<s>", "START"))
            for i in range(0, len(sentence)):
                v_col = viterbi[i]
                word = sentence[i][0]
                max_tag = max(v_col.items(), key=lambda obj:obj[1])[0]
                
                sen_result.append((word, max_tag))

            sen_result.append(('</s>', 'END'))
            result.append(sen_result)

        return result
    
    def forward_backward_tag(self):
        sentences=self.test_sents

        result = []

        for sentence in sentences:
            forward = []
            backward = []

            # INITIAL 
            initial_f = {} 
            initial_b = {}
            
            for tag in self.tags:
                initial_f[tag] = self.transitions.logprob(('START', tag)) + self.emissions[tag].logprob(sentence[0][0])
                initial_b[tag] = self.transitions.logprob((tag, 'END'))
            forward.append(initial_f)
            backward.append(initial_b)

            # INTERMEDIARY 
            i = 1
            while (i < len(sentence) - 1):
                j = len(sentence) - i
                token_f = sentence[i][0]
                token_b = sentence[j][0]
                intermed_f = {}
                intermed_b = {}

                for tag in self.tags:
                    inner_f = [forward[i - 1][prev_tag] + self.transitions.logprob((prev_tag, tag)) + self.emissions[tag].logprob(token_f) for prev_tag in self.tags]
                    inner_b = [backward[i - 1][next_tag] + self.transitions.logprob((tag, next_tag)) + self.emissions[next_tag].logprob(token_b) for next_tag in self.tags]

                    intermed_f[tag] = self.logsumexp(inner_f)
                    intermed_b[tag] = self.logsumexp(inner_b)

                forward.append(intermed_f)
                backward.append(intermed_b)
                i += 1

            # FINAL
            final_f = {}
            final_b = {}
            final_f['END'] = self.logsumexp([forward[i-1][tag] + self.transitions.logprob((tag, 'END')) for tag in self.tags])
            final_b['START'] = self.logsumexp([backward[i-1][tag] + self.transitions.logprob(('START', tag)) + self.emissions[tag].logprob(sentence[0][0]) for tag in self.tags])
            forward.append(final_f)
            backward.append(final_b)

            backward.reverse() # reverse backwards matrix to align with forward matrix 

            #Back Track
            sen_result = []
            sen_result.append(("<s>", "START"))
            for i in range(1, len(sentence) - 1):
                word = sentence[i][0]
                f_col = forward[i]
                b_col = backward[i]
                combined = self.combine_dicts(f_col, b_col)
                max_tag = max(combined.items(), key = lambda obj:obj[1])[0]
                sen_result.append((word, max_tag))
            sen_result.append(("</s>", "END"))
                
            result.append(sen_result)

        return result

    def combine_dicts(self, dict1, dict2):
        result = {}
        for key in dict1.keys():
            result[key] = dict1[key] + dict2[key]
        return result

    # Adding a list of probabilities represented as log probabilities.
    def logsumexp(self, vals):
        if len(vals) == 0:
            return self.min_log_prob
        m = max(vals)
        if m == self.min_log_prob:
            return self.min_log_prob
        else:
            return m + log(sum([exp(val - m) for val in vals]))

    def calc_accuracy(self, predictions):
        num_correct = 0.0
        total = 0.0

        for i in range(0, len(self.test_sents)):
            pred_sent = predictions[i]
            label_sent = self.test_sents[i]

            for j in range(0, len(label_sent)):
                pred_tag = pred_sent[j][1]
                label_tag = label_sent[j][1]

                if pred_tag == label_tag:
                    num_correct += 1.0

                total += 1.0
        return num_correct/total

def main():
    tagger = Tagger('en')

    # result = tagger.eager_tag()

    # print('Eager :', tagger.calc_accuracy(result))

    result = tagger.viterbi_tag()
    
    print('Viterbi :', tagger.calc_accuracy(result))

    result = tagger.forward_backward_tag()

    print('F-B :', tagger.calc_accuracy(result))

    
    
if __name__ == '__main__':
    main()

    