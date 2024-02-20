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


class Tagger():
    '''
        Necessary Keys:
            id
            form
            lemma
            upos 

    '''
   
    def __init__(self, lang):
        self.min_log_prob = -float_info.max
        self.train_sents = self.preprocess_sentences(conllu_corpus(train_corpus(lang)))
        self.test_sents =  self.preprocess_sentences(conllu_corpus(test_corpus(lang)))

        self.tags = set([tag for sentence in self.train_sents for (_, _, tag) in sentence])

        self.smoothed_emissions_dist = self.init_smoothed_tag_distribution(self.train_sents, self.tags)
        self.smoothed_transitions_dist = self.init_smoothed_transition_dist(self.train_sents, self.tags)

    def preprocess_sentences(self, sentences):
        '''
            Takes corpus sentence in conllu form and adds start and end of sentence markers to each sentence
            Modifies each token into form  : (id, word, tag)

            Arguments 
                sentences : list of sentences in conllu form 
            Returns
                list of sentences of form : [[(id, word, tag)]]
        '''

        sents = [] # [(id, form, upos)]
        
        # Convert conllu format to tuples of form (id, word, tag)
        for sentence in sentences:  
            sent = [(0, '<s>', 'START')] # add start of sentence marker 

            for token in sentence:
                sent.append((token['id'], token['form'], token['upos']))

            sent.append((len(sentence) + 1, '</s>', 'END')) # append end of sentence marker 
            sents.append(sent)

        return sents

    def init_smoothed_emission_dist(self, sentences, tags):
        '''
            Calculates smoothed distribution of emission probabilities P(word | tag)

            Arguments:
                sentences : list of sentences [[(id, word, tag)]]

            Returns : emission probability distribution (with Witten-Bell smoothing)
        '''
        distribution = {}

        for tag in tags:
            words = [w for sentence in sentences for (_, w, t) in sentence if t == tag]
            distribution[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

        return distribution
    
    def init_smoothed_transition_dist(self, sentences):
        '''
            Calculates smoothed distribution of transition probabilities P( tag[i] | tag[i-1] )

            Arguments:
                sentences : list of sentences [[(id, word, tag)]]

            Returns : transition probability trellis (with witten-bell smoothing)
        '''
        bigrams = [] 

        for sentence in sentences:
            tags = [obj[2] for obj in sentence]
            bigrams += ngrams(tags, 2)

        transition_trellis = WittenBellProbDist(FreqDist(bigrams), bins=1e5)

        return transition_trellis
        
    def test(self):
        return None


def main():
    tagger = Tagger('en')


    
if __name__ == '__main__':
    main()

    