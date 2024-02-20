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

        self.setup(elems)

    def setup(self, elems):
        index = 1 
        for elem in elems:
            self.to_id[elem] = index
            self.to_text[index] = elem
            index += 1

    def __getitem__(self, key):
        if isinstance(key, int):
            try:
                return self.to_text[key]
            except KeyError:
                return 'UNK'
        else:
            try:
                return self.to_id[key]
            except KeyError:
                return 0

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.to_text[key] = value
            self.to_id[value] = key
        else:
            self.to_text[value] = key
            self.to_id[key] = value

class Tagger():
   
    def __init__(self, lang):
        self.min_log_prob = -float_info.max

        self.train_sents = conllu_corpus(train_corpus(lang))
        self.test_sents = conllu_corpus(test_corpus(lang))

        # Setup bijection between text to integer index for tags and types 
        self.word_vocab, self.tag_vocab = self.setup_vocab(self.train_sents)

        # pre-process train and test sentences 
        self.train_sents = self.preprocess_sentences(self.train_sents)
        self.test_sents =  self.preprocess_sentences(self.test_sents)

        # get set of all unique tags 
        self.tags = set([tag for sentence in self.train_sents for ( _, tag) in sentence])

        # get smoothed emission and transisions (bigram)
        self.smoothed_emissions_dist = self.init_smoothed_emission_dist(self.train_sents, self.tags)
        self.smoothed_transitions_dist = self.init_smoothed_transition_dist(self.train_sents)


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
            sent = [(self.word_vocab['<s>'], self.tag_vocab['START'])]

            for token in sentence:
                sent.append((self.word_vocab[token['form']], self.tag_vocab[token['upos']]))

            sent.append((self.word_vocab['</s>'], self.tag_vocab['END']))
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

    def test(self):
        return None
        

class EagerTagger(Tagger):
    def __init__(self, lang):
        super().__init__(lang)
    
    def tag(self):
        start = time.time()
        sentences=self.test_sents[0:10]
        result = [] # will contain list of sentences, tagged using eager algorithm

        for sentence in sentences:
            pred_sent = [sentence[0]] # initialize with start-of-sentence
            prev_tag=0
            for token in sentence[1:]:
                word = token[0] # the word to predict tag for 

                # list of all possible (tag, emission_prob * transistion_prob) for the given word
                probs = [(tag, self.smoothed_emissions_dist[tag].prob(word) * self.smoothed_transitions_dist.prob((prev_tag, tag))) for tag in self.tags]
               
                # tag with highest probability 
                max_prob_tag = max(probs, key=lambda obj:obj[1])[0]

                pred_sent.append((word,max_prob_tag))
                prev_tag = max_prob_tag
                # NOTE : we are leaving out end-of-sentence marker
            result.append(pred_sent)

        end = time.time()
        print(end - start)

        return result

    def convert(self, sentences):
        sents = []
        for sentence in sentences:
            sent = []
            for token in sentence:
                sent.append((self.word_vocab[token[0]], self.tag_vocab[token[1]]))
            sents.append(sent)

        return sents
    


def main():
    tagger = EagerTagger('en')
    tagged_sentences = tagger.convert(tagger.tag())
    for sentence in tagged_sentences:
        for token in sentence:
            print(token[0], token[1])
        print()

    
    
if __name__ == '__main__':
    main()

    