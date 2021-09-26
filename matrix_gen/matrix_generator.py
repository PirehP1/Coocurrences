#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:40:20 2021

@author: Audrey Quessada Vial, Stephane Lamassé
Institutions: Pireh-Lamop
LICENCE GNU
This script aims to generate cooccurrence matrix
"""

__version__ = "0.1.0"
__authors__ = "Audrey Quessada Vial","Stephane Lamassé"



import sys
sys.path.append("../")
import os
import numpy as np
import pandas as pd
#import networkx
import collections
import itertools
import scipy.sparse as sp
import time
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from traitement.traitement import Preprocessing, read_stop_word, timer
from metrique.metrique import *



class MatrixGenerator(Preprocessing):
    """
    """
    def __init__(self, *args, **kwargs):
        super(MatrixGenerator, self).__init__(*args, **kwargs)
        
                       
    @timer
    def get_vocab(self, text_clean, **kwargs):
        """
        This function retrieves the vocabulary of a corpus

        Parameters
        ----------
        text_clean : TYPE str
            DESCRIPTION. clean text 
        **kwargs : TYPE
            separator: str, separator used to get the token from a text. 
            For more complex languages, please design a new function
            stopwords: list of stop_words to use
            filter_by_length: int, if we want to filter the tokens by their length
            DESCRIPTION.

        Returns
        -------
        vocab : TYPE
            DESCRIPTION.

        """
        filter_by_length = kwargs.get("filter_by_length", self.filter_by_length)
        stopwords = kwargs.get("stop_words", self.stop_words)
        separator = kwargs.get("separator", self.separator)
        
        vocab = {}
        doc = text_clean.split(separator)
        index = 0
        for token in doc:
            if token not in stopwords and len(token)>= filter_by_length:
                if token in vocab.keys():
                    pass
                else:
                    vocab[token] = index
                    index += 1
        return vocab
    
    
    @timer
    def get_vocab_filter_by_count(self, text_clean, **kwargs):
        """
        

        Parameters
        ----------
        text_clean : TYPE str
            DESCRIPTION. clean text
        **kwargs : TYPE
            separator: str, separator used to get the token from a text. 
            For more complex languages, please design a new function
            stopwords: list of stop_words to use
            filter_by_length: int, if we want to filter the tokens by their length
            filter_by_count: int, if we want to filter the tokens by their count in the full text
            DESCRIPTION.

        Returns
        -------
        vocab : TYPE
            DESCRIPTION.

        """
        filter_by_length = kwargs.get("filter_by_length", self.filter_by_length)
        stopwords = kwargs.get("stop_words", self.stop_words)
        separator = kwargs.get("separator", self.separator)
        filter_by_count = kwargs.get("filter_by_count", 0)
        
        list_words = [tok for tok in text_clean.split(separator) if (tok not in stopwords and len(tok) >= filter_by_length)]
        full_counter = collections.Counter(list_words)
        vocab = {}
        index = 0
        for token, count in full_counter.items():
            if count >= filter_by_count:
                if token in vocab.keys():
                    pass
                else:
                    vocab[token] = index
                    index += 1                
        return vocab
        
  
    @timer
    def counter_full(self, text_clean, vocab, **kwargs):
        """
        This function creates a word counter over the full text

        Parameters
        ----------
        text_clean : TYPE
            DESCRIPTION.
        vocab : TYPE
            DESCRIPTION.
            separator: str, separator used to get the token from a text. 
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if vocab == {}:
            print("empty vocabulary, cannot process, exit")
            sys.exit()
        separator = kwargs.get("separator", self.separator)
        list_words = [tok for tok in text_clean.split(separator) if tok in vocab.keys()]
        return collections.Counter(list_words)
    
  
    @timer    
    def counter_by_sentence(self, raw_text, vocab, regex_clean, separator_sent=".", **kwargs):
        """
        This function creates a generator of word counter by sentences

        Parameters
        ----------
        raw_text : TYPE
            DESCRIPTION.
        vocab : TYPE
            DESCRIPTION.
        regex_clean : TYPE
            DESCRIPTION.
        separator_sent : TYPE, optional
            DESCRIPTION. The default is ".".
        **kwargs : TYPE
            case: bool, if we want to lower cases
            separator: str, separator used to get the token from a text. 
            For more complex languages, please design a new function

        Returns
        -------
        None.

        """
        if vocab == {}:
            print("empty vocabulary, cannot process, exit")
            sys.exit()
        case = kwargs.get("case", self.case)
        separator = kwargs.get("separator", self.separator)
        text = self.basic_text_cleaning(raw_text, 
                                        regex_clean,
                                        case=case,
                                        remove_punctuation=False)
        list_sent = text.split(separator_sent)
        for sent in list_sent:
            list_tok = [tok for tok in sent.split(separator) if tok in vocab.keys()]
            yield collections.Counter(list_tok)
     
            
    @timer 
    def counter_by_window(self, text_clean, vocab, **kwargs):
        """
        This function creates a generator of word counter by window

        Parameters
        ----------
        text_clean : TYPE
            DESCRIPTION.
        vocab : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            window: int, the size of the window
            separator: str, separator used to get the token from a text. 
            For more complex languages, please design a new function

        Yields
        ------
        TYPE
            DESCRIPTION.

        """
        if vocab == {}:
            print("empty vocabulary, cannot process, exit")
            sys.exit()
            
        window = kwargs.get("window", self.window)
        separator = kwargs.get("separator", self.separator)
        
        list_words = [tok for tok in text_clean.split(separator) if tok in vocab.keys()]
        nb_chunks = len(list_words) // window
        list_window = []
        if len(list_words) < window:
            print("text size lower than window size, please select another window size")
            sys.exit()
        for i in range(nb_chunks):
            chunk = list_words[i*window:(i+1)*window]
            list_window.append(chunk)
        last_chunk = list_words[nb_chunks*window:]
        list_window.append(last_chunk)
        for chunk in list_window:
            yield collections.Counter(chunk)
            
            
    @timer        
    def counter_by_sliding_window(self, text_clean, vocab, step=1, **kwargs):
        """
        This function creates a generator of word counter by sliding window

        Parameters
        ----------
        text_clean : TYPE
            DESCRIPTION.
        vocab : TYPE
            DESCRIPTION.

        step : TYPE, optional
            DESCRIPTION. The default is 1.
        **kwargs : TYPE
            window: int, the size of the sliding window
            separator: str, separator used to get the token from a text. 
            For more complex languages, please design a new function

        Yields
        ------
        TYPE
            DESCRIPTION.

        """
        if vocab == {}:
            print("empty vocabulary, cannot process, exit")
            sys.exit()
            
        window = kwargs.get("window", self.window)
        separator = kwargs.get("separator", self.separator)
        
        list_words = [tok for tok in text_clean.split(separator) if tok in vocab.keys()]
        for i in range(0, len(list_words), step):
            chunk = list_words[i:i+window+1]
            yield collections.Counter(chunk)
            
    
    @timer
    def get_directed(self, data, vocab, option="full_text"):
        """
        

        Parameters
        ----------
        data : TYPE clean text (iterator) or generator of sentences or windows
            DESCRIPTION.
        vocab : TYPE dictionnary of word: index
            DESCRIPTION.
        option : TYPE, optional str
            DESCRIPTION. The default is "full_text".

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if vocab == {}:
            print("empty vocabulary, cannot process, exit")
            sys.exit()
            
        list_vocab = list(vocab.keys())
        n_vocab = len(list_vocab)
        dir_mat = np.zeros((n_vocab, n_vocab))
        
        if option == "full_text":
            counter = self.counter_full(data, vocab)
            for token, index in vocab.items():
                    dir_mat[index,:] += counter[token]
                
        else:
            for counter in data:
                for ((token1, count1), (token2, count2)) in itertools.permutations(counter.items(),2):
                    i = vocab[token1]
                    j = vocab[token2]
                    if token1 != token2:
                        dir_mat[i, j] += count1
        return pd.DataFrame(dir_mat, columns=list_vocab, index=list_vocab)  
      
    
    @timer
    def get_non_directed(self, data, **kwargs):
        """
        

        Parameters
        ----------
        data : TYPE, clean text data iterator or generator
            DESCRIPTION.
        **kwargs : TYPE
            separator: str, separator used to get the token from a text. 
            For more complex languages, please design a new function
            stopwords: list of stop_words to use
            filter_by_length: int, if we want to filter the tokens by their length
            joint: bool, 
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        
        filter_by_length = kwargs.get("filter_by_length", self.filter_by_length)
        joint = kwargs.get("joint", False)
        separator = kwargs.get("separator", self.separator)
        stopwords = kwargs.get("stop_words", self.stop_words)
        list_token = list(self.get_token_window(data, 
                                                filter_by_length=filter_by_length, 
                                                joint=joint,
                                                separator=separator,
                                                stop_words=stopwords))
        
        indices = []
        data = []
        vocab = {}
        indptr = [0]
        for i, d in enumerate(list_token):
            for term in d:
                index = vocab.setdefault(term, len(vocab))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))
        mat = sp.csr_matrix((data, indices, indptr), dtype=int)
        mat_coocs = np.dot(mat.T, mat)
        return pd.DataFrame(data=mat_coocs.toarray(), columns=vocab.keys(), index=vocab.keys())
    
 
#TODO create your own word analyzer ofr Counter and TFIDF matrices    
    @timer
    def get_tfidf_matrix(self, data, preprocess=True, **kwargs):
        """
        This function creates a word matrix from TFIDF from sklearn

        Parameters
        ----------
        data : TYPE clean text data iterator or generator
            DESCRIPTION.
        preprocess : TYPE, optional
            DESCRIPTION. The default is True.
        **kwargs : TYPE
            separator: str, separator used to get the token from a text. 
            For more complex languages, please design a new function
            stopwords: list of stop_words to use
            filter_by_length: int, if we want to filter the tokens by their length
            joint: bool, 
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        
        filter_by_length = kwargs.get("filter_by_length", self.filter_by_length)
        separator = kwargs.get("separator", self.separator)
        stopwords = kwargs.get("stop_words", self.stop_words)
        joint = kwargs.get("joint", True)
        if preprocess == True:
            gen_token = self.get_token_window(data, 
                                              filter_by_length=filter_by_length, 
                                              joint=joint,
                                              separator=separator, 
                                              stop_words=stopwords)
            list_token = list(gen_token)
        else:
            list_token = list(data)
        tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words=stopwords)
        tfidf_wm = tfidfvectorizer.fit_transform(list_token)
        tfidf_tokens = tfidfvectorizer.get_feature_names()
        mat_tfidf_wm = np.dot(tfidf_wm.T, tfidf_wm)
        return pd.DataFrame(data=mat_tfidf_wm.toarray(), columns=tfidf_tokens, index=tfidf_tokens)
    
        
    @timer
    def get_counter_matrix(self, data, preprocess=True, **kwargs):
        """
        This function creates a word matrix based on Counter from sklearn

        Parameters
        ----------
        data : TYPE clean text data iterator or generator
            DESCRIPTION.
        preprocess : TYPE, optional
            DESCRIPTION. The default is True.
        **kwargs : TYPE
            separator: str, separator used to get the token from a text. 
            For more complex languages, please design a new function
            stopwords: list of stop_words to use
            filter_by_length: int, if we want to filter the tokens by their length
            joint: bool,
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        filter_by_length = kwargs.get("filter_by_length", self.filter_by_length)
        separator = kwargs.get("separator", self.separator)
        stopwords = kwargs.get("stop_words", self.stop_words)
        joint = kwargs.get("joint", True)
        if preprocess == True:
            gen_token = self.get_token_window(data, 
                                              filter_by_length=filter_by_length, 
                                              joint=joint,
                                              separator=separator, 
                                              stop_words=stopwords)
            list_token = list(gen_token)
        else:
            list_token = list(data)
        countvectorizer = CountVectorizer(analyzer= 'word', stop_words=stopwords)
        count_wm = countvectorizer.fit_transform(list_token)
        count_tokens = countvectorizer.get_feature_names()
        mat_count_wm = np.dot(count_wm.T, count_wm)
        return pd.DataFrame(data=mat_count_wm.toarray(), columns=count_tokens, index=count_tokens)



def save_result(result, filename, option="csv"):
    """
    Save a dataframe result

    Parameters
    ----------
    result : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.
    option : TYPE, optional
        DESCRIPTION. The default is "csv".

    Returns
    -------
    None.

    """
    print(f"save {filename} format .{option}")
    if option == "csv":
        result.to_csv(filename)
    if option == "npy":
        print("Careful, you will loose  index and column names")
        result_to_save = result.values
        np.save(filename, result_to_save)
    if option =="pkl":
        result.to_pickle(filename)
    if option == "hdf5":
        result.to_hdf(filename)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default=None, help="folder path of the text files, str")
    parser.add_argument('--path_stop', default=None, help="path of stop words list")
    parser.add_argument('--stopwords', default=None, help="list of stop words")
    parser.add_argument("--window", default=5, type=int, help="analysis window, nb of words") 
    parser.add_argument("--remove_punctuation", default=True, type=bool, help="boolean to know if we want to remove all the punctuation")
    parser.add_argument("--filter_by_length", default=0, type=int, help="value to filter words by their length")
    parser.add_argument("--separator", default=" ", type=str, help="simple separator to get tokens from text")
    parser.add_argument("--case", default=True, type=bool, help="boolean to know if we want to lower cases")
    parser.add_argument("--save", default=None, type=str, help="folder path to save results")
    args = parser.parse_args()
    if args.input_path == None:
        input_path = "../Data/corpusbytime"
    else:
        input_path = args.input_path
    if args.save == None:
        save_path = "../Data/results"
    else:
        save_path = args.save
    if isinstance(args.stopwords, list):
        stop_words = args.stopwords
    else:
        if args.path_stop == None and args.stopwords==None:
            stop_words = []
        else:
            path_stop_words = "../Data/stop_words.txt"
            stop_words = read_stop_word(path_stop_words)
    #print(stop_words)
    window = args.window
    remove_punctuation = args.remove_punctuation
    filter_by_length = args.filter_by_length
    separator = args.separator
    case = args.case
    # input_path = "../Data/corpusbytime"
    # path_stop_words = "../Data/stop_words.txt"
    #reading stop words
    # stop_words = read_stop_word(path_stop_words)
    stop_words_null = []

    #initialization matrix_generator
    matrix_gen = MatrixGenerator(input_path=input_path,
                            window=window,
                            stop_words=stop_words)
    matrix_gen50 = MatrixGenerator(input_path=input_path,
                            window=50,
                            stop_words=stop_words)
    #getting list of paths to process
    list_file = matrix_gen.get_files_simple()
    #print(list_file)
    #regex definition
    regex_clean = "(\')?(\d*)?(\\n)[']"
    
    text_basic = """Le but de toute association politique est la conservation des droits naturels et imprescriptibles de l'Homme. 
    Ces droits sont la liberté, la propriété, la sûreté, et la résistance à l'oppression."""
    text_test = "I enjoy flying. I like NLP just like you. I like Deep Learning"
    with open(list_file[0], "r") as f:
        text_file = f.read()
    
    print("-----------LECTURE DU TEXTE FULL + BASIC CLEANING ----------------")
    print("...LECTURE DE TEXT_TEST FULL TEXT")
    text_full_test = matrix_gen.read_text_full(None, regex_clean, texte=text_test)
    print(text_full_test)
    print("...LECTURE DE TEXT_BASIC FULL TEXT")
    text_full_basic = matrix_gen.read_text_full(None, regex_clean, texte=text_basic)
    print(text_full_basic)
    print("...LECTURE DE TEXT_FILE FULL TEXT")
    text_full_file = matrix_gen.read_text_full(list_file[0], regex_clean)
    print(text_full_file[0][:50])
    print(" ")
    print("-------------VERIFICATION GET_VOCAB FULL TEXT----------------")
    print("...1.GET_VOCAB POUR TEXT_TEST FULL TEXT")
    vocab_test_count = matrix_gen.get_vocab_filter_by_count(text_full_test[0],
                                                            filter_by_length=0,
                                                            stop_words=stop_words_null)
    print(vocab_test_count)
    print("...2.GET_VOCAB POUR TEXT_BASIC FULL TEXT")
    vocab_basic_count = matrix_gen.get_vocab_filter_by_count(text_full_basic[0],
                                                              filter_by_length=2,
                                                              stop_words=stop_words)
    print(vocab_basic_count)
    print("...3.GET_VOCAB POUR TEXT_FILE FULL TEXT")
    vocab_file_count = matrix_gen.get_vocab_filter_by_count(text_full_file[0],
                                                            filter_by_length=2,
                                                            stop_words=stop_words,
                                                            filter_by_count=5)
    print(len(vocab_file_count.keys()))
    print(" ")
    print("-------------VERIFICATION GET_VOCAB FULL TEXT----------------")
    print("...4.GET_VOCAB POUR TEXT_TEST FULL TEXT")
    vocab_test = matrix_gen.get_vocab(text_full_test[0],
                                      filter_by_length=0,
                                      stop_words=stop_words_null)
    print(vocab_test)
    print("...5.GET_VOCAB POUR TEXT_BASIC FULL TEXT")
    vocab_basic = matrix_gen.get_vocab(text_full_basic[0],
                                      filter_by_length=2,
                                      stop_words=stop_words)
    print(vocab_basic)
    print("...6.GET_VOCAB POUR TEXT_FILE FULL TEXT")
    vocab_file = matrix_gen.get_vocab(text_full_file[0],
                                      filter_by_length=2,
                                      stop_words=stop_words)
    print(len(vocab_file.keys()))
    print(" ")
    print("-------------VERIFICATION COUNTER_FULL----------------")
    print("...7.COUNTER_FULL POUR TEXT_TEST")
    counter_full_test = matrix_gen.counter_full(text_full_test[0], vocab_test)
    print(counter_full_test)
    print("...8.COUNTER_FULL POUR TEXT_BASIC")
    counter_full_basic = matrix_gen.counter_full(text_full_basic[0], vocab_basic)
    print(counter_full_basic)
    print("...9.COUNTER_FULL POUR TEXT_FILE")
    counter_full_file = matrix_gen.counter_full(text_full_file[0], vocab_file)
    print(len(counter_full_file.keys()))
    print(" ")
    print("-------------VERIFICATION COUNTER_BY_SENTENCE----------------")
    print("...10.COUNTER_BY_SENTENCE POUR TEXT_TEST")
    gen_count_sent_test = matrix_gen.counter_by_sentence(text_test, 
                                                          vocab_test, 
                                                          regex_clean, 
                                                          separator_sent=".")
    print(next(gen_count_sent_test))
    gen_count_sent_test = matrix_gen.counter_by_sentence(text_test, 
                                                          vocab_test, 
                                                          regex_clean, 
                                                          separator_sent=".")
    print("...11.COUNTER_BY_SENTENCE POUR TEXT_BASIC")
    gen_count_sent_basic = matrix_gen.counter_by_sentence(text_basic, 
                                                          vocab_basic, 
                                                          regex_clean, 
                                                          separator_sent=".")
    print(next(gen_count_sent_basic))
    gen_count_sent_basic = matrix_gen.counter_by_sentence(text_basic, 
                                                          vocab_basic, 
                                                          regex_clean, 
                                                          separator_sent=".")
    print("...12.COUNTER_BY_SENTENCE POUR TEXT_FILE")
    gen_count_sent_file = matrix_gen.counter_by_sentence(text_file, 
                                                          vocab_file, 
                                                          regex_clean, 
                                                          separator_sent=".")
    print(next(gen_count_sent_file))
    gen_count_sent_file = matrix_gen.counter_by_sentence(text_file, 
                                                          vocab_file, 
                                                          regex_clean, 
                                                          separator_sent=".")
    print(" ")
    print("-------------VERIFICATION COUNTER_BY_WINDOW----------------")
    print("...13.COUNTER_BY_WINDOW POUR TEXT_TEST")
    gen_count_wnd_test = matrix_gen.counter_by_window(text_full_test[0], 
                                                      vocab_test, 
                                                      window=5)
    print(next(gen_count_wnd_test))
    gen_count_wnd_test = matrix_gen.counter_by_window(text_full_test[0], 
                                                      vocab_test, 
                                                      window=5)
    print("...14.COUNTER_BY_WINDOW POUR TEXT_BASIC")
    gen_count_wnd_basic = matrix_gen.counter_by_window(text_full_basic[0], 
                                                      vocab_basic, 
                                                      window=5)
    print(next(gen_count_wnd_basic))
    gen_count_wnd_basic = matrix_gen.counter_by_window(text_full_basic[0], 
                                                      vocab_basic, 
                                                      window=5)
    print("...15.COUNTER_BY_WINDOW POUR TEXT_FILE")
    gen_count_wnd_file = matrix_gen.counter_by_window(text_full_file[0], 
                                                      vocab_file, 
                                                      window=50)
    print(next(gen_count_wnd_file))
    gen_count_wnd_file = matrix_gen.counter_by_window(text_full_file[0], 
                                                      vocab_file, 
                                                      window=50)
    print(" ")
    print("-------------VERIFICATION COUNTER_BY_SLIDING_WINDOW----------------")
    print("...16.COUNTER_BY_SLIDING_WINDOW POUR TEXT_TEST")
    gen_count_swnd_test = matrix_gen.counter_by_sliding_window(text_full_test[0], 
                                                                vocab_test, 
                                                                window=4)
    print(next(gen_count_swnd_test))
    gen_count_swnd_test = matrix_gen.counter_by_sliding_window(text_full_test[0], 
                                                                vocab_test, 
                                                                window=4)
    print("...17.COUNTER_BY_SLIDING_WINDOW POUR TEXT_BASIC")
    gen_count_swnd_basic = matrix_gen.counter_by_sliding_window(text_full_basic[0], 
                                                                vocab_basic, 
                                                                window=5,
                                                                step=2)
    print(next(gen_count_swnd_basic))
    gen_count_swnd_basic = matrix_gen.counter_by_sliding_window(text_full_basic[0], 
                                                                vocab_basic, 
                                                                window=5,
                                                                step=2)
    print("...18.COUNTER_BY_SLIDING_WINDOW POUR TEXT_FILE")
    gen_count_swnd_file = matrix_gen.counter_by_sliding_window(text_full_file[0], 
                                                                vocab_file, 
                                                                window=50,
                                                                step=10)
    print(next(gen_count_swnd_file))
    gen_count_swnd_file = matrix_gen.counter_by_sliding_window(text_full_file[0], 
                                                                vocab_file, 
                                                                window=50,
                                                                step=10)
    print(" ")
    print("-------------VERIFICATION GET_DIRECTED_MATRIX FULL_TEXT----------------")
    print("...19.VERIFICATION GET_DIRECTED_MATRIX POUR TEXT_TEST FULL TEXT")
    mat_dir_full_test = matrix_gen.get_directed(text_full_test[0], 
                                                vocab_test, 
                                                option="full_text")
    print(mat_dir_full_test)
    print("...20.VERIFICATION GET_DIRECTED_MATRIX POUR TEXT_BASIC FULL TEXT")
    mat_dir_full_basic = matrix_gen.get_directed(text_full_basic[0], 
                                                vocab_basic, 
                                                option="full_text")
    print(mat_dir_full_basic)
    print("...21.VERIFICATION GET_DIRECTED_MATRIX POUR TEXT_FILE FULL TEXT")
    mat_dir_full_file = matrix_gen.get_directed(text_full_file[0], 
                                                vocab_file, 
                                                option="full_text")
    print(mat_dir_full_file)
    print(" ")
    print("-------------VERIFICATION GET_DIRECTED_MATRIX SENTENCE----------------")
    print("...22.VERIFICATION GET_DIRECTED_MATRIX POUR TEXT_TEST SENTENCE")
    mat_dir_sent_test = matrix_gen.get_directed(gen_count_sent_test, 
                                                vocab_test, 
                                                option="")
    print(mat_dir_sent_test)
    print("...23.VERIFICATION GET_DIRECTED_MATRIX POUR TEXT_BASIC SENTENCE")
    mat_dir_sent_basic = matrix_gen.get_directed(gen_count_sent_basic, 
                                                vocab_basic, 
                                                option="")
    print(mat_dir_sent_basic)
    print("...24.VERIFICATION GET_DIRECTED_MATRIX POUR TEXT_FILE SENTENCE")
    mat_dir_sent_file = matrix_gen.get_directed(gen_count_sent_file, 
                                                vocab_file, 
                                                option="")
    print(mat_dir_sent_file)
    print(" ")
    print("-------------VERIFICATION GET_DIRECTED_MATRIX WINDOW----------------")
    print("...25.VERIFICATION GET_DIRECTED_MATRIX POUR TEXT_TEST WINDOW")
    mat_dir_wnd_test = matrix_gen.get_directed(gen_count_wnd_test, 
                                                vocab_test, 
                                                option="")
    print(mat_dir_wnd_test)
    print("...26.VERIFICATION GET_DIRECTED_MATRIX POUR TEXT_BASIC WINDOW")
    mat_dir_wnd_basic = matrix_gen.get_directed(gen_count_wnd_basic, 
                                                vocab_basic, 
                                                option="")
    print(mat_dir_wnd_basic)
    print("...27.VERIFICATION GET_DIRECTED_MATRIX POUR TEXT_TEST WINDOW")
    mat_dir_wnd_file = matrix_gen.get_directed(gen_count_wnd_file, 
                                                vocab_file, 
                                                option="")
    print(mat_dir_wnd_file)
    print(" ")
    print("-------------VERIFICATION GET_DIRECTED_MATRIX SLIDING WINDOW----------------")
    print("...28.VERIFICATION GET_DIRECTED_MATRIX POUR TEXT_TEST SLIDING WINDOW")
    mat_dir_swnd_test = matrix_gen.get_directed(gen_count_swnd_test, 
                                                vocab_test, 
                                                option="")
    print(mat_dir_swnd_test)
    print("...29.VERIFICATION GET_DIRECTED_MATRIX POUR TEXT_BASIC SLIDING WINDOW")
    mat_dir_swnd_basic = matrix_gen.get_directed(gen_count_swnd_basic, 
                                                vocab_basic, 
                                                option="")
    print(mat_dir_swnd_basic)
    print("...30.VERIFICATION GET_DIRECTED_MATRIX POUR TEXT_FILE SLIDING WINDOW")
    mat_dir_swnd_file = matrix_gen.get_directed(gen_count_swnd_file, 
                                                vocab_file, 
                                                option="")
    print(mat_dir_swnd_file)
    print(" ")
    print("-------------VERIFICATION GET_NON_DIRECTED_MATRIX FULL_TEXT----------------")
    print("...31.VERIFICATION GET_NON_DIRECTED_MATRIX POUR TEXT_TEST FULL TEXT")
    mat_nondir_full_test = matrix_gen.get_non_directed(text_full_test, 
                                                        preprocess=True, 
                                                        filter_by_length=0,
                                                        stop_words=stop_words_null)
    print(mat_nondir_full_test)
    print("...32.VERIFICATION GET_NON_DIRECTED_MATRIX POUR TEXT_BASIC FULL TEXT")
    mat_nondir_full_basic = matrix_gen.get_non_directed(text_full_basic)
    print(mat_nondir_full_basic)
    print("...33.VERIFICATION GET_NON_DIRECTED_MATRIX POUR TEXT_FILE FULL TEXT")
    mat_nondir_full_file = matrix_gen.get_non_directed(text_full_file)
    print(mat_nondir_full_file)
    print(" ")
    print("-------------VERIFICATION GET_NON_DIRECTED_MATRIX SENTENCE----------------")
    print("...34.VERIFICATION GET_NON_DIRECTED_MATRIX POUR TEXT_TEST SENTENCE")
    gen_sent_test = matrix_gen.read_text_sentence(None, regex_clean, texte=text_test)
    mat_nondir_sent_test = matrix_gen.get_non_directed(gen_sent_test, 
                                                        preprocess=True, 
                                                        filter_by_length=0,
                                                        stop_words=stop_words_null)
    print(mat_nondir_sent_test)
    print("...35.VERIFICATION GET_NON_DIRECTED_MATRIX POUR TEXT_BASIC SENTENCE")
    gen_sent_basic = matrix_gen.read_text_sentence(None, regex_clean, texte=text_basic)
    mat_nondir_sent_basic = matrix_gen.get_non_directed(gen_sent_basic, filter_by_length=2)
    print(mat_nondir_sent_basic)
    print("...36.VERIFICATION GET_NON_DIRECTED_MATRIX POUR TEXT_FILE SENTENCE")
    gen_sent_file = matrix_gen.read_text_sentence(list_file[0], regex_clean)
    mat_nondir_sent_file = matrix_gen.get_non_directed(gen_sent_file)
    print(mat_nondir_sent_file)
    print(" ")
    print("-------------VERIFICATION GET_NON_DIRECTED_MATRIX WINDOW----------------")
    print("...37.VERIFICATION GET_NON_DIRECTED_MATRIX POUR TEXT_TEST WINDOW")
    gen_wnd_test = matrix_gen.read_text_window(None, regex_clean, texte=text_test)
    mat_nondir_wnd_test = matrix_gen.get_non_directed(gen_wnd_test, 
                                                        preprocess=True, 
                                                        filter_by_length=0,
                                                        stop_words=stop_words_null)
    print(mat_nondir_wnd_test)
    print("...38.VERIFICATION GET_NON_DIRECTED_MATRIX POUR TEXT_BASIC WINDOW")
    gen_wnd_basic = matrix_gen.read_text_window(None, regex_clean, texte=text_basic)
    mat_nondir_wnd_basic = matrix_gen.get_non_directed(gen_wnd_basic)
    print(mat_nondir_wnd_basic)
    print("...39.VERIFICATION GET_NON_DIRECTED_MATRIX POUR TEXT_TEST WINDOW")
    gen_wnd_file = matrix_gen.read_text_window(list_file[0], regex_clean, window=50)
    mat_nondir_wnd_file = matrix_gen.get_non_directed(gen_wnd_file)
    print(mat_nondir_wnd_file)
    print(" ")
    print("-------------VERIFICATION GET_NON_DIRECTED_MATRIX SLIDING WINDOW----------------")
    print("...40.VERIFICATION GET_NON_DIRECTED_MATRIX POUR TEXT_TEST SLIDING WINDOW")
    gen_swnd_test = matrix_gen.read_text_sliding_window(None, regex_clean, texte=text_test)
    mat_nondir_swnd_test = matrix_gen.get_non_directed(gen_swnd_test, 
                                                        preprocess=True, 
                                                        filter_by_length=0,
                                                        stop_words=stop_words_null)
    print(mat_nondir_swnd_test)
    print("...41.VERIFICATION GET_NON_DIRECTED_MATRIX POUR TEXT_BASIC SLIDING WINDOW")
    gen_swnd_basic = matrix_gen.read_text_sliding_window(None, 
                                                          regex_clean, 
                                                          texte=text_basic,
                                                          window=5,
                                                          step=2)
    mat_nondir_swnd_basic = matrix_gen.get_non_directed(gen_swnd_basic)
    print(mat_nondir_swnd_basic)
    print("...42.VERIFICATION GET_NON_DIRECTED_MATRIX POUR TEXT_FILE SLIDING WINDOW")
    gen_swnd_file = matrix_gen.read_text_sliding_window(list_file[0], 
                                                          regex_clean, 
                                                          window=50,
                                                          step=10)
    mat_nondir_swnd_file = matrix_gen.get_non_directed(gen_swnd_file)
    print(mat_nondir_swnd_file)
    print(" ")
    print("-------------VERIFICATION GET_TFIDF_MATRIX FULL_TEXT----------------")
    print("...43.VERIFICATION GET_TFIDF_MATRIX POUR TEXT_TEST FULL TEXT")
    mat_tfidf_full_test = matrix_gen.get_tfidf_matrix(text_full_test, 
                                                        preprocess=True, 
                                                        filter_by_length=0,
                                                        stop_words=stop_words_null)
    print(mat_tfidf_full_test)
    print("...44.VERIFICATION GET_TFIDF_MATRIX POUR TEXT_BASIC FULL TEXT")
    mat_tfidf_full_basic = matrix_gen.get_tfidf_matrix(text_full_basic)
    print(mat_tfidf_full_basic)
    print("...45.VERIFICATION GET_TFIDF_MATRIX POUR TEXT_FILE FULL TEXT")
    mat_tfidf_full_file = matrix_gen.get_tfidf_matrix(text_full_file)
    print(mat_tfidf_full_file)
    print(" ")
    print("-------------VERIFICATION GET_TFIDF_MATRIX SENTENCE----------------")
    print("...46.VERIFICATION GET_NON_TFIDF_MATRIX POUR TEXT_TEST SENTENCE")
    gen_sent_test = matrix_gen.read_text_sentence(None, regex_clean, texte=text_test)
    mat_tfidf_sent_test = matrix_gen.get_tfidf_matrix(gen_sent_test, 
                                                        preprocess=True, 
                                                        filter_by_length=0,
                                                        stop_words=stop_words_null)
    print(mat_tfidf_sent_test)
    print("...47.VERIFICATION GET_TFIDF_MATRIX POUR TEXT_BASIC SENTENCE")
    gen_sent_basic = matrix_gen.read_text_sentence(None, regex_clean, texte=text_basic)
    mat_tfidf_sent_basic = matrix_gen.get_tfidf_matrix(gen_sent_basic)
    print(mat_tfidf_sent_basic)
    print("...48.VERIFICATION GET_TFIDF_MATRIX POUR TEXT_FILE SENTENCE")
    gen_sent_file = matrix_gen.read_text_sentence(list_file[0], regex_clean)
    mat_tfidf_sent_file = matrix_gen.get_tfidf_matrix(gen_sent_file)
    print(mat_tfidf_sent_file)
    print(" ")
    print("-------------VERIFICATION GET_TFIDF_MATRIX WINDOW----------------")
    print("...49.VERIFICATION GET_TFIDF_MATRIX POUR TEXT_TEST WINDOW")
    gen_wnd_test = matrix_gen.read_text_window(None, regex_clean, texte=text_test)
    mat_tfidf_wnd_test = matrix_gen.get_tfidf_matrix(gen_wnd_test)
    print(mat_tfidf_wnd_test)
    print("...50.VERIFICATION GET_TFIDF_MATRIX POUR TEXT_BASIC WINDOW")
    gen_wnd_basic = matrix_gen.read_text_window(None, regex_clean, texte=text_basic)
    mat_tfidf_wnd_basic = matrix_gen.get_tfidf_matrix(gen_wnd_basic)
    print(mat_tfidf_wnd_basic)
    print("...51.VERIFICATION GET_TFIDF_MATRIX POUR TEXT_TEST WINDOW")
    gen_wnd_file = matrix_gen.read_text_window(list_file[0], regex_clean, window=50)
    mat_tfidf_wnd_file = matrix_gen.get_tfidf_matrix(gen_wnd_file)
    print(mat_tfidf_wnd_file)
    print(" ")
    print("-------------VERIFICATION GET_TFIDF_MATRIX SLIDING WINDOW----------------")
    print("...52.VERIFICATION GET_TFIDF_MATRIX POUR TEXT_TEST SLIDING WINDOW")
    gen_swnd_test = matrix_gen.read_text_sliding_window(None, regex_clean, texte=text_test)
    mat_tfidf_swnd_test = matrix_gen.get_tfidf_matrix(gen_swnd_test)
    print(mat_tfidf_swnd_test)
    print("...53.VERIFICATION GET_TFIDF_MATRIX POUR TEXT_BASIC SLIDING WINDOW")
    gen_swnd_basic = matrix_gen.read_text_sliding_window(None, 
                                                          regex_clean, 
                                                          texte=text_basic,
                                                          window=5,
                                                          step=2)
    mat_tfidf_swnd_basic = matrix_gen.get_tfidf_matrix(gen_swnd_basic)
    print(mat_tfidf_swnd_basic)
    print("...54.VERIFICATION GET_TFIDF_MATRIX POUR TEXT_FILE SLIDING WINDOW")
    gen_swnd_file = matrix_gen.read_text_sliding_window(list_file[0], 
                                                          regex_clean, 
                                                          window=50,
                                                          step=10)
    mat_tfidf_swnd_file = matrix_gen.get_tfidf_matrix(gen_swnd_file)
    print(mat_tfidf_swnd_file)
    print(" ")
    print("-------------VERIFICATION GET_COUNTER_MATRIX FULL_TEXT----------------")
    print("...55.VERIFICATION GET_COUNTER_MATRIX POUR TEXT_TEST FULL TEXT")
    mat_counter_full_test = matrix_gen.get_counter_matrix(text_full_test, 
                                                        preprocess=True, 
                                                        filter_by_length=0,
                                                        stop_words=stop_words_null)
    print(mat_counter_full_test)
    print("...56.VERIFICATION GET_COUNTER_MATRIX POUR TEXT_BASIC FULL TEXT")
    mat_counter_full_basic = matrix_gen.get_counter_matrix(text_full_basic)
    print(mat_counter_full_basic)
    print("...57.VERIFICATION GET_COUNTER_MATRIX POUR TEXT_FILE FULL TEXT")
    mat_counter_full_file = matrix_gen.get_counter_matrix(text_full_file)
    print(mat_counter_full_file)
    print(" ")
    print("-------------VERIFICATION GET_COUNTER_MATRIX SENTENCE----------------")
    print("...58.VERIFICATION GET_COUNTER_MATRIX POUR TEXT_TEST SENTENCE")
    gen_sent_test = matrix_gen.read_text_sentence(None, regex_clean, texte=text_test)
    mat_counter_sent_test = matrix_gen.get_counter_matrix(gen_sent_test, 
                                                        preprocess=True, 
                                                        filter_by_length=0,
                                                        stop_words=stop_words_null)
    print(mat_counter_sent_test)
    print("...59.VERIFICATION GET_COUNTER_MATRIX POUR TEXT_BASIC SENTENCE")
    gen_sent_basic = matrix_gen.read_text_sentence(None, regex_clean, texte=text_basic)
    mat_counter_sent_basic = matrix_gen.get_counter_matrix(gen_sent_basic)
    print(mat_counter_sent_basic)
    print("...60.VERIFICATION GET_COUNTER_MATRIX POUR TEXT_FILE SENTENCE")
    gen_sent_file = matrix_gen.read_text_sentence(list_file[0], regex_clean)
    mat_counter_sent_file = matrix_gen.get_counter_matrix(gen_sent_file)
    print(mat_counter_sent_file)
    print(" ")
    print("-------------VERIFICATION GET_COUNTER_MATRIX WINDOW----------------")
    print("...61.VERIFICATION GET_COUNTER_MATRIX POUR TEXT_TEST WINDOW")
    gen_wnd_test = matrix_gen.read_text_window(None, regex_clean, texte=text_test)
    mat_counter_wnd_test = matrix_gen.get_counter_matrix(gen_wnd_test)
    print(mat_counter_wnd_test)
    print("...62.VERIFICATION GET_COUNTER_MATRIX POUR TEXT_BASIC WINDOW")
    gen_wnd_basic = matrix_gen.read_text_window(None, regex_clean, texte=text_basic)
    mat_counter_wnd_basic = matrix_gen.get_counter_matrix(gen_wnd_basic)
    print(mat_counter_wnd_basic)
    print("...63.VERIFICATION GET_COUNTER_MATRIX POUR TEXT_TEST WINDOW")
    gen_wnd_file = matrix_gen.read_text_window(list_file[0], regex_clean, window=50)
    mat_counter_wnd_file = matrix_gen.get_counter_matrix(gen_wnd_file)
    print(mat_counter_wnd_file)
    print(" ")
    print("-------------VERIFICATION GET_COUNTER_MATRIX SLIDING WINDOW----------------")
    print("...64.VERIFICATION GET_COUNTER_MATRIX POUR TEXT_TEST SLIDING WINDOW")
    gen_swnd_test = matrix_gen.read_text_sliding_window(None, regex_clean, texte=text_test)
    mat_counter_swnd_test = matrix_gen.get_counter_matrix(gen_swnd_test)
    print(mat_counter_swnd_test)
    print("...65.VERIFICATION GET_COUNTER_MATRIX POUR TEXT_BASIC SLIDING WINDOW")
    gen_swnd_basic = matrix_gen.read_text_sliding_window(None, 
                                                          regex_clean, 
                                                          texte=text_basic,
                                                          window=5,
                                                          step=2)
    mat_counter_swnd_basic = matrix_gen.get_counter_matrix(gen_swnd_basic)
    print(mat_counter_swnd_basic)
    print("...66.VERIFICATION GET_COUNTER_MATRIX POUR TEXT_FILE SLIDING WINDOW")
    gen_swnd_file = matrix_gen.read_text_sliding_window(list_file[0], 
                                                          regex_clean, 
                                                          window=50,
                                                          step=10)
    mat_counter_swnd_file = matrix_gen.get_counter_matrix(gen_swnd_file)
    print(mat_counter_swnd_file)
    print(" ")
    print(" ")
    print("-------------VERIFICATION GET_DIRECTED_MATRIX WINDOW POUR TOUT LE CORPUS----------------")
    start = time.time()
    list_time = []
    window_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    file = list_file[0]
    for wind in window_list:
        startl = time.time()
        text_full_file = matrix_gen.read_text_full(file, regex_clean)
        vocab_file = matrix_gen.get_vocab(text_full_file[0],
                                          filter_by_length=2,
                                          stop_words=stop_words)
        gen_count_wnd_file = matrix_gen.counter_by_window(text_full_file[0], 
                                                          vocab_file, 
                                                          window=wind)
        
        mat_dir_wnd_file = matrix_gen.get_directed(gen_count_wnd_file, 
                                                    vocab_file, 
                                                    option="")
        stopl = time.time()
        print(mat_dir_wnd_file)
        list_time.append(stopl - startl)
        print(" ")
    stop = time.time()
    print(f"temps total pour traiter l'ensemble des textes {stop -start}")
    print(" ")
    import matplotlib.pyplot as plt
    plt.plot(window_list, list_time)
    plt.xlabel("taille de la fenêtre")
    plt.ylabel("temps de traitement(s)")
    plt.show()
    print("-------------VERIFICATION GET_DIRECTED_MATRIX WINDOW POUR TOUT LE CORPUS----------------")
    start = time.time()
    dict_corpus_direct = {}
    for file in list_file:
        text_full_file = matrix_gen.read_text_full(file, regex_clean)
        vocab_file = matrix_gen.get_vocab(text_full_file[0],
                                          filter_by_length=2,
                                          stop_words=stop_words)
        gen_count_wnd_file = matrix_gen.counter_by_window(text_full_file[0], 
                                                          vocab_file, 
                                                          window=50)
        
        mat_dir_wnd_file = matrix_gen.get_directed(gen_count_wnd_file, 
                                                    vocab_file, 
                                                    option="")
        print(mat_dir_wnd_file)
        name_file = file.split("/")[-1].split(".")[0]
        dict_corpus_direct[name_file] = mat_dir_wnd_file
        print(" ")
    stop = time.time()
    print(f"temps total pour traiter l'ensemble des textes {stop -start}")
    print(" ")
    # print("-------------SAUVEGARDE DES DONNEES-------------")
    # for key, df in dict_corpus_direct.items():
    #     filename = os.path.join(save_path, key+".csv")
    #     save_result(df, filename, option="csv")
    # print(" ")
    print("-------------METRIQUES POUR 2 VECTEURS DE 2 DATES----------------")
    df_1483 = dict_corpus_direct["1483"]
    df_1498 = dict_corpus_direct["1498"]
    
    columns_1483 = list(df_1483.columns)
    columns_1498 = list(df_1498.columns)
    vocab_commun = list(set(columns_1483).intersection(set(columns_1498)))
    #print(vocab_commun)
    dieu_1483 = df_1483.loc["dieu", vocab_commun]
    #print(dieu_1483)
    dieu_1498 = df_1498.loc["dieu", vocab_commun]
    dieu_nofiltre_1483 = df_1483["dieu"]
    dieu_nofiltre_1498 = df_1498["dieu"]
    print("------------VERIFICATION DE LA FONCTION COMPARE_TRANSFORM DE METRIQUE----------------")
    u, v, vocab_com = compare_transform(dieu_nofiltre_1483, dieu_nofiltre_1498, option="filtre")
    assert u.all() == dieu_1483.all()
    dieu_both_trans, _ = compare_transform(dieu_nofiltre_1483, dieu_nofiltre_1498, option="transform", name1="dieu_1483", name2="dieu_1498")
    dieu_both, _ = compare_transform(dieu_nofiltre_1483, dieu_nofiltre_1498, option="no transform", name1="dieu_1483", name2="dieu_1498")
    print(dieu_both_trans)
    print(dieu_both)
    print("------------VERIFICATION DE LA FONCTION CALCULATE_DICE----------------")
    print("indice de dice pour dieu 1483 et 1498", calculate_dice(dieu_1483, dieu_1498))
    print("indice de dice pour dieu 1483 et 1498 compare transform", calculate_dice(dieu_both.dieu_1483, dieu_both.dieu_1498))
    print("-------------METRIQUE COSINE POUR 2 VECTEURS DE 2 DATES")
    print("similarité cosinus pour dieu 1483 et 1498", calculate_cosine(dieu_1483, dieu_1498))
    print("similarité cosinus pour dieu 1483 et 1498 compare transform", calculate_cosine(dieu_both.dieu_1483, dieu_both.dieu_1498))
    print("-------------METRIQUE JACCARD POUR 2 VECTEURS DE 2 DATES")
    print("similarité jaccard pour dieu 1483 et 1498", calculate_jaccard(dieu_1483, dieu_1498))
    print("similarité jaccard pour dieu 1483 et 1498 compare transform", calculate_jaccard(dieu_both.dieu_1483, dieu_both.dieu_1498))
    print("-------------METRIQUE PEARSON POUR 2 VECTEURS DE 2 DATES")
    print("Correlation Pearson pour dieu 1483 et 1498", calculate_pearson(dieu_1483, dieu_1498)[0])
    print("Correlation Pearson pour dieu 1483 et 1498 compare transform", calculate_pearson(dieu_both.dieu_1483, dieu_both.dieu_1498)[0])
    print("-------------METRIQUE LOG LIKELIHOOD POUR 2 VECTEURS DE 2 DATES")
    print("Similarité likelihood pour dieu 1483 et 1498", calculate_likelihood(dieu_1483, dieu_1498))
    print("Similarité likelihood pour dieu 1483 et 1498 compare transform", calculate_likelihood(dieu_both.dieu_1483, dieu_both.dieu_1498))
    print("-------------METRIQUE ENTROPY POUR 1 VECTEUR")
    print("Entropy pour dieu 1483", calculate_approximate_entropy(dieu_1483))
    print("Entropy pour dieu 1483 compare transform", calculate_approximate_entropy(dieu_both.dieu_1483))
    print("-------------METRIQUE MUTUAL INFORMATION SCORE POUR 2 VECTEURS DE 2 DATES")
    print("Score Mutual Information pour dieu 1483 et 1498", calculate_mutual_information(dieu_1483, dieu_1498))
    print("Score Mutual Informationpour dieu 1483 et 1498 compare transform", calculate_mutual_information(dieu_both_trans.dieu_1483, dieu_both_trans.dieu_1498))



    
    