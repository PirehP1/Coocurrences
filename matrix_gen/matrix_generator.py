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
import collections
import itertools
import scipy.sparse as sp
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from traitement.traitement import Preprocessing, read_stop_word
from metrique.metrique import *



class MatrixGenerator(Preprocessing):
    """
    """
    def __init__(self, *args, **kwargs):
        super(MatrixGenerator, self).__init__(*args, **kwargs)
        
                       
    #@timer
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
            if (token in stopwords) or (len(token)<= filter_by_length) or (token==""):
                pass
            else:
                if token in vocab.keys():
                    pass
                else:
                    vocab[token] = index
                    index += 1
        return vocab
    
    
    #@timer
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
        
        list_words = []
        for token in text_clean.split(separator):
            if (token in stopwords) or (len(token)<= filter_by_length) or (token==""):
                pass
            else:
                list_words.append(token)
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
        
  
    #@timer
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
    
  
   # @timer    
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
     
            
    #@timer 
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
            
            
    #@timer        
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
            
    
    #@timer
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
                        dir_mat[i, j] += count2 #ou count1, sinon il faut prendre la transposée
        
        return pd.DataFrame(dir_mat, columns=list_vocab, index=list_vocab)  
      
    
    #@timer
    def get_non_directed(self, data, vocab, **kwargs):
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
        indptr = [0]
        for i, d in enumerate(list_token):
            for term in d:
                if term in vocab.keys():
                    index = vocab[term]
                    indices.append(index)
                    data.append(1)
            indptr.append(len(indices))
        mat = sp.csr_matrix((data, indices, indptr), dtype=int)
        mat_coocs = np.dot(mat.T, mat)
        return pd.DataFrame(data=mat_coocs.toarray(), columns=vocab.keys(), index=vocab.keys())
    
 
#TODO create your own word analyzer ofr Counter and TFIDF matrices    
    #@timer
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
    
        
    #@timer
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


    #@timer
    def specificite(self, data, vocab, matrix, counter, option="logfrac", seuil=0.5):
        sum_rows = matrix.sum(axis=1).tolist()
        list_token = matrix.columns.tolist()
        T = sum(counter.values())
        final = np.zeros_like(matrix)
        if option =="cosine":
            return pd.DataFrame(calculate_cosine(matrix.values), index=list_token, columns=list_token)
        
        for count in data:
            for ((token1, count1), (token2, count2)) in itertools.permutations(count.items(),2): 
                if token1 in matrix.columns and token2 in matrix.columns:
                    i = vocab[token1]
                    j = vocab[token2]
                    t = sum_rows[i] # comprends toujours pas                
                    F = counter[token2]
                    f = count2 #tester si c'est count de 1 ou 2

                    if token1 != token2:
                        if option == "combinatory":
                            final[i, j] = hypergeometric(F, f, t, T, option="combinatory")
                        elif option == "scipy":
                            final[i, j] = hypergeometric(F, f, t, T, option="scipy")
                        elif option == "ramanujan":
                            final[i, j] = hypergeometric(F, f, t, T, option="ramanujan")
                        elif option == "logfrac":
                            final[i, j] = hypergeometric(F, f, t, T, option="logfrac")
                        elif option == "spec":
                            final[i, j] = ca_coeff_spec(T, t, F, f, seuil, option="logfrac") 
                        elif option == "dice":
                            final[i, j] = coeffdice(f,t,F)
        return pd.DataFrame(final, index=list_token, columns=list_token)
    
    
    #@timer
    def get_adjacency(self, matrix):
        adj = matrix.copy()
        adj[adj > 0] = 1
        return adj

# FONCTIONS UTILES



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


def concatenate_matrix(list_matrix, list_file):
    """
    Create a big matrix

    Parameters
    ----------
    list_matrix : TYPE list
        DESCRIPTION. list of the matrices we want to concatenate
    list_file : TYPE list
        DESCRIPTION. list of the filenames

    Returns
    -------
    pandas DataFrame
    """
    list_name = [file.split("/")[-1].split(".")[0] for file in list_file]
    for df, name in zip(list_matrix, list_name):
        df["Date_Fichier"] = [name]*df.shape[0]
    return pd.concat(list_matrix, axis=0)
    
        
      

