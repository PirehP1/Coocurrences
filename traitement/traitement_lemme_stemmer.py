#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:37:39 2021

@author: audrey quessada vial
this script aims to preprocess texts and get tokens
"""
__version__ = "0.1.0"
__authors__ = "Audrey Quessada Vial"
__contact__ = "audrey.qvial@gmail.com"


import sys
sys.path.append("../")
import re
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from nltk.stem.snowball import FrenchStemmer
import itertools
import os
import functools
import time


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer



class Preprocessing:
    """
    """
    
    def __init__(self,
                 input_path=None,
                 window=1, 
                 stop_words=[]):
        self.input_path = input_path # chemin d'un dossier ou d'un fichier à traiter, str
        self.window = window #taille de la fenêtre à considérer, int
        self.stop_words = stop_words
        
    #TODO décommenter le decorateur pour avoir les performances d'execution
    #@timer 
    def get_files(self, ext="txt"):
        """
        Parameters
        ----------
        ext : str, optional
            The default is "txt" extension du fichier par exemple "txt"

        Returns
        -------
        list
            liste des chemins des fichiers à traiter

        """
        if self.input_path.endswith(ext):
            print("=> traitement d'un seul fichier")
            return [self.input_path]
        else:
            list_file = []
            for root, dirs, files in os.walk(self.input_path, topdown=True):
                for name in files:
                    if name.endswith(ext):
                        list_file.append(os.path.join(root, name))
            return list_file
        
        
    #@timer    
    def basic_text_cleaning(self, text, regex_clean, remove_punctuation=True):
        """

        Parameters
        ----------
        text : str
            texte sur lequel on veut effectuer un nettoyage basique.
        regex_clean : str
            regex à utiliser pour le nettoyage
        remove_punctuation: bool, optional
            

        Returns
        -------
        text : str
            DESCRIPTION.

        """
        text = text.strip()
        if remove_punctuation == True:
            #transformer = str.maketrans(" "," ", string.punctuation)
            #text = text.translate(transformer)
            text = re.sub("[',;:!\\/.?{}\|\-_§&#()\[\]@<>\+\-\*%]", " ", text)
        text = re.sub(regex_clean, "", text)
        text = re.sub("(\s{2,})", " ", text)
        return text
    
    

    #@timer
    def get_token_window(self, gen_window, filter_by_length=2, process="", joint=False):
        """
        

        Parameters
        ----------
        gen_window : TYPE
            DESCRIPTION.
        filter_by_length : TYPE, optional
            DESCRIPTION. The default is 2.
        process : TYPE, optional
            DESCRIPTION. The default is "". Les autres valeurs sont "Lemme" 
            (pour faire appel à un Lemmatizer) et "Stem" pour faire appel à un stemmer
        joint : TYPE, optional
            DESCRIPTION. The default is False

        Yields
        ------
        list_token : TYPE
            DESCRIPTION.

        """
        lemmatizer = FrenchLefffLemmatizer()
        stemmer = FrenchStemmer()
        for wind in gen_window:
            doc = wind.split(" ")
            if doc != []:
                list_token = []
                for token in doc:
                    token = re.sub("(\s{2})?", "", token)
                    if (token not in self.stop_words): #on filtre les mots qui ne sont pas des stop words et dont la longueur est > 2
                        if len(token) > filter_by_length: 
                            if process == "Lemme":
                                token = lemmatizer.lemmatize(token.lower())
                            elif process == "Stem":
                                token = stemmer.stem(token.lower())
                            else:
                                token = token.lower()
                            list_token.append(token)
                if list_token != []:
                    if joint == True:
                        yield " ".join(list_token)
                    else:
                        yield list_token

                        
      
                    
    #@timer                
    def get_token(self, text, filter_by_length=2, process=""):
        """
        

        Parameters
        ----------
        text : TYPE
            DESCRIPTION.
        filter_by_length : TYPE, optional
            DESCRIPTION. The default is 2.

        Yields
        ------
        TYPE
            DESCRIPTION.

        """
        lemmatizer = FrenchLefffLemmatizer()
        stemmer = FrenchStemmer()
        doc = text.split(" ")
        if doc == []:
            print("empty doc, exit system")
            sys.exit()
        for token in doc:
            token = re.sub("(\s{2})?", "", token)
            if (token not in self.stop_words): #on filtre les mots qui ne sont pas des stop words et dont la longueur est > 2
                if len(token) > filter_by_length:
                    if process == "Lemme":
                        token = lemmatizer.lemmatize(token.lower())
                    elif process == "Stem":
                        token = stemmer.stem(token.lower())
                    else:
                        token = token.lower()
                    yield token
                    
                    

    #@timer
    def read_text_full(self, file, regex_clean, **kwargs):
        """
        

        Parameters
        ----------
        file : TYPE
            DESCRIPTION.
        regex_clean : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        print("=> reading full text")
        if file is not None:
            with open(file, "r") as f:
                doc = f.read()
        else:
            doc = kwargs.get("texte", None)
        if doc is None:
            print("empty file or text string")
            sys.exit()
        return self.basic_text_cleaning(doc, regex_clean)
    
    
    
    #@timer
    def read_text_sentence(self, file, regex_clean, **kwargs):
        """
        

        Parameters
        ----------
        file : TYPE
            DESCRIPTION.
        regex_clean : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Yields
        ------
        TYPE
            DESCRIPTION.

        """

        print("=> reading by sentence")
        if file is not None:
            with open(file, "r") as f:
                doc = f.read()
        else:
            doc = kwargs.get("texte", None)
            
        if doc is None:
            print("empty file or text string")
            sys.exit()
        punct = kwargs.get("punct", ".")
        if punct not in doc:
            print(f"cannot split into sentences with this punctuation {punct}, please indicate another punctuation")
            sys.exit()
        lines = doc.split(punct)
        for line in lines:
            yield self.basic_text_cleaning(line, regex_clean)
            
                
    #@timer            
    def read_text_window(self, file, regex_clean, **kwargs):
        """
        

        Parameters
        ----------
        file : TYPE
            DESCRIPTION.
        regex_clean : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Yields
        ------
        chunk : TYPE
            DESCRIPTION.

        """

        print("= reading by window")
        if file is not None:
            with open(file, "r") as f:
                doc = f.read()
        else:
            doc = kwargs.get("texte", None)
            
        if doc is None:
            print("empty file or text string")
            sys.exit()
        doc = self.basic_text_cleaning(doc, regex_clean)
        list_words = doc.split(" ")
        if len(list_words) < self.window:
            print("text size lower than window size, please select another window size")
            sys.exit()
        list_window = []
        nb_chunks = len(list_words) // self.window
        for i in range(nb_chunks):
            chunk = list_words[i*self.window:(i+1)*self.window]
            list_window.append(chunk)
        last_chunk = list_words[nb_chunks*self.window:]
        list_window.append(last_chunk)
        for chunk in list_window:
            yield " ".join(chunk)
   

    #@timer
    def read_text_sliding_window(self, file, regex_clean, step=1, **kwargs):

        print("= reading by sliding window")
        if file is not None:
            with open(file, "r") as f:
                doc = f.read()
        else:
            doc = kwargs.get("texte", None)
            
        if doc is None:
            print("empty file or text string")
            sys.exit()
        doc = self.basic_text_cleaning(doc, regex_clean)
        list_words = doc.split(" ")
        if len(list_words) < self.window:
            print("text size lower than window size, please select another window size")
            sys.exit()
        list_window = []
        for i in range(0, len(list_words), step):
            chunk = " ".join(list_words[i:i+self.window+1])
            yield chunk
            
                   
#@timer                    
def read_stop_word(path):
    """
    

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    with open(path, "r") as f:
        stop_words = f.readlines()
    f.close()
    return [words.strip() for words in stop_words]
                  
                        
                    
                
if __name__ == "__main__":
    input_path = "../Data/corpusbytime"
    path_stop_words = "../Data/stop_words.txt"
    stop_words = read_stop_word(path_stop_words)
    #print(stop_words)
    preproc = Preprocessing(input_path=input_path,
                            window=5,
                            stop_words=stop_words)
    print(r"....TEST FONCTION GET_FILES")
    list_file = preproc.get_files()
    print(list_file)
    #definition d'une regex
    regex_clean = "(\')?(\d*)?(\\n)"
    print("----------------")
    print(r"....TEST FONCTION BASIC_TEST_CLEANING AVEC PONCTUATION")
    text_basic = """Le but de toute association politique est la conservation des droits naturels et imprescriptibles de l'Homme. 
    Ces droits sont la liberté, la propriété, la sûreté, et la résistance à l'oppression.<>?;:!/§+-*&@{}[|]()"""
    text_basic1 = preproc.basic_text_cleaning(text_basic, regex_clean, remove_punctuation=True)
    print(text_basic1)
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION BASIC_TEST_CLEANING SANS PONCTUATION")
    text_basic2 = preproc.basic_text_cleaning(text_basic, regex_clean, remove_punctuation=False)
    print(text_basic2)
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION READ_TEXT_FULL SANS KWARGS")
    doc = preproc.read_text_full(list_file[0], regex_clean)
    print(doc[:10])
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION READ_TEXT_FULL AVEC KWARGS")
    doc1 = preproc.read_text_full(None, regex_clean, texte=text_basic)
    print(doc1)
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION GET_TOKEN AVEC FULL TEXT")
    gen_tok_full = preproc.get_token(doc1, filter_by_length=2) #filter_by_length can be  0
    print(list(gen_tok_full))
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION READ_TEXT_SENTENCE SANS KWARGS")
    gen_sent = preproc.read_text_sentence(list_file[0], regex_clean)
    print(next(gen_sent))
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION READ_TEXT_SENTENCE AVEC KWARGS")
    gen_sent1 = preproc.read_text_sentence(None, regex_clean, texte=text_basic)
    print(next(gen_sent1))
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION GET_TOKEN_WINDOW AVEC SENTENCE SANS LEMMATIZER STEMMER")
    gen_sent1 = preproc.read_text_sentence(None, regex_clean, texte=text_basic)
    tok_sent = preproc.get_token_window(gen_sent1, filter_by_length=2, process="")
    print(next(tok_sent))
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION GET_TOKEN_WINDOW AVEC SENTENCE AVEC LEMMATIZER")
    gen_sent1 = preproc.read_text_sentence(None, regex_clean, texte=text_basic)
    tok_sent = preproc.get_token_window(gen_sent1, filter_by_length=2, process="Lemme")
    print(next(tok_sent))
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION GET_TOKEN_WINDOW AVEC SENTENCE AVEC STEMMER")
    gen_sent1 = preproc.read_text_sentence(None, regex_clean, texte=text_basic)
    tok_sent = preproc.get_token_window(gen_sent1, filter_by_length=2, process="Stem")
    print(next(tok_sent))
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION READ_TEXT_WINDOW SANS KWARGS")
    gen_wind = preproc.read_text_window(list_file[0], regex_clean)
    print(next(gen_wind))
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION READ_TEXT_WINDOW AVEC KWARGS")
    gen_wind1 = preproc.read_text_window(None, regex_clean, texte=text_basic)
    print(next(gen_wind1))
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION GET_TOKEN_WINDOW AVEC WINDOW")
    tok_wind = preproc.get_token_window(gen_wind1, filter_by_length=2)
    print(next(tok_wind))
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION READ_TEXT_SLIDING_WINDOW SANS KWARGS")
    gen_slwind = preproc.read_text_sliding_window(list_file[0], regex_clean, step=1)
    print(next(gen_slwind))
    print("----------------")
    print(" ")
    print(r"....TEST FONCTION READ_TEXT_SLIDING_WINDOW AVEC KWARGS")
    gen_slwind1 = preproc.read_text_sliding_window(None, regex_clean, step=2, texte=text_basic)
    tok_slwind1 = preproc.get_token_window(gen_slwind1, filter_by_length=2, process="", joint=True)
    for tok in tok_slwind1:
        print(tok)
    
       
        
    
            

