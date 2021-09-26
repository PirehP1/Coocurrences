#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:37:39 2021

@author: Audrey Quessada Vial, Stephane Lamassé
Institutions: Pireh-Lamop
LICENCE GNU
This script aims to process the text data
"""

__version__ = "0.1.0"
__authors__ = "Audrey Quessada Vial","Stephane Lamassé"



import sys
sys.path.append("../")
import re
import itertools
import os
import functools
import time
import argparse


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
                 stop_words=[],
                 remove_punctuation=True,
                 filter_by_length=2,
                 separator=" ",
                 case=True):
        self.input_path = input_path # chemin d'un dossier ou d'un fichier à traiter, str
        self.window = window #taille de la fenêtre à considérer, int
        self.stop_words = stop_words #liste des stop_words
        self.remove_punctuation = remove_punctuation #bool pour enlever la ponctuation
        self.filter_by_length = filter_by_length #taille des mots à filtrer si on veut les enlever
        self.separator = separator #separateur pour récupérer les tokens
        self.case = case #si on veut tout mettre en minuscule
        
    #TODO décommenter le decorateur pour avoir les performances d'execution
    #@timer 
    def get_files_simple(self, ext="txt"):
        """        
        This function retrieves files paths from a root folder
        Parameters
        ----------
        ext : TYPE, optional
            DESCRIPTION. The default is "txt". ext is the extension of the files

        Returns
        -------
        None.

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

                            
    def get_files_bigvolume(self, ext="txt"):
        """
        
        This function retrieves files paths from a root folder when the number of files is important
        Parameters
        ----------
        ext : TYPE, optional
            DESCRIPTION. The default is "txt". ext is the extension of the files

        Returns
        -------
        None.

        """
        print("= traitement d'un gros volume de fichiers")
        for root, dirs, files in os.walk(self.input_path, topdown=True):
            for name in files:
                if name.endswith(ext):
                    yield os.path.join(root, name)                
        
        
    #@timer    
    def basic_text_cleaning(self, text, regex_clean, **kwargs):
        """
        
        This function performs a basic text cleaning according to a specific regex and punctuation removal
        Parameters
        ----------
        text : TYPE, str
            DESCRIPTION. Text tro clean
        regex_clean : TYPE, str
            DESCRIPTION. Regex to apply on the text

        **kwargs : TYPE
            remove_punctuation: bool, if we want to remove punctuation
            regex_punctuation: str, regex to remove desire punctuation
            case: bool, if we want to lower cases
            DESCRIPTION.

        Returns
        -------
        text : TYPE
            DESCRIPTION.

        """

        regex_punctuation = kwargs.get("regex_punctuation", "[',;:!\\/.?{}\|\-_§&#()\[\]@<>\+\-\*%]")
        remove_punctuation = kwargs.get("remove_punctuation", self.remove_punctuation)
        case = kwargs.get("case", self.case)
        text = text.strip()
        if case == True:
            text = text.lower()
        if remove_punctuation == True:
            text = re.sub(regex_punctuation, " ", text)
        text = re.sub(regex_clean, "", text)
        text = re.sub("(\s{2,})", " ", text)
        return text
    
    
    #TODO ajouter POS-tags éventuellement dans le preprocessing
    #@timer
    def get_token_window(self, generator, joint=False, **kwargs):
        """
        This function retrieves all the token per text chunk with the desired window-size

        Parameters
        ----------
        generator : TYPE
            DESCRIPTION. generator of sentences or generator of windows, or generator of sliding windows
        filter_by_length : TYPE, optional
            DESCRIPTION. The default is 2.
        joint : TYPE, optional
            DESCRIPTION. The default is False. If we want the str (joint=True) or the list of token
        **kwargs : TYPE
            separator: str, separator used to get the token from a text. 
            For more complex languages, please design a new function
            stopwords: list of stop_words to use
            filter_by_length: int, if we want to filter the tokens by their length
            DESCRIPTION.

        Yields
        ------
        TYPE
            DESCRIPTION.

        """

        separator = kwargs.get("separator", self.separator)
        stopwords = kwargs.get("stop_words", self.stop_words)
        filter_by_length = kwargs.get("filter_by_length", self.filter_by_length)
        
        for wind in generator:
            doc = wind.split(separator)
            if doc != []:
                list_token = []
                for token in doc:
                    token = re.sub("(\s{2})?", "", token)
                    if (token not in stopwords): #on filtre les mots qui ne sont pas des stop words et dont la longueur est > 2
                        if len(token) >= filter_by_length:
                            list_token.append(token)
                if list_token != []:
                    if joint == True:
                        yield " ".join(list_token)
                    else:
                        yield list_token

                    
    #@timer                
    def get_token(self, data, **kwargs):
        """
        
        This function retrieves all the token in the text
        Parameters
        ----------
        data : TYPE generator or iterator of text
            DESCRIPTION.
        filter_by_length : TYPE, optional
            DESCRIPTION. The default is 2.
        type_data : TYPE, optional
            DESCRIPTION. The default is "full_text".
        **kwargs : TYPE
            separator: str, separator used to get the token from a text. 
            For more complex languages, please design a new function
            stopwords: list of stop_words to use
            filter_by_length: int, if we want to filter the tokens by their length
            DESCRIPTION.
            
        Yields
        ------
        token : TYPE
            DESCRIPTION. generator of token

        """
        separator = kwargs.get("separator", self.separator)
        stopwords = kwargs.get("stop_words", self.stop_words)
        filter_by_length = kwargs.get("filter_by_length", self.filter_by_length)

        for doc in data:
            doc = doc.split(separator)
            if doc != []:
                for token in doc:
                    token = re.sub("(\s{2})?", "", token) #on enlève les espaces supplémentaires quand il y en a
                    if token not in stopwords: #on filtre les mots qui ne sont pas des stop words et dont la longueur est > 2
                        if len(token) >= filter_by_length:
                            yield token


    #@timer
    def read_text_full(self, file, regex_clean, **kwargs):
        """
        This function reads a full text and clean it with the function basic cleaning
        Parameters
        ----------
        file : TYPE
            DESCRIPTION.
        regex_clean : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            remove_punctuation: bool, if we want to remove punctuation
            regex_punctuation: str, regex to remove desire punctuation
            case: bool, if we want to lower cases
        Returns
        -------
        TYPE
            DESCRIPTION. list of str

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
        remove_punctuation = kwargs.get("remove_puntuation", self.remove_punctuation)
        regex_punctuation = kwargs.get("regex_punctuation", "[',;:!\\/.?{}\|\-_§&#()\[\]@<>\+\-\*%]")
        case = kwargs.get("case", self.case)
        return [self.basic_text_cleaning(doc, 
                                         regex_clean, 
                                         case=case,
                                         remove_punctuation=remove_punctuation, 
                                         regex_punctuation=regex_punctuation)]
    
    
    
    #@timer
    def read_text_sentence(self, file, regex_clean, **kwargs):
        """
        This function extract sentences from a text given a specific separator

        Parameters
        ----------
        file : TYPE
            DESCRIPTION.
        regex_clean : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            remove_punctuation: bool, if we want to remove punctuation
            regex_punctuation: str, regex to remove desire punctuation
            case: bool, if we want to lower cases
            punct: str, separator used to extract sentences
            DESCRIPTION.

        Yields
        ------
        TYPE
            DESCRIPTION.

        """

        print("=> reading by sentence")
        remove_punctuation = kwargs.get("remove_puntuation", self.remove_punctuation)
        regex_punctuation = kwargs.get("regex_punctuation", "[',;:!\\/.?{}\|\-_§&#()\[\]@<>\+\-\*%]")
        case = kwargs.get("case", self.case)
        punct = kwargs.get("punct", ".")
        if file is not None:
            with open(file, "r") as f:
                doc = f.read()
        else:
            doc = kwargs.get("texte", None) 
        if doc is None:
            print("empty file or text string")
            sys.exit()
       
        if punct not in doc:
            print(f"cannot split into sentences with this punctuation {punct}, please indicate another punctuation")
            sys.exit()
        lines = doc.split(punct)
        for line in lines:
            yield self.basic_text_cleaning(line, 
                                           regex_clean, 
                                           case=case,
                                           remove_punctuation=remove_punctuation,
                                           regex_punctuation=regex_punctuation)
            
                
    #@timer            
    def read_text_window(self, file, regex_clean, **kwargs):
        """
        This function reads a text by chunk of window size

        Parameters
        ----------
        file : TYPE
            DESCRIPTION.
        regex_clean : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.
            remove_punctuation: bool, if we want to remove punctuation
            regex_punctuation: str, regex to remove desire punctuation
            case: bool, if we want to lower cases
            separator: str, separator used to get the token from a text. 
            For more complex languages, please design a new function
            stopwords: list of stop_words to use
            filter_by_length: int, if we want to filter the tokens by their length
            window: int, the size of the window

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
        separator = kwargs.get("separator", self.separator)
        case = kwargs.get("case", self.case)
        stopwords = kwargs.get("stop_words", self.stop_words)
        filter_by_length = kwargs.get("filter_by_length", self.filter_by_length)
        remove_punctuation = kwargs.get("remove_puntuation", self.remove_punctuation)
        regex_punctuation = kwargs.get("regex_punctuation", "[',;:!\\/.?{}\|\-_§&#()\[\]@<>\+\-\*%]")
        window = kwargs.get("window", self.window)
        #ATTENTION, ON ENLEVE LA PONCTUATION AVANT DE RECUPERER LES TOKENS, 
        # du coup faire attention au type de separator ou à la regex de ponctuation 
        # pour qu'elle ne prenne pas en compte le separator
        doc = [self.basic_text_cleaning(doc, 
                                        regex_clean, 
                                        case=case,
                                        remove_punctuation=remove_punctuation,
                                        regex_punctuation=regex_punctuation)]
        list_words = list(self.get_token(doc, 
                                         filter_by_length=filter_by_length,
                                         separator=separator,  
                                         stop_words=stopwords))

        if len(list_words) < window:
            print("text size lower than window size, please select another window size")
            sys.exit()
        list_window = []
        nb_chunks = len(list_words) // window
        for i in range(nb_chunks):
            chunk = list_words[i*window:(i+1)*window]
            list_window.append(chunk)
        last_chunk = list_words[nb_chunks*window:]
        list_window.append(last_chunk)
        for chunk in list_window:
            yield " ".join(chunk)
   

    #@timer
    def read_text_sliding_window(self, file, regex_clean, step=1, **kwargs):
        """
        This function reads a text by sliding a window

        Parameters
        ----------
        file : TYPE
            DESCRIPTION.
        regex_clean : TYPE
            DESCRIPTION.
        step : TYPE, optional
            DESCRIPTION. The default is 1.
        **kwargs : TYPE
            remove_punctuation: bool, if we want to remove punctuation
            regex_punctuation: str, regex to remove desire punctuation
            case: bool, if we want to lower cases
            separator: str, separator used to get the token from a text. 
            For more complex languages, please design a new function
            stopwords: list of stop_words to use
            filter_by_length: int, if we want to filter the tokens by their length
            window: int, the size of the sliding window
            DESCRIPTION.

        Yields
        ------
        chunk : TYPE
            DESCRIPTION.

        """

        print("= reading by sliding window")
        if file is not None:
            with open(file, "r") as f:
                doc = f.read()
        else:
            doc = kwargs.get("texte", None)
            
        if doc is None:
            print("empty file or text string")
            sys.exit()
        separator = kwargs.get("separator", self.separator)
        case = kwargs.get("case", self.case)
        stopwords = kwargs.get("stop_words", self.stop_words)
        filter_by_length = kwargs.get("filter_by_length", self.filter_by_length)
        remove_punctuation = kwargs.get("remove_puntuation", self.remove_punctuation)
        regex_punctuation = kwargs.get("regex_punctuation", "[',;:!\\/.?{}\|\-_§&#()\[\]@<>\+\-\*%]")
        window = kwargs.get("window", self.window)
        doc = [self.basic_text_cleaning(doc, 
                                        regex_clean,
                                        case=case,
                                        remove_punctuation=remove_punctuation,
                                        regex_punctuation=regex_punctuation)]
        list_words = list(self.get_token(doc, 
                                         filter_by_length=filter_by_length,
                                         separator=separator, 
                                         stop_words=stopwords))
        if len(list_words) < window:
            print("text size lower than window size, please select another window size")
            sys.exit()
        list_window = []
        for i in range(0, len(list_words), step):
            chunk = " ".join(list_words[i:i+window+1])
            yield chunk
            
                   
#@timer                    
def read_stop_word(path):
    """
    This function reads a text file of stop words and returns a list of stop words

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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default=None, help="folder path of the text files, str")
    parser.add_argument('--path_stop', default=None, help="path of stop words list")
    parser.add_argument('--stopwords', default=None, help="list of stop words")
    parser.add_argument("--window", default=5, type=int, help="analysis window, nb of words") 
    parser.add_argument("--remove_punctuation", default=True, type=bool, help="boolean to know if we want to remove all the punctuation")
    parser.add_argument("--filter_by_length", default=0, type=int, help="value to filter words by their length")
    parser.add_argument("--separator", default=" ", type=str, help="simple separator to get tokens from text")
    parser.add_argument("--case", default=True, type=bool, help="boolean to know if we want to lower cases")
    args = parser.parse_args()
    if args.input_path == None:
        input_path = "../Data/corpusbytime"
    else:
        input_path = args.input_path
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
    preproc = Preprocessing(input_path=input_path,
                            window=window,
                            stop_words=stop_words,
                            remove_punctuation=remove_punctuation,
                            filter_by_length=filter_by_length,
                            separator=separator,
                            case=case)
    
    print(r"....TEST FONCTION GET_FILES")
    list_file = preproc.get_files_simple()
    print(list_file)
    gen_file = preproc.get_files_bigvolume()
    print(next(gen_file))
    #definition d'une regex
    regex_clean = "(\')?(\d*)(\\n)"
    print("----------------")
    print(r"....1.TEST FONCTION BASIC_TEST_CLEANING AVEC PONCTUATION")
    text_basic = """Le but de toute association politique est la conservation des droits naturels et imprescriptibles de l'Homme. 
    Ces droits sont la liberté, la propriété, la sûreté, et la résistance à l'oppression.<>?;:!/§+-*&@{}[|]()"""
    text_basic1 = preproc.basic_text_cleaning(text_basic, regex_clean, remove_punctuation=True)
    print(text_basic1)
    print("----------------")
    print(" ")
    print(r"....2.TEST FONCTION BASIC_TEST_CLEANING SANS PONCTUATION")
    text_basic2 = preproc.basic_text_cleaning(text_basic, regex_clean, remove_punctuation=False)
    print(text_basic2)
    print("----------------")
    print(" ")
    print(r"....3.TEST FONCTION READ_TEXT_FULL SANS KWARGS")
    doc = preproc.read_text_full(list_file[0], regex_clean)
    print(doc[0][:100])
    print("----------------")
    print(" ")
    print(r"....4.TEST FONCTION READ_TEXT_FULL AVEC KWARGS")
    doc1 = preproc.read_text_full(None, regex_clean, texte=text_basic)
    print(doc1)
    print("----------------")
    print(" ")
    print(r"....5.TEST FONCTION GET_TOKEN AVEC FULL TEXT")
    gen_tok_full = preproc.get_token(doc1, filter_by_length=2) #filter_by_length can be  0
    print(list(gen_tok_full))
    print("----------------")
    print(" ")
    print(r"....6.TEST FONCTION READ_TEXT_SENTENCE SANS KWARGS")
    gen_sent = preproc.read_text_sentence(list_file[0], regex_clean)
    print(next(gen_sent))
    print("----------------")
    print(" ")
    print(r"....7.TEST FONCTION READ_TEXT_SENTENCE AVEC KWARGS")
    gen_sent1 = preproc.read_text_sentence(None, regex_clean, texte=text_basic)
    print(next(gen_sent1))
    print("----------------")
    print(" ")
    print(r"....8.TEST FONCTION GET_TOKEN POUR SENTENCE AVEC KWARGS")
    gen_sent1 = preproc.read_text_sentence(None, regex_clean, texte=text_basic)
    tok_sent1 = preproc.get_token(gen_sent1)
    for tok in tok_sent1:
        print(tok)
    print("----------------")
    print(" ")
    print(r"....9.TEST FONCTION GET_TOKEN_WINDOW AVEC SENTENCE")
    gen_sent1 = preproc.read_text_sentence(None, regex_clean, texte=text_basic)
    tok_sent = preproc.get_token_window(gen_sent1, filter_by_length=2)
    for tok in tok_sent:
        print(tok)
    print("----------------")
    print(" ")
    print(r"....10.TEST FONCTION READ_TEXT_WINDOW SANS KWARGS")
    gen_wind = preproc.read_text_window(list_file[0], regex_clean)
    print(next(gen_wind))
    print("----------------")
    print(" ")
    print(r"....11.TEST FONCTION READ_TEXT_WINDOW AVEC KWARGS")
    gen_wind1 = preproc.read_text_window(None, regex_clean, texte=text_basic)
    print(next(gen_wind1))
    print("----------------")
    print(" ")
    print(r"....12.TEST FONCTION GET_TOKEN_WINDOW AVEC WINDOW")
    tok_wind = preproc.get_token_window(gen_wind1, filter_by_length=2)
    print(next(tok_wind))
    print("----------------")
    print(" ")
    print(r"....13.TEST FONCTION READ_TEXT_SLIDING_WINDOW SANS KWARGS")
    gen_slwind = preproc.read_text_sliding_window(list_file[0], regex_clean, step=1)
    print(next(gen_slwind))
    print("----------------")
    print(" ")
    print(r"....14.TEST FONCTION READ_TEXT_SLIDING_WINDOW AVEC KWARGS")
    gen_slwind1 = preproc.read_text_sliding_window(None, regex_clean, step=2, texte=text_basic)
    for gen in gen_slwind1:
        print(gen)
    tok_slwind1 = preproc.get_token_window(gen_slwind1, filter_by_length=2, joint=True)
    for tok in tok_slwind1:
        print(tok)
    print("----------------")
    print(" ")
    print(r"....15.TEST FONCTION GET_TOKEN AVEC WINDOW")
    gen_wind3 = preproc.read_text_window(None, regex_clean, texte=text_basic)
    tok_sent3 = preproc.get_token(gen_wind3)
    for tok in tok_sent3:
        print(tok)
    print("----------------")
    print(" ")
    print(r"....16.TEST FONCTION GET_TOKEN_WINDOW AVEC FULL_TEXT")
    tok_slwind2 = preproc.get_token_window(doc, filter_by_length=2, joint=False)
    for tok in tok_slwind2:
        print(tok)
    
       
        
    
            

