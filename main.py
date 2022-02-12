#!/usr/bin/env python3


"""
Created on Thu Jul 15 10:40:20 2021
@author: Audrey Quessada Vial, Stephane Lamassé
Institutions: Pireh-Lamop
LICENCE GNU
This script aims to generate cooccurrence matrix
"""

__version__ = "0.1.0"
__authors__ = "Audrey Quessada Vial","Stephane Lamassé"


from traitement.traitement import Preprocessing, read_stop_word
from metrique.metrique import *
from matrix.matrix_generator import *



if __name__ == "__main__":
    """ATTENTION ICI MODIFIER LE PATH VERS LES FICHIERS TEXTS"""
    dir_out = "Results/"
    input_path = "Data/"#"../Data/corpusbyTime"
    save_path = "Results"

    """ATTENTION DONNER LE PATH VERS LES STOP-WORDS"""
    path_stop_words = "traitement/stop_words.txt"
    stop_words = read_stop_word(path_stop_words)

    """ATTENTION SELECTIONNER L'ENCODING POUR LIRE LE TEXTE"""
    encoding = "ISO-8859-1"#args.encoding       
    # création d'un dictionnaire pour répertorier le temps d'exécution
    dict_time  = {}
    #création d'un dictionnaire qui répertorie tous les mots de tous les textes
    dict_all_vocab = {}
    #définition des attributs de la classe MatrixGenerator
    window = 30#args.window
    remove_punctuation = True
    filter_by_length = 2
    separator = " "
    case = True

    #initialization matrix_generator 
    matrix_gen = MatrixGenerator(input_path=input_path,
                            window=window,
                            stop_words=stop_words)
    #getting list of paths to process
    list_file = matrix_gen.get_files_simple(ext="txt")
    # regex definition
    regex_clean = "(\')?(\d*)?(\\n)[']"


    # Exemple sur une centaine de fichiers
    list_matrix = []
    i = 0
    n = len(list_file) -1  # nombre de textes lus 
    while i <= n:
        file = list_file[i]
        if file.endswith(".txt"):
            filename = file.split("/")[-1].split(".")[0]
        else:
            with open(file, "r", encoding=encoding) as f:
                f.seek(7)
                filename = f.read(4)
#        print("-----------1.LECTURE DU TEXTE FULL + BASIC CLEANING ----------------")
        text_full_file = matrix_gen.read_text_full(file, regex_clean, encoding=encoding, start=12)
        key = "lecture et nettoyage " + filename
        list_words = list(matrix_gen.get_token(text_full_file, 
                                         filter_by_length=filter_by_length,
                                         separator=separator, 
                                         stop_words=stop_words))
        key = "récupération des token " + filename
        if len(list_words) <= window:
            pass
        else:
 #           print("-------------2.VERIFICATION GET_VOCAB FULL TEXT NON FILTRE----------------")
 
            vocab_file = matrix_gen.get_vocab(text_full_file[0],
                                      filter_by_length=filter_by_length,
                                      stop_words=stop_words)
            key = "creation vocabulaire " + filename
            for key, value in vocab_file.items():
                if key not in dict_all_vocab.keys():
                    dict_all_vocab[key] = value
                else:
                    dict_all_vocab[key] += value
            '''
            On produit le dictionnaire avec toutes les occurrences, méthode : counter_full
            ''' 
            counter_full_file = matrix_gen.counter_full(text_full_file[0], vocab_file)
#            print("-------------3.VERIFICATION GET_DIRECTED_MATRIX WINDOW----------------")
            '''
            compter l'occurrence des mots par fenêtre, méthode : counter_by_windows
            On utilise un générateur 
            vocab_file tous les mots auquels on donne un index 
            '''
            gen_count_wnd_file = matrix_gen.counter_by_window(text_full_file[0], 
                                                      vocab_file, 
                                                      window=window)
            '''
            on génère la matrice dirigé à partir du générateur 
            '''
            mat_dir_wnd_file = matrix_gen.get_directed(gen_count_wnd_file, 
                                                vocab_file, 
                                                option="")
            key = "matrice dirigée par fenêtre " + filename
            

 #           print("-------------4.VERIFICATION SPECIFICITE GET_DIRECTED_MATRIX WINDOW----------------")
            ''' 
            il faut le reprendre pour calculer les specificité 
            '''
            gen_count_wnd_file = matrix_gen.counter_by_window(text_full_file[0], 
                                                      vocab_file, 
                                                      window=window)
            '''
            spec = utilise la fonction ca_coeff spec 
            specificité 
            '''
            spec_wnd = matrix_gen.specificite(gen_count_wnd_file, 
                                            vocab_file, 
                                            mat_dir_wnd_file, 
                                            counter_full_file,
                                            option="spec")
            key = "spécificité matrice dirigée par phrase " + filename
            list_matrix.append(spec_wnd)
            nme = dir_out  + filename + "spec.csv"
            spec_wnd.to_csv(nme, sep=";")
        i += 1

