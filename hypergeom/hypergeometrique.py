#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 19:03:14 2021

@author: Audrey Quessada Vial, Stephane Lamassé
Institutions: Pireh-Lamop
LICENCE GNU
This script aims to calculate hypergeometric distribution density
"""

__version__ = "0.1.0"
__authors__ = "Audrey Quessada Vial","Stephane Lamassé"

import sys
sys.path.append("../")
from scipy.special import comb
import numpy as np
from collections import Counter
from traitement.traitement import Preprocessing, read_stop_word, timer
from matrix_gen.matrix_generator import MatrixGenerator as MG



def calc_hdd(text, sample_size):
    """
    Calcul de la métrique de la distribution hypergéométrique D (HD-D)
    La description:
        Implémentation la plus fiable de l'algorithme VocD (2010, McCarthy & Jarvis)
        L'algorithme est basé sur la méthode de sélection aléatoire à partir du texte de segments de 32 à 50 mots et
        calculer le TTR pour eux avec une moyenne ultérieure
    Аrguments:
        text (list[str]): liste de mots
        sample_size (int): Longueur du segment
    Output:
        float: valeur de la métrique
    """

    def hyper(successes, sample_size, population_size, freq):
        """
        La probabilité qu'un mot apparaisse dans au moins un segment, dont chacun
        généré sur la base d'une distribution hypergéométrique
        """
        try:
            prob = 1.0 - (
                float(
                    (
                        comb(freq, successes)
                        * comb((population_size - freq), (sample_size - successes))
                    )
                )
                / float(comb(population_size, sample_size))
            )
            prob = prob * (1 / sample_size)
        except ZeroDivisionError:
            prob = 0
        return prob

    n_words = len(text)
    if n_words < 50:
        return -1
    hdd = 0.0
    lexemes = list(set(text))
    freqs = Counter(text)
    for lexeme in lexemes:
        prob = hyper(0, sample_size, n_words, freqs[lexeme])
        hdd += prob
    return hdd


if __name__=="__main__":
    path_text = "../Data/corpusbytime/1489.txt"
    path_stop_words = "../Data/stop_words.txt"
    stop_words = read_stop_word(path_stop_words)
    preproc = Preprocessing(input_path="",
                            window=40,
                            stop_words=stop_words,
                            remove_punctuation=True,
                            filter_by_length=2,
                            separator=" ",
                            case=True)
    regex_clean = "(\')?(\d*)(\\n)"
    text = preproc.read_text_full(path_text, regex_clean)
    gen_token = list(preproc.get_token(text, filter_by_length=2))
    hdd = calc_hdd(gen_token, 40)
    print(hdd)
    