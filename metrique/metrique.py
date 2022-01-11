#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:36:32 2021

@author: Audrey Quessada Vial, Stephane Lamassé
Institutions: Pireh-Lamop
LICENCE GNU
This script aims to calculate metrics for cooccurrence matrix
"""

__version__ = "0.1.0"
__authors__ = "Audrey Quessada Vial","Stephane Lamassé"


import sys
sys.path.append("../")
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy import stats
import itertools
from scipy.stats import hypergeom
from scipy.special import comb
from mpmath import exp,log, log10
import decimal
from decimal import Decimal
import math
from scipy.spatial.distance import jaccard, dice
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from traitement.traitement import Preprocessing, read_stop_word, timer



def compare_transform(u, v, option="", **kwargs):
    """
    This function allows to get the common vocabulary between 2 pandas Series 
    and vectors that can be compared through the metrics

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    option : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    vocab_commun : TYPE
        DESCRIPTION.

    """
    #"u, v 2 pandas series"
    name1 = kwargs.get("name1", "")
    name2 = kwargs.get("name2", "")
    vocab_u = list(u.index)
    vocab_v = list(v.index)
    vocab_commun = list(set(vocab_u).intersection(set(vocab_v)))
    if option == "filtre":
        return u.loc[vocab_commun], v.loc[vocab_commun], vocab_commun
    elif option == "transform":
        df = pd.concat([u,v], axis=1).fillna(0)
        df.columns.values[0] = name1
        df.columns.values[1] = name2 
        print(df.shape, df)
        df[df != 0] = 1
       
        #df.mask(~df.index.isin(vocab_commun), 0, inplace=True )
        return df, vocab_commun
    else:
        df = pd.concat([u,v], axis=1).fillna(0)
        df.columns.values[0] = name1
        df.columns.values[1] = name2  
        return df, vocab_commun
                     

def calculate_dice_vector(u, v, empty_score=1.0):
    """
    This function calculates dice coefficient between 2 vectors

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    empty_score : TYPE, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    u1 = u.copy()
    v1 = v.copy()
    u1 = np.asarray(u1).astype(bool)
    v1 = np.asarray(v1).astype(bool)
    return dice(u1, v1)


def coeffdice(f,t,F):
       '''
       Ce coefficient doit permettre d'identifier des couples
       présentant un degré particulièrement élevé de cohésion
       lexicale. 
       Ce coefficient a été construit par Lee Raymond Dice et Thorval
       Sorensen 
       Cf, Smadja,  Frank;  McKeown,  Kathleen R.;  Hatzivassiloglou,
       Vasileios (1996).   Trans-lating collocations for bilingual
       lexicons:  A statistical
       approach.ComputationalLinguistics,22(1), 1-38 [https://link.springer.com/chapter/10.1007/11940098_6]
       '''
       coeffDice = (2 * f) / (t + F)
       return coeffDice
   

def ramanujan2(x):
    """
    This function calculates an approximation for fractional

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    fact : TYPE
        DESCRIPTION.

    """
    #decimal.getcontext().prec = 300
    #x = Decimal(x)
    fact = math.sqrt(math.pi)*(x/math.e)**x
    fact *= (((8*x + 4)*x + 1)*x + 1/30.)**(1./6.)
    if isinstance(x, int):
        fact = int(fact)
    return fact



def ca_coeff_spec(T, t, F, f, seuil, option="logfrac"):
       """
       Expliqué dans cet article : https://www.persee.fr/doc/mots_0243-6450_1980_num_1_1_1008
       """
       if f > t or F > T or T - F - t + f < 0:
              coeff = 0
              return coeff
       else:
              positif=1
              pp=hypergeometric(F, f, t, T, option=option)
              if pp > seuil:
                     coeff = 0
                     return coeff
              else :
                     """
                     Si la valeur de l'hypergeometrique est inférieure au   seuil introduit dans la fonction comme option 
                     """
                     p = pp
                     mode = int(((t+1)*(F+1))/(T+2)) # la valeur la plus probable d'apparation 
                     if f < mode :
                            '''
                            si la fréquence au coté du mot pôle est   inférieure à la valeur mode 
                            alors c'est un spécificité négative
                            '''
                            positif = 0
                            for i in range(f, mode):
                                '''
                                on balaie l'interval entre la   fréquence observée et la valeur attendue 
                                on calcul pour chaque moment la  probabilité d'apparition que l'on additione à la précédente 
                                puis on compare au seuil 
                                '''
                                p = (p * i * (T-F-t+i))/((F-i+1) * (t-i+1)) 
                                pp += p
                                if pp > seuil:
                                    coeff = 0
                                    return coeff
                                if p < 0.0000000001:
                                    break
                     else :
                            '''
                            spécificité positive
                            '''
                            for j in range(f,F):
                                   p = (p * (F-j) * (t-j))/((j+1) * (T-F-t+j+1))
                                   pp += p
                                   if pp > seuil:
                                          coeff = 0
                                          return coeff
                                   if p < 0.0000000001:
                                          break
              if pp <= 0:
                  coeff = 0
              else:
                  coeff = (log10(pp)-1)
                  # print("----")
                  if positif == 1:
                     coeff *= -1

       return coeff


def log_stirling(n):
    #decimal.getcontext().prec = 300
    fn = float(n)
    if fn < 0:
        fn = np.abs(fn)
    fn = n * math.log(fn)+ math.log(2 * math.pi * fn) / 2 - fn + \
    (fn ** -1) / 12 - (fn ** -3) / 360 + (fn ** -5) / 1260 - \
    (fn ** -7) / 1680 + (fn ** -9) / 1188
    return(fn)

   
    
def hypergeometric(F, f, t, T, option="logfrac"):
    '''
    T : le nombre d'occurrences dans le corpus
    t : le nombre d'occurrences dans les contextes du pôle
    F : la fréquence du cooccurrent dans le corpus
    f : la fréquence du cooccurrent dans les contextes du pôle
    '''
    if (T - F - t + f <= 0) or (T - F <= 0) or (T - t <= 0) :
        return 0
    else:
        if F - f <= 0 or t - f <= 0:
            return math.exp(log_stirling(T - F) + log_stirling(T - t) - log_stirling(T) - log_stirling(T - F - t + f))
        if option=="combinatory":
            return comb(F,f) * comb(T- F, t - f) / comb(T, t)
        elif option=="scipy":
            return hypergeom(T, F, t).pmf(f)
        elif option=="ramanujan":
            return ramanujan2(F) * ramanujan2(T - F) * ramanujan2(t) * ramanujan2(T-t) / (ramanujan2(f) * ramanujan2(F - f) * ramanujan2(t - f) * ramanujan2(T-F-t+f) * ramanujan2(T))
        elif option=="logfrac":           
            num = log_stirling(F) + log_stirling(T - F) + log_stirling(t) + log_stirling(T - t)
            denom = log_stirling(f) + log_stirling(F - f) + log_stirling(t - f) + log_stirling(T - F - t + f) + log_stirling(T)
            return math.exp(num - denom)


def calculate_cosine(u,v=None):
    """
    This function calculates the cosine similarity between 2 vectors

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if v:
        return cosine_similarity(u, v)
    else:
        return cosine_similarity(u)


def calculate_likelihood(u, v, **kwargs):
    """
    This function calculates the log_likelihood between 2 vectors

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    ll : TYPE
        DESCRIPTION.

    """
    weights = kwargs.get("weights", np.ones((u.size)).T)
    scores = np.dot(u, weights)
    ll = np.sum( v * scores - np.log(1 + np.exp(scores)) )
    return ll



def calculate_approximate_entropy(u, m=2, r=0.2):
    """
    This function implements a vectorized Approximate entropy algorithm.
   
    https://en.wikipedia.org/wiki/Approximate_entropy

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    m : TYPE, optional
        DESCRIPTION. The default is 2. Length of compared run of data
    r : TYPE, optional
        DESCRIPTION. The default is 0.2. Filtering level, must be positive

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    x = np.array(u)
    N = len(x)
    r *= np.std(x)
    if r < 0:
        raise ValueError("Parameter r must be positive.")
    if N <= m+1:
        return 0
   
    def _phi(m):
        x_re = np.array([x[i:i+m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x_re[:, np.newaxis] - x_re[np.newaxis, :]),
                              axis=2) <= r, axis=0) / (N-m+1)
        return np.sum(np.log(C)) / (N - m + 1.0)
   
    return np.abs(_phi(m) - _phi(m + 1))


def calculate_mutual_information(u, v, option="normalized"):
    """
    This function calculates the mutual information socre between 2 vectors

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if option=="normalized":
        return normalized_mutual_info_score(u, v)
    else:
        return mutual_info_score(u, v)


def calculate_jaccard(u, v):
    """
    This function calculates Jaccard similarity between 2 vectors

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    u1 = u.copy()
    v1 = v.copy()
    u1 = np.asarray(u1).astype(np.bool)
    v1 = np.asarray(v1).astype(np.bool)
    return jaccard(u1, v1)


def calculate_pearson(u, v):
    """
    This function calculates Pearson's correlation between 2 vectors'

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return pearsonr(u, v)

# @timer
# def calculate_metric_matrix(directed_matrix, metric="dice"):
#     list_columns = list(directed_matrix.columns)
#     metric_matrix = np.zeros((len(list_columns), len(list_columns)))
#     #df_metric = pd.DataFrame(metric_matrix, columns=list_columns, index=list_columns) 
#     for ((i,token1), (j,token2))in itertools.combinations(enumerate(list_columns),2):
#         u = directed_matrix[token1]
#         v = directed_matrix[token2]
#         if metric == "dice":
#             metric_matrix[i, j] = calculate_dice(u, v)
#         if metric=="cosine":
#             metric_matrix[i, j] = calculate_cosine(u, v)
#     return pd.DataFrame(metric_matrix, columns=list_columns, index=list_columns) 
        

if __name__=="__main__":
    pass
     # # generate an independent variable 
     # x = np.linspace(-10, 30, 100)
     # # generate a normally distributed residual
     # e = np.random.normal(10, 5, 100)
     # # generate ground truth
     # y = 10 + 4*x + e
     # #df = pd.DataFrame({'x':x, 'y':y})
     # print(calculate_likelihood(x, y))
     # path_input = "../Data/results/1489.csv"
     # df_dir = pd.read_csv(path_input)
     # df_dir = df_dir.drop(columns="Unnamed: 0")
     # print(df_dir.head())
     # df_metric = calculate_metric_matrix(df_dir, metric="dice")
     # print(df_metric)
     
