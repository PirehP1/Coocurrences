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
from scipy.stats import hypergeom
from scipy import stats
from scipy.spatial.distance import jaccard, dice
from scipy.stats import pearsonr, chi2
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score


def compare_transform(u, v, option="filtre", **kwargs):
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
        DESCRIPTION. The default is "filtre".

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
                     

def calculate_dice(u, v, empty_score=1.0):
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
    u1 = np.asarray(u1).astype(np.bool)
    v1 = np.asarray(v1).astype(np.bool)
    return dice(u1, v1)


def calculate_cosine(u,v):
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
    return 1.0 - np.dot(u, v) / (norm(u) * norm(v))


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

if __name__=="__main__":
     # generate an independent variable 
     x = np.linspace(-10, 30, 100)
     # generate a normally distributed residual
     e = np.random.normal(10, 5, 100)
     # generate ground truth
     y = 10 + 4*x + e
     #df = pd.DataFrame({'x':x, 'y':y})
     print(calculate_likelihood(x, y))
     
     
     