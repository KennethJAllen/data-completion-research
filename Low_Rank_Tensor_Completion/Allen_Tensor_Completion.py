import numpy as np
from numpy import linalg as LA

#in progress

def Allen_Tensor_Completion(T_Omega):
    #written by K. Allen under Dr. Ming-Jun Lai's supervision
    #from K. Allen's dissertation A Geometric Approach to Low-Rank Matrix and Tensor Completion

    #given an m x n x p partially known tensor T_Omega, where T_Omega has the structure:
    #T_Omega[:,:,0:r] = [A B;C unknowns]
    #T_Omega[:,:,(r+1):] = [D unknowns;unknowns unknowns]
    #completes T_Omega into a multilinear rank (r,r,r) tensor T
    #A = T[0:r,0:r,0:r] is an r x r x r multilinear rank (r,r,r) fully known subtensor
    #B is corresponding r x (n-r) x r subtensor
    #C is corresponding (m-r) x r x r subtensor
    #D is corresponding r x r x (p-r) subtensor

    #assumes that the first r x r sub-matrix of every mode-i unfolding of A is invertible

    T = T_Omega.copy()

    return T

#example
m = 18
n = 19
p = 20 #m x n x p tensor