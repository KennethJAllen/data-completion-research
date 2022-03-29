import numpy as np
from numpy import linalg as LA

#in progress

def unfold(T, i):
    #returns the mode-i unfolding of T
    #i = 1, 2, or 3
    return np.reshape(np.moveaxis(T, i-1, 0), (T.shape[i-1], -1), order='F')

def refold(T, tensor_size, i):
    #refolds the mode-i unfolding of T back into T
    #i = 1, 2, or 3
    #refold((unfold(T,i),np.shape(T),i) = T
    return T

def Allen_Tensor_Completion(T0):
    #written by K. Allen under Dr. Ming-Jun Lai's supervision
    #from K. Allen's dissertation A Geometric Approach to Low-Rank Matrix and Tensor Completion

    #given an m x n x p partially known tensor T_Omega, where T_Omega has the block tensor structure:
    #T0[]:,:,0:r] = [[A, B],[C, unknowns]]
    #T0[:,:,r:] = [[D, unknowns],[unknowns, unknowns]]
    #completes T_Omega into a multilinear rank (r,r,r) tensor T
    #A = T[0:r,0:r,0:r] is an r x r x r multilinear rank (r,r,r) fully known subtensor
    #B = T[0:r,r:,0:r] is the corresponding r x (n-r) x r subtensor
    #C = T[r:,0:r,0:r] is the corresponding (m-r) x r x r subtensor
    #D = T[0:r,0:r,r:] is the corresponding r x r x (p-r) subtensor

    #assumes that the first r x r sub-matrix of every mode-i unfolding of A is invertible

    T = T0.copy()
    (m,n,p) = np.shape(T)
    A = T[0:r,0:r,0:r]
    return T

def forget_EFGH(T,r):
    #if T has the tensor block structure:
    #T[:,:,0:r] = [[A,B],[C,G]]
    #T[:,:,r:] = [[D,F],[E,H]]
    #replaces entries in positions E, F, G, and H with zeros
    T0 = T.copy()
    T0[r:,r:,0:r] = 0 #sets G to zero
    T0[r:,0:r,r:] = 0 #sets E to zero
    T0[0:r,r:,r:] = 0 #sets F to zero
    T0[r:,r:,r:] = 0 #sets H to zero
    return T0

#example
m = 3
n = 4
p = 5 #m x n x p tensor
r = 2 #rank of the tensor

T = np.zeros([m,n,p])
for i in range(r): #generates a random rank r order three tensor
    a = np.random.rand(m)
    b = np.random.rand(n)
    c = np.random.rand(p)
    A = np.tensordot(a,b,0)
    X = np.tensordot(A,c,0) #tensor product of a, b, and c
    T = T + X #T is the sum of r random rank one tensors, so has rank r with probability one

T0 = forget_EFGH(T,r)

