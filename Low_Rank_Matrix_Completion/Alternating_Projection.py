import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt 

#def alternating_projection(Omega,r,M0):
#This is the matrix completion algorithm based on work by Lai, Varghese

def random_mask(m,n,k):
    #creates a random mxn mask with exactly k true entries
    perm = np.random.choice(m*n, k, replace=False)
    Omega_linear = np.zeros(m*n)
    Omega_linear[perm] = 1
    Omega = np.reshape(Omega_linear,(m,n))
    Omega = Omega==1 #turns Omega into a logical matrix
    return Omega

m = 4
n = 3 #matrix size mxn
r = 2; #rank of completion
#A = np.random.rand(m,r)
#B = np.random.rand(r,n) #random factor matrices
A = np.array([[1, 0], [0, 1], [1, 1], [1, 2]])
B = np.array([[1, 1, 2],[0, 1, 1]])
M = A @ B #random rank r matrix

known_ratio = 0.75 #ratio of known to unknown entries
num_known = round(known_ratio*m*n) #number of known entries
#Omega = random_mask(m,n,num_known) #mask of unknown entries
Omega = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 0],[1, 0, 0]],dtype=bool) #mask
known_entries = M[Omega] #records the known entries of M

N = 300 #number of itterations
singular_values = np.zeros(N) #singular value at ech step
X = np.copy(M) #X initial completion guess
X[~Omega] = 0 #sets unknown entries equalo to zero

for k in range(N):
    U, S, Vh = LA.svd(X) #singular value decompostion of X
    Sigma = np.zeros((m,n))
    np.fill_diagonal(Sigma,S) #diagonal matrix of singular values
    X = U[:,0:r] @ Sigma[0:r,0:r] @ Vh[0:r,:] #rank r projection
    singular_values[k] = S[r] #records (r+1)st singular value of X
    X[Omega] = M[Omega] #known_entries #projection onto plane of completions

x = np.arange(0,N)
plt.title("Error at nth step")
plt.xlabel("nth step")
plt.ylabel("error")
plt.plot(x,singular_values) #error plot
plt.show()