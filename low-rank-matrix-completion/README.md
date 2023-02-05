# low-rank-matrix-completion

Given a matrix with missing entries, denoted $M_\Omega$, the task of low-rank matrix completion is to find a rank $r$ matrix $M$ such that the known entries of $M_\Omega$ are equal to the corresponding entries in $M$.

## alternating projection
Based on work by Lai, Varghese. The alternating projection algorithm alternates between projecting onto the manifold of rank r matrices and projecting onto the manifold of possible matrix completions.

## alternating projection with MVSD
The alternating projection with MVSD is simnilar to the alternating projection algorithm. Instead of a rank r projection calculated with the expensive SVD, an apporximation is calculated with the efficient MVSD.

## Schur maximum volume gradient descent
The Schur maximum volume gradient descent algorithm is a gradient descent algorithm based on a combination of the maximum volume algorithm and the Schur complement.
