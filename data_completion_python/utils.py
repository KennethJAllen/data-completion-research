"""Utility functions."""
import numpy as np

def random_mask(m: int, n: int, k: int):
    """creates a random m x n mask with exactly k true entries"""
    perm = np.random.choice(m*n, k, replace=False)
    linear_mask = np.zeros(m*n)
    linear_mask[perm] = 1
    mask = np.reshape(linear_mask, (m, n))
    mask = mask==1 # turns mask into a logical matrix
    return mask
