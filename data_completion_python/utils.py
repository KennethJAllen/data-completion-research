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

def forget_compliment(X: np.ndarray, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Inputs: Matrix X, row indices I, column indices J."""
    X_copy = X.copy()
    mask = np.zeros_like(X, dtype=bool)
    mask[I, :] = True # Set the mask to True for rows in I
    mask[:, J] = True # Set the mask to True for columns in J
    X_copy[~mask] = 0
    return X_copy

def psnr(img1: np.ndarray, img2: np.ndarray, max_pixel_value: float = 255.0):
    """Peak signal to nosie ratio."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Infinite PSNR if MSE is zero
    return 10 * np.log10((max_pixel_value ** 2) / mse)
