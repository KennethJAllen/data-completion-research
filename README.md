# Data Completion Research

This repository consists of my graduate research on data completion via low-rank matrix and tensor completion, and maximum volume algorithms for finding dominant submatrices of matrices. All algorithms are implemented in MATLAB and/or Python.

## 🔧 Installation


### Clone the Repository:

```
git clone https://github.com/KennethJAllen/ygo-small-world
cd ygo-small-world
```
### Python

#### Set Up Environment

*   Install Poetry if not already installed.
*   Run the following command in the project directory:

```
poetry install
```
#### Activate the Virtual Environment
```
poetry shell
```
You can now run the project's scripts within the poetry shell.

## 🔢 Low-Rank Matrix Completion

Low-Rank matrix completion is the task of completing missing entries of a partially complete matrix subject to the constraint that the rank of the resulting matrix is minimized. This is a non-convex minimization problem, and as such is considered difficult to solve.

### Algorithms
1. alternating projection
   * `data_completion_python\matrix_completion.py`
   * `data_completion_matlab\alternating_projection_images.m`
3. alternating projection with maximum volume skeleton decomposition (MVSD)
   * `data_completion_matlab\alternating_projection_mvsd.m`
5. schur maximum volume gradient descent
   * `data_completion_matlab\schur_maxvol_gradient_descent.m`

## 🧊 Low-Rank Tensor Completion

Like Low-Rank matrix completion, Low-Rank tensor completion is the task of completing the missing entries of a partially complete tensor subject to the constraint that the rank of the resulting tensor is minimized. Also like low-rank matrix completion, it is also a non-convex minimization problem.

Unlike matrices, there are multiple distinct definitions for the rank of a tensor. In this repository, one tensor completion algorithm is presented which completes partially complete tensors with a particular structure of known entries subject to the constraint that the multilinear rank is minimized.

Python and MATLAB tensor completion via generalized CUR decomposition functions can be found in `data_completion_python\tensor_completion.py` and `data_completion_matlab\tensor_completion_demo.m`.

## 🔼 Maximum Volume

Maximum volume algorithms find $r \times r$ dominant submatrices of matrices. They are used for creating low-rank approximations using the skeleton decomposition. While the singular value decomposition is the gold standard for finding low-rank approximations, the computational complexity is $O(nm^2)$ for an $n \times m$ matrix with $m$ at most $n$. In comparison, the maximum volume skeleton decomposition is significantly faster, while often only increasing the error by a negligible amount.

### Algorithms
##### for $m \times r$ matrices
1. maxvol
   * `data_completion_python\maxvol.py`
   * `data_completion_matlab\functions\maxvol-algorithms\greedy_maxvol.m`
3. greedy maxvol
   * `data_completion_matlab\functions\maxvol-algorithms\greedy_maxvol.m`

##### for $m \times n$ matrices
3. alternating maxvol
   * `data_completion_python\maxvol.py`
   * `data_completion_matlab\functions\maxvol-algorithms\alternating_maxvol.m`
5. algernating greedy maxvol
   * `data_completion_matlab\functions\maxvol-algorithms\alternating_greedy_maxvol.m`

## 🖼️ Examples
A demo notebook for python scripts can be found in `notebooks/demo.ipynb`.

### Image Completion

Consider the following $128 \times 128$ image of a penny:

![penny](images/input/penny.jpg)

Suppose 25% of the pixels are deleted at random:

![penny missing](images/output/three_fourths_partial_penny.jpg)

We would like to recover the missing entries. Using the alternating projection algorithm, we recover the following image:

![penny recovered with alternating projection](images/output/alternating_projection_recovered_penny_rank18.jpg)

Using the Schur maximum volume gradient descent method, we recover the following image:

![penny recovered with alternating projection](images/output/maxvol_grad_descent_recovered_penny_rank_18.jpg)

### Maximum Volume CUR Decomposition

Maximum volume algorithms find close to maximum volume submatrices of a matrix. Here the volume of a matrix is the absolute value of the determinant.

Maximum volume submatrices are important to find because they minimize the error of the CUR decomposition of the matrix if they matrix is high rank.

Conisder the following $512 x 512 image of a house:

![house](images/input/house.png)

We take the CUR decomposition with respect to rows and columns chosen by the alternating maximum volume algorithm.

For comparison, we also show the results of the CUR decomposition with respect to arbitrarily chosen columns and rows.

The quality of the decomposition is measured by the peak signal to nosie ratio (PSNR). The PSNR of the CUR decomposition is much better when taken with respect to a maximum volume submatrix.

![house_cur](images/output/demo/cur_img_compression.png)
