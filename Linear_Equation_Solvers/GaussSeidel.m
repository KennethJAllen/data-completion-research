function x = GaussSeidel(A,b,x0)
%approximate the solution to Ax=b with Gauss-Seidel itteration, and initial
%guess x0
%gaurenteed convergence if A is strictly diagonally domanant
%gaurenteed convergence if A is symmetric positive definite
%n=5; B=rand(n); A=B*B'; b=rand(n,1); x0=zeros(n,1);

D = diag(diag(A)); %diag(A) returns a column vector, diag(v) returns a matrix
U = triu(A)-D; %upper triangular matrix of A with zeros on diagonal
L = tril(A)-D; %lower triangular matrix of A with zeros on diagonal
x = x0;

for i = 1:500
    x = (D+L)\(b-U*x);
end
end