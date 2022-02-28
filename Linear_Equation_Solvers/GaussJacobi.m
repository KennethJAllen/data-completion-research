function x = GaussJacobi(A,b,x0)
%approximate the solution to Ax=b with Gauss-Jacobi itteration, and initial
%initial guess x0
%gaurenteed convergence if A is strictly diagonally domanant
%A = [7 -1 3 2;1 4 0 -2;-1 1 -3 0;-3 -2 3 -9]; b=rand(4,1); x0=zeros(4,1);

D = diag(A); %diag(A) returns a column vector, diag(v) returns a matrix
R = 1./D; %inverts each element of D
D = diag(D);
invD = diag(R);
U = triu(A)-D; %upper triangular matrix of A with zeros on diagonal
L = tril(A)-D; %lower triangular matrix of A with zeros on diagonal
x = x0;

for i = 1:500
    x = invD*(b-L*x-U*x);
end
end