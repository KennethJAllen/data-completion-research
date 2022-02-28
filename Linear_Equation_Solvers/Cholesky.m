function C = Cholesky(A)
%For A symmetric and positive definite, Cholesky decomposition.
%A = C'*C, C upper triangular
%n=5; B = rand(n); A=B*B';

[n,~] = size(A);
C = zeros(n,n);

for i = 1:n
    C(i,i:n) = A(1,:)/sqrt(A(1,1)); %calculates the top entries of C
    A = A(2:n+1-i,2:n+1-i) - C(i,i+1:n)'*C(i,i+1:n); %reduces the size of A
end
end