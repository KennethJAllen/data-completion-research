function x = BSM(U,b)
%backwards substitution method
%input: nxn upper triangular matrix U, nx1 vector b
%output: x such that Ux=b
%n=5; U = triu(rand(n)); b = rand(n,1);

n = length(b);
x = zeros(n,1);
for i=0:n-1
    j = n-i;
    s = 0;
    for k = j+1:n
        s = s+U(j,k)*x(k);
    end
    x(j) = (b(j)-s)/U(j,j);
end
end