function y = FSM(L,b)
%forwards substitution method
%input: nxn lower triangular matrix L, nx1 vector b
%output is x such that Lx=b
%n=5; L = tril(rand(n)); b = rand(n,1);

n = length(b);
y = zeros(n,1);
for i=1:n
    s = 0;
    for k=1:i-1
        s = s+y(k)*L(i,k);
    end
    y(i) = (b(i)-s)/L(i,i);
end
end