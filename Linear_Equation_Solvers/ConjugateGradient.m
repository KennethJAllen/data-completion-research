function x = ConjugateGradient(x0,A,b,epsilon)
%conjugate gradient descent algorithm for A nxn symmetric, positive definite
%solves Ax=b for initial guess x0 with tolerance epsilon
%gaurenteed convergence in n steps
%n=5; B=rand(n); A=B*B'; b=rand(n,1); x0=zeros(n,1); epsilon=1e-10;

[n,~] = size(A);
r = b-A*x0; %initialization r0
v = r; %initialization v0
x = x0; %initialization x0

if norm(r,inf) < epsilon
    return
end

for k=1:n+1
    t = r'*r/(v'*A*v);
    x = x+t*v;
    q = r; %remember r(k)
    r = r-t*A*v;
    if norm(r,inf) < epsilon
        return
    end
    s = r'*r/(q'*q); %r(k+1)'*r(k+1)/r(k)'*r(k)
    v = r+s*v;
end