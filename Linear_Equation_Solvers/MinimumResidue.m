function x = MinimumResidue(A,b,x0)
%Minimal Residual Iteration
%solve Ax=b itteratively with initial guess x0
%converges if the smallest eigenvalue of (A+A')/2 is larger than 0
%n=5; B=rand(n); A=B*B'; b=rand(n,1); x0=zeros(n,1);

x = x0;
for i = 1:5000
    r = b-A*x;
    if norm(r,inf) < 1e-8
        break
    end
    alpha = dot(A*r,r)/dot(A*r,A*r);
    x = x+alpha*r;
end
end