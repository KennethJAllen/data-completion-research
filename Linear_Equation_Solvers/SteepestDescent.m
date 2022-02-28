function x = SteepestDescent(A,b,x0)
%Steepest Descent Iteration
%solve Ax=b itteratively with initial guess x0
%converges A is symmetric possitive definite
%n=5; B=rand(n); A=B*B'; b=rand(n,1); x0=zeros(n,1);

x = x0;
for i = 1:500
    r = b-A*x;
    if norm(r,inf) < 1e-8
        break
    end
    t = dot(r,r)/dot(A*r,r);
    x = x+t*r;
    norm(A*x-b)
end
end