function [Q,R] = myqr(A)
%QR decomposition of A
%A = Q*R
%Q orthogonal
%R = [R1; 0], R1 is upper triangular

[n,m] = size(A); %n > m
Q = eye(n); %start with identity matrix
R = zeros(n,m); %start with zero matrix
I = eye(n); %get standard basis vectors
%solve Q'*A=R using Hausholder transform

for i=1:m
    s = sign(A(i,i));
    a = A(i:n,i); %the ith through nth entries of the ith column
    e = I(i:n,i); %= [1;0;...;0] %n-i zeros
    if s < 0
        v = a-norm(a)*e;
    else
        v = a+norm(a)*e;
    end
    H = I;
    H(i:n,i:n) = Hausholder(v);
    Q = H*Q;
    A = H*A;
    for j=i:m
        R(i,j) = A(i,j);
    end
end
Q = Q';
end

function H = Hausholder(v)
%Hauseholder transform
[n,~] = size(v);
I = eye(n); %identity matrix
alpha = 2/(v'*v);
H = I-alpha*(v*v');
end