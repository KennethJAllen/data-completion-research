function [X,sigma] = Alternating_Projection(Omega,r, M0)
%This is the matrix completion algorithm based on work by Lai, Varghese

%Omega = mask of unknown entries
%r = rank of completion
%M0 initial completion guess
%Z = completion
%sigma = singular values of completion

N = 10000; %number of itterations
sigma = zeros(1,N); %(r+1)st singular value on step k

known_entries = M0(Omega);
X=M0;
for k = 1:N
    [U, S, V] = svd(X);
    X = U(:,1:r)*S(1:r,1:r)*V(:,1:r)'; %rank r projection
    S = diag(S);
    sigma(k) = S(r+1);
    X(Omega) = known_entries; %projection onto A_Omega
end
end