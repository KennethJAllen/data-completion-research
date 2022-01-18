function [Z,sigma] = AlternatingProjection(Omega,r, M0)
%r = rank
%Omega = unknown entries
%M0 initial guess

N = 10000; %number of itterations
sigma = zeros(1,N); %(r+1)st singular value on step k

known_entries = M0(Omega);
Z=M0;
    %A=OR1MC(M,mOmega,r);
    for k = 1:N
        [U, S, V] = svd(Z);
        Z = U(:,1:r)*S(1:r,1:r)*V(:,1:r)'; %rank r projection
        S = diag(S);
        sigma(k) = S(r+1);
        Z(Omega) = known_entries; %projection onto A_Omega
    end
end