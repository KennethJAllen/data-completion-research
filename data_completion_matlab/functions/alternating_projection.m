function X = alternating_projection(Omega, r, X0)

%alternating projection matrix completion algorithm
%based on work by Lai, Varghese

%inputs:
%Omega = logical mask of unknown entries
%r = rank of completion
%M0 = initial completion guess

%outputs:
%X = rank r completion approximation if one exists

N = 500; %number of itterations
known_entries = X0(Omega);
X=X0;
for k = 1:N
    [U, S, V] = svd(X);
    X = U(:,1:r)*S(1:r,1:r)*V(:,1:r)'; %rank r projection
    X(Omega) = known_entries; %projection onto hyperplane of completions
end
end
