function I = maxvol(X,I_initial)
%Algorithm based on paper How to Find a Good Submatrix
%written by K. Allen under Dr. Ming-Jun Lai's supervision

%finds a close to dominant rxr submatrix of nxr matrix X
%I_initial are the indices of the initial submatrix in X
%X(I_initial,:) is initial submatrix
%A = X(I,:) is the resulting close to dominant submatrix
%abs(det(A)) is close to maximum over all choices of submatrixes

epsilon = 1e-8; %tolerance
[n,r] = size(X);
[r_check1,r_check2] = size(I_initial);
if max(r_check1,r_check2) ~= r %checks that initial guess has the proper size
    error('The size of the initial guess does not match the width of the matrix')
end

I = I_initial; %index set of submatrix rows in X
A = X(I,:); %initial submatrix in X

if cond(A) > 1e12 %initial submatrix must be nonsingular
    error('Initial submatrix is close to singular')
end

for k=1:1000
    Y = abs(X/A);
    [y,linear_index] = max(Y(:));
    if y <= 1+epsilon %if det(A) has not changed by a factor of more than 1+epsilon
        break
    elseif k==1000
        error('maxvol did not converge in 1000 steps.')
    end
    
    [i,j] = ind2sub([n,r],linear_index); %converts linear index to matrix index
    I(j) = i; %replace jth row of A with ith row of X
    A = X(I,:);
end