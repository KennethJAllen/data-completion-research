function [I,J] = alt_maxvol(X,I_initial,J_initial)
%Algorithm based on paper How to Find a Good Submatrix modified for two-directional search
%written by K. Allen under Dr. Ming-Jun Lai's supervision
%alternating version of the one directional maxvol algorithm

%finds close to dominant rxr submatrix of an mxn matrix
%I_initial are the row indices of the initial submatrix in X
%J_initial are the column indices of the initial submatrix in X
%X(I_initial,J_initial) is the initial submatrix
%A = X(I,J) is the resulting close to dominant submatrix
%abs(det(A)) is close to maximum over all choices of submatrices

epsilon = 1e-8; %tolerance
[m,n] = size(X);

I = I_initial; %index set of submatrix rows in X
J = J_initial; %index set of submatrix columns in X
A = X(I,J); %initial submatrix in X

r = size(I); %submatrix A should be rxr
r = max(r);
r2 = size(J);
r2 = max(r2);
if r~=r2 %checks that initial index sizes are the same r=r2
    error('initial submatrix is not rxr.')
end

row_dom = 0; %indicates if near dominant in rows

if cond(A) > 1e12 %initial submatrix must be nonsingular
    error('Initial submatrix is close to singular')
end

for k=1:1000
    Y = abs(X(:,J)/A);
    [y,linear_index_I] = max(Y(:)); %y is max in columns
    if y > 1+epsilon %if the volume change by swapping one column
        [i,j] = ind2sub([m r],linear_index_I); %[i,j] = index
        I(j) = i; %replace jth row of A with ith row of X(:,J)
        A = X(I,J);
        column_dom = 0; %indicates not near dominant in columns
    elseif row_dom == 1
        break %if near dominant in rows and columns
    else
        column_dom = 1; %indicates near dominant in columns
    end
    
    Z = abs(A\X(I,:));
    [z,linear_index_J] = max(Z(:)); %z is max in rows
    if z > 1+epsilon
        [p,q] = ind2sub([r n],linear_index_J); %[i,j] = index
        J(p) = q; %replace pth column of A with qth row of X(I,:)
        A = X(I,J);
        row_dom = 0; %indicates not near dominant in rows
    elseif column_dom == 1
        break %if near dominant in rows and columns
    else
        row_dom = 1; %indicates near dominant in rows
    end
    
    if k==1000
        error('alt_maxvol did not converge in 1000 steps')
    end
end