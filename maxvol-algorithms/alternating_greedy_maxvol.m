function [I,J] = alternating_greedy_maxvol(X,I_initial,J_initial)
%written by K. Allen under Dr. Ming-Jun Lai's supervision
%from K. Allen's dissertation A Geometric Approach to Low-Rank Matrix and Tensor Completion
%alternating version of the one directional greedy maxvol algorithm

%finds close to dominant rxr submatrix of an mxn matrix
%swaps up to r rows each iteration
%I_initial are the row indices of the initial submatrix in X
%J_initial are the column indices of the initial submatrix in X
%X(I_initial,J_initial) is the initial submatrix
%A = X(I,J) is the resulting close to dominant submatrix
%abs(det(A)) is close to maximum over all choices of submatrices

%gives indices of an rxr submatrix of mxn with maximum determinant in modulus
%num_iter = number of backslash operations to converge

epsilon = 1e-8; %error tolerance
[m,n] = size(X);

I = I_initial; %index set of sub-matrix rows
J = J_initial; %index set of sub-matrix columns
A = X(I,J); %initial sub-matrix

%if abs(det(A)) < 1e-12
%    error('Initial A is not invertible')
%end

r = size(I);
r = max(r);
r2 = size(J);
r2 = max(r2);
if r~=r2
    error('initial submatrix is not rxr.')
end

row_dom = 0; %indicates if near dominant in rows
%num_iter = 0; %number of iterations to converge

i = zeros(r,1); %row indices
j = zeros(r,1); %column indices

for k=1:1000
    Y = X(:,J)/A;
    %num_iter = num_iter+1;
    Y_abs = abs(Y);
    
    [y,linear_index_I] = max(Y(:)); %y = largest entry
    if y > 1+epsilon
        [i(1),j(1)] = ind2sub([m,r],linear_index_I); %index of y
        I(j(1)) = i(1); %replace jth row with ith row
        Y_abs(:,j(1)) = 0; %makes sure next largest value is not in the same column
        Y_abs(i(1),:) = 0; %makes sure next largest value is not in the same row
        for s=2:r
            [~,linear_index_I] = max(Y_abs(:)); %searches for largest index not in the original column
            [i(s),j(s)] = ind2sub([m,r],linear_index_I); %[i,j] = index
        
            V = Y(i(1:(s-1)),j(1:(s-1)));
            B = Y(i(1:(s-1)),j(s));
            C = Y(i(s),j(1:(s-1)));
            y = Y(i(s),j(s));
            if  abs(y-C*(V\B)) > 1 %makes sure volume is increasing
                I(j(s)) = i(s); %replaces jth row with ith row
                Y_abs(:,j(s)) = 0;
                Y_abs(i(s),:) = 0;
            else
                break
            end
        end
        A = X(I,J);
        column_dom = 0; %indicates not near dominant in columns
    elseif row_dom == 1
        break
    else
        column_dom = 1; %indicates near dominant in columns
    end
    
    Z = A\X(I,:);
    %num_iter = num_iter+1;
    Z_abs = abs(Z);
    
    [z,linear_index_J] = max(Z(:)); %z = largest entry
    if z > 1+epsilon
        [i(1),j(1)] = ind2sub([r,n],linear_index_J); %index of y
        J(i(1)) = j(1); %replace jth row with ith row
        Z_abs(:,j(1)) = 0; %makes sure next largest value is not in the same column
        Z_abs(i(1),:) = 0; %makes sure next largest value is not in the same row
        for s=2:r
            [~,linear_index_J] = max(Z_abs(:)); %searches for largest index not in the original column
            [i(s),j(s)] = ind2sub([r,n],linear_index_J); %[i,j] = index
        
            V = Z(i(1:(s-1)),j(1:(s-1)));
            B = Z(i(1:(s-1)),j(s));
            C = Z(i(s),j(1:(s-1)));
            z = Z(i(s),j(s));
            if  abs(z-C*(V\B)) > 1 %makes sure volume is increasing
                J(i(s)) = j(s); %replaces jth row with ith row
                Z_abs(:,j(s)) = 0;
                Z_abs(i(s),:) = 0;
            else
                break
            end
        end
        A = X(I,J);
        row_dom = 0; %indicates not near dominant in rows
    elseif column_dom == 1
        break
    else
        row_dom = 1; %indicates near dominant in rows
    end
    
    if k == 1000
        error('Algorithm did not converge in 1000 steps.')
    end
end