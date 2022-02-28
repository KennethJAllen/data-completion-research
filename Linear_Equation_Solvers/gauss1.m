function U = gauss1(A)
%Gaussian ellimination without pivoting
%Assume a_ii not zero at each step.

%choose c such that c*a11 + a21 = 0
[n,m] = size(A);
for j=1:m %indexing on the rows
    for i = j+1:n %indexing on the columns
        c = -A(i,j)/A(j,j);
        A(i,:) = c*A(j,:)+A(i,:);
    end
end
U = A;
end