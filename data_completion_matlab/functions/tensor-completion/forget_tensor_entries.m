function X = forget_tensor_entries(T,r)
%replaces entries of T in unknown positions with zeros
%input: T(:,:,1:r) = [A B;C G], T(:,:,r+1:p) = [D F;E H]
%output: X(:,:,1:r) = [A B;C 0], X(:,:,r+1:p) = [D 0;0 0]
[m,n,p] = size(T);
X = T;
X(r+1:m,r+1:n,1:r) = zeros(m-r,n-r,r); %sets G to zero
X(r+1:m,1:r,r+1:p) = zeros(m-r,r,p-r); %sets E to zero
X(1:r,r+1:n,r+1:p) = zeros(r,n-r,p-r); %sets F to zero
X(r+1:m,r+1:n,r+1:p) = zeros(m-r,n-r,p-r); %sets H to zero
end