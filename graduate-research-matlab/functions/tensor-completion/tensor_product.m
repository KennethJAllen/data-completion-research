function X = tensor_product(a,b,c)
%input: length m vector a, length n vector b, length p vector c
%a, b, c column vectors
%outputs tensor product a \otimes b \otimes c
%X is an m x n x p tensor
m = numel(a);
n = numel(b);
p = numel(c);
X = zeros(m,n,p);
A = a*b';
for i=1:p
    X(:,:,i) = A*c(i);
end
end