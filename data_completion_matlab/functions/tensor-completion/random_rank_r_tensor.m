function T_rand = random_rank_r_tensor(m,n,p,r)
%generate a random m x n x p rank r tensor
T_rand = zeros(m,n,p);
for i=1:r
    a = rand(m,1);
    b = rand(n,1);
    c = rand(p,1);
    X = tensor_product(a,b,c);
    T_rand = T_rand + X;
end
end