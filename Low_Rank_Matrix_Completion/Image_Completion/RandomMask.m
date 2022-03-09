function Omega = RandomMask(m,n,k)
%creates a random mxn logical matrix (mask) Omega with exactly k ones
entries = randperm(m*n,k);
Omega_linear = zeros(m*n,1);
Omega_linear(entries) = 1;
Omega = reshape(Omega_linear,m,n);
Omega = Omega==1;
end