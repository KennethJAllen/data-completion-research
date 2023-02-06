function X = unfold_tensor(T, i)
%returns the mode-i unfolding of T
%i = 1, 2, or 3
dim = size(T);
X = reshape(shiftdim(T,i-1), dim(i), []);
end