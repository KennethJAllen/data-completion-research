function X = fold_tensor(T,tensor_size,i)
%folds the mode-i unfolding of T back into T
%i = 1, 2, or 3
%fold((unfold(T,i),size(T),i) = T
X = reshape(T, circshift(tensor_size,-i+1));
X = shiftdim(X,-i+4);
end