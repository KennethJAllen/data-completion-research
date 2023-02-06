%written by K. Allen under Dr. Ming-Jun Lai's supervision
%from K. Allen's dissertation A Geometric Approach to Low-Rank Matrix and Tensor Completion

%given an m x n x p partially known tensor T_Omega, where T_Omega has the block tensor structure:
%T_Omega(:,:,1:r) = [A B;C unknowns]
%T_Omega(:,:,(r+1):end) = [D unknowns;unknowns unknowns]
%completes T_Omega into a multilinear rank (r,r,r) tensor T
%if a multilinear rank (r,r,r) completion exists, it is unique

%A = T(1:r,1:r,1:r) is an r x r x r multilinear rank (r,r,r) fully known subtensor
%B is corresponding r x (n-r) x r subtensor
%C is corresponding (m-r) x r x r subtensor
%D is corresponding r x r x (p-r) subtensor

%assumes that the first r x r sub-matrix of every mode-i unfolding of A is invertible

addpath(genpath('functions'))

%m x n x p tensor
m = 20;
n = 19;
p = 18;

r = 8; %rank, doesn't work for r=1

%generate a random rank r tensor
T_rand = random_rank_r_tensor(m, n, p, r);

%for movies:
%load('Ymean_CON_M') %Y is brain scan
%tensor low rank approximation of Y
%Vhat = cpd(Y,r);
%Yhat = cpdgen(Vhat);
%T_true = Yhat;

T_true = T_rand;
T_Omega = forget_tensor_entries(T_true,r); %replaces unknown entries of T_true with zeros
%T_Omega hass the structure T_Omega(:,:,1:r) = [A B;C 0], T_Omega(:,:,(r+1):k) = [D 0;0 0]

tic
T = complete_tensor(T_Omega,r); %completes T_Omega into a rank r tensor T
t = toc; %time to complete
disp(['time to complete ', num2str(t)])

err = norm(T(:)-T_true(:),inf); %error
disp(['error ',num2str(err)])

close all
fig = figure;
fig.Visible = 'off';

together = [T_true,T_Omega,T]; %puts together original, partially known, and completed tensors
M = struct('cdata',[],'colormap',[]); %M is a movie of the tensors
for i=1:p
    imshow(together(:,:,i),'InitialMagnification', 600)
    axis image
    M(i) = getframe;
end

fig.Visible = 'on';
title('left: original, middle: partially known, right: completed')
movie(M,5) %plays the movie M 5 times
%left part of movie is the true tensor
%middle part of movie is partially known tensor
%right part of movie is completed tensor

num_known = r^3+(m-r)*r^2+(n-r)*r^2+(p-r)*r^2; %number of known entries in T_Omega
known_ratio = num_known/(m*n*p); %ratio of number of known entries to number of unknown entries