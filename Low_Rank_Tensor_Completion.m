%written by K. Allen under Dr. Ming-Jun Lai's supervision
%from K. Allen's dissertation A Geometric Approach to Low-Rank Matrix and Tensor Completion

%given an m x n x p partially known tensor T_Omega, where T_Omega has the form:
%T_Omega = [A B;C unknowns],[D unknowns;unknowns unknowns]
%completes T_Omega into a multilinear rank (r,r,r) tensor T
%A is an r x r x r multilinear rank (r,r,r) fully known subtensor
%B is corresponding r x (n-r) x r subtensor
%C is corresponding (m-r) x r x r subtensor
%D is corresponding r x r x (p-r) subtensor

%assumes that the first r x r sub-matrix of every mode-i unfolding of A is invertible
%to do: search for r x r invertible submatrices of the mode-i unfoldings of A

%m x n x p tensor
m = 18;
n = 19;
p = 20;

r = 10; %rank, doesn't work for r=1

%generate a random rank r tensor
T_rand = zeros(m,n,p);
for i=1:r
    a = rand(m,1);
    b = rand(n,1);
    c = rand(p,1);
    X = tensorproduct(a,b,c);
    T_rand = T_rand + X;
end

%for movies:
%load('Ymean_CON_M') %Y is brain scan
%tensor low rank approximation of Y
%Vhat = cpd(Y,r);
%Yhat = cpdgen(Vhat);
%T_true = Yhat;

T_true = T_rand;
T_Omega = forget(T_true,r); %replaces unknown entries of T_true with zeros
%T_Omega hass the structure T_Omega(:,:,1:r) = [A B;C 0], T_Omega(:,:,(r+1):k) = [D 0;0 0]

tic
T = complete(T_Omega,r); %completes T_Omega into a rank r tensor T
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

function X = complete(T_Omega,r)
    %completes the partially known tensor T_Omega into the rank r tensor X
    %the unknown entries of T_Omega must have the following structure
    %T(:,:,1:r) = [A B;C G]
    %T(:,:,(r+1):p) = [D F;E H]
    %A,B,C,D are fully known, E,F,G,H are unknown
    [m,n,p] = size(T_Omega);
    A = T_Omega(1:r,1:r,1:r); %assume A has multilinear rank (r,r,r)
    if r>1
        A1 = unfold(A,1);
        A2 = unfold(A,2);
        A3 = unfold(A,3);
        r1 = rank(A1);
        r2 = rank(A2);
        r3 = rank(A3);
        if any([r1 r2 r3] ~= [r r r]) %makes sure A has multilinear rank [r r r]
            error('The multilinear rank of A is not equal to [r r r]')
        end
    else %if r=1
        A1 = T_Omega(1,1,1);
        A2 = A1;
        A3 = A1;
        if T_Omega(1,1,1) == 0
            error('The multilinear rank of A is not equal to [r r r]')
        end
    end
    B = T_Omega(1:r,(r+1):n,1:r);
    C = T_Omega((r+1):m,1:r,1:r);
    D = T_Omega(1:r,1:r,(r+1):p);
    
    A1 = unfold(A,1);
    AJ = A1(1:r,1:r); %Assumes A1(1:r,1:r) is full rank. To do: find full rank submatrix
    B1 = unfold(B,1);
    C1 = unfold(C,1);
    CJ = C1(1:(m-r),1:r);
    D1 = unfold(D,1);

    G1 = (CJ/AJ)*B1; %complete G
    E1 = (CJ/AJ)*D1; %complete E
    G = refold(G1,[m-r,n-r,r],1);
    E = refold(E1,[m-r,r,p-r],1);
    A2 = unfold(A,2);
    AI = A2(1:r,1:r);
    B2 = unfold(B,2);
    BI = B2(1:(n-r),1:r);
    D2 = unfold(D,2);
    E2 = unfold(E,2);

    F2 = (BI/AI)*D2;
    H2 = (BI/AI)*E2;
    F = refold(F2,[r,n-r,p-r],2);
    H = refold(H2,[m-r,n-r,p-r],2);
    X = zeros(m,n,p);
    X(:,:,1:r) = [A B;C G];
    X(:,:,(r+1):p) = [D F;E H];
    
    %checks to see if Tcomplete has multilinear rank [r r r]
    %r1 = rank(unfold(T_Omega,1));
    %r2 = rank(unfold(T_Omega,2));
    %r3 = rank(unfold(T_Omega,3));
    %if any([r1 r2 r3] ~= [r r r]) %makes sure Tcomplete has multilinear rank [r r r]
    %    disp('Warning: T may not have a rank [r r r] completion.')
    %end
end

function X = unfold(T, i)
%mode-i unfolding of T
dim = size(T);
X = reshape(shiftdim(T,i-1), dim(i), []);
end

function X = refold(T,tensor_size,i)
%refolds the mode-i unfolding of T back into T
%refold((unfold(T,i),[n m k],i) = T
X = reshape(T, circshift(tensor_size,-i+1));
X = shiftdim(X,-i+4);
end

function X = tensorproduct(a,b,c)
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

function X = forget(T,r)
%replaces entries of T in unknown positions with zeros
%input: T(:,:,1:r) = [A B;C G], T(:,:,r+1:k) = [D F;E H]
%output: X(:,:,1:r) = [A B;C 0], X(:,:,r+1:k) = [D 0;0 0]
[m,n,p] = size(T);
X = T;
X(r+1:m,r+1:n,1:r) = zeros(m-r,n-r,r); %sets G to zero
X(r+1:m,1:r,r+1:p) = zeros(m-r,r,p-r); %sets E to zero
X(1:r,r+1:n,r+1:p) = zeros(r,n-r,p-r); %sets F to zero
X(r+1:m,r+1:n,r+1:p) = zeros(m-r,n-r,p-r); %sets H to zero
end