%written by K. Allen under Dr. Ming-Jun Lai's supervision
%from K. Allen's dissertation A Geometric Approach to Low-Rank Matrix and Tensor Completion

%schur complement gradient descent method w/ maxvol algorithm for image completion
close all

r = 18; %rank
plots_on = 1; %set equal to 1 for singular value plots

%some matrices to use
load penny %P is penny picture
[U,Sigma_P,V] = svd(P);

M = P; %which true matrix to use
[m,n] = size(M);

frac_num_known = 0.75; %fraction of entries known
num_known = round(m*n*frac_num_known); %number of known entries
Omega = RandomMask(m,n,num_known); %random mask

X0 = zeros(m,n);
X0(Omega) = M(Omega); %initial guess with known entries

I = randperm(m,r); %random initial rows for A
J = randperm(n,r); %random initial columns for A

N = 3000; %number of steps
N_pre = 50; %number of preconditioning steps
h = ones(1,N)*1e-3; %step size

G = zeros(m,n);
X = X0;

%preconditions using CUR approximation with maxvol
for i=1:N_pre
    I_initial = randperm(m,r); %random initial rows for A
    J_initial = randperm(n,r); %random initial columns for A
    [I,J] = alt_maxvol(X,I_initial,J_initial);
    C = X(:,J);
    U = X(I,J);
    R = X(I,:);
    X = C*(U\R); %rank r approximation of X
    X(Omega) = M(Omega); %sets known entries
    X(X>255) = 255;
    X(X<0) = 0; %makes sure entries are bounded
    MSE = (1/(m*n))*norm(X(:)-M(:))^2;
    PSNR = 10*log10(65025/MSE)%peak signal to noise ratio
end

%gradient descent
for k=1:N
    %maximum volume step
    [I,J] = alt_maxvol(X,I,J); %this file may be on another folder
    I_comp = setdiff(1:m,I); %complement of I in 1:m
    J_comp = setdiff(1:n,J); %complement of J in 1:n
    
    %gradient descent step
    A = X(I,J);
    B = X(I,J_comp);
    C = X(I_comp,J);
    D = X(I_comp,J_comp);
    S = D-(C/A)*B; %schur complement of X with respect to A
    
    %gradients
    GA = (A'\C')*S*(B'/A');
    GB = -(A'\C')*S;
    GC = -S*(B'/A');
    GD = S;
    G(I,J) = GA;
    G(I,J_comp) = GB;
    G(I_comp,J) = GC;
    G(I_comp,J_comp) = GD;
    G(Omega) = 0; %sets entries in known positions equal to zero

    %gradient step
    X = X-h(k)*G;
    X(X>255) = 255; %ensures entries are bounded, maximum value
    X(X<0) = 0; %minimum value
    C_norm = norm(X(:)-M(:),'inf');
    MSE = (1/(m*n))*norm(X(:)-M(:))^2;
    PSNR = 10*log10(65025/MSE)%peak signal to noise ratio
end

if plots_on==1
    [~,SigmaM,~] = svd(M);
    sv_M = diag(SigmaM);
    [~,Sigma0,~] = svd(X0);
    sv0 = diag(Sigma0);
    [~,Sigma,~] = svd(X);
    sv_X = diag(Sigma);

    %plot parameters
    width = 3;     % Width in inches
    height = 3;    % Height in inches
    alw = 0.75;    % AxesLineWidth
    fsz = 11;      % Fontsize
    lw = 1.5;      % LineWidth
    msz = 6;       % MarkerSize
    xx = 1:min(m,n);

    %plot
    h = figure('units','inch');
    fig1_comps.fig = gcf;
    set(gca, 'LineWidth', alw,'YScale', 'log'); %<- Set properties
    set(gcf, 'Position',  [0, 0, 8, 6])
    hold on
    fig1_comps.p1 = plot(xx,sv0,'b');
    fig1_comps.p2 = plot(xx,sv_M,'r');
    fig1_comps.p3 = plot(xx,sv_X,'g');
    hold off

    %labels
    legend('Initial Guess','Original','Completed')
    xlabel('Index');
    ylabel('Sigular Value')
    title('Singular Values')
    %plot properties
    set(fig1_comps.p1, 'LineStyle', '-', 'LineWidth', lw, 'Marker', 'o', 'MarkerSize', 8, 'MarkerIndices',1:2:length(xx));
    set(fig1_comps.p2, 'LineStyle', '-', 'LineWidth', lw, 'Marker', 's', 'MarkerSize', 8, 'MarkerIndices',1:2:length(xx));
    set(fig1_comps.p3, 'LineStyle', '-', 'LineWidth', lw, 'Marker', '^', 'MarkerSize', 8, 'MarkerIndices',1:2:length(xx));
end

figure
imshow(M/255,'InitialMagnification', 800)
title('true image')

figure
imshow(X0/255,'InitialMagnification', 800)
title('partially known image')

figure
imshow(X/255,'InitialMagnification', 800)
title('completed image with schur-complement MV grad descent')