%written by K. Allen under Dr. Ming-Jun Lai's supervision
%from K. Allen's dissertation A Geometric Approach to Low-Rank Matrix and Tensor Completion

%alternating projection method for image completion
%projection is calculated with MVSD
addpath('maxvol-algorithms')

r = 18; %rank
plots_on = 0;

load penny %P is penny picture

M = P; %which true matrix to use
[m,n] = size(M);

frac_num_known = 0.75; %fraction of entries known
num_known = round(m*n*frac_num_known); %number of known entries
Omega = random_mask(m,n,num_known);

X0 = zeros(m,n);
X0(Omega) = M(Omega); %initial guess with known entries

N = 1000; %number of steps

X = X0;
for k=1:N
    I_initial = randperm(m,r); %initial rows for A
    J_initial = randperm(n,r); %initial columns for A
    [I,J] = alternating_maxvol(X,I_initial,J_initial);
    C = X(:,J);
    U = X(I,J);
    R = X(I,:);
    X = C*(U\R); %rank r approximation of X
    X(Omega) = M(Omega); %sets known entries
    X(X>255) = 255;
    X(X<0) = 0; %makes sure entries are bounded
    MSE = (1/(m*n))*norm(X-M,'fro')^2; %MSE
    PSNR = 10*log10(65025/MSE); %peak signal to noise ratio
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
title('completed image with MVSD alternating projection')

pause
