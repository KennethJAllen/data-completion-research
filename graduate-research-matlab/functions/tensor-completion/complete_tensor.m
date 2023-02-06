function X = complete_tensor(T_Omega,r)
    %completes the partially known tensor T_Omega into the rank r tensor X
    %the unknown entries of T_Omega must have the following structure
    %T_Omega(:,:,1:r) = [A B;C G]
    %T_Omega(:,:,(r+1):p) = [D F;E H]
    %A,B,C,D are fully known, E,F,G,H are unknown
    [m,n,p] = size(T_Omega);
    A = T_Omega(1:r,1:r,1:r); %assume A has multilinear rank (r,r,r)
    if r>1
        A1 = unfold_tensor(A,1);
        A2 = unfold_tensor(A,2);
        A3 = unfold_tensor(A,3);
        r1 = rank(A1);
        r2 = rank(A2);
        
        if any([r1 r2] ~= [r r]) %makes sure A has multilinear rank [r r r]
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
    
    A1 = unfold_tensor(A,1);
    AJ = A1(1:r,1:r); %Assumes A1(1:r,1:r) is full rank. To do: find full rank submatrix
    B1 = unfold_tensor(B,1);
    C1 = unfold_tensor(C,1);
    size(C1)
    CJ = C1(1:(m-r),1:r);
    D1 = unfold_tensor(D,1);
    G1 = (CJ/AJ)*B1; %complete G
    E1 = (CJ/AJ)*D1; %complete E
    G = fold_tensor(G1,[m-r,n-r,r],1); %folds mode-1 unfolding of G
    E = fold_tensor(E1,[m-r,r,p-r],1); %folds mode-1 unfolding of E

    A2 = unfold_tensor(A,2);
    AI = A2(1:r,1:r);
    B2 = unfold_tensor(B,2);
    BI = B2(1:(n-r),1:r);
    D2 = unfold_tensor(D,2);
    E2 = unfold_tensor(E,2);
    F2 = (BI/AI)*D2; %comples F
    H2 = (BI/AI)*E2; %completes H
    F = fold_tensor(F2,[r,n-r,p-r],2); %folds mode-2 unfolding of F
    H = fold_tensor(H2,[m-r,n-r,p-r],2); %folds mode-2 unfolding of H
    
    X = zeros(m,n,p);
    %assembling completion
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