function [Y_forc,alpha_mat,psi_mat,sigma_mat,gamma_mat,omega_mat]  = SSVS_VAR_MINN(y,stdata,p,delta,lambda,theta,V_prior,ndraws,forc_h)

disp('Now running SSVS')
%--------------------------DATA HANDLING-----------------------------------
[Traw M] = size(y);

if stdata == 1
    [y,mm,ss] = standard(y);
else
    mm = zeros(1,M);
    ss = ones(1,M);
end

% Generate lagged Y matrix. This will be part of the X matrix
ylag = mlag(y,p); % Y is [T x M]. ylag is [T x (pM)]

% Insert "ylag" defined above to form X matrix
X = [ones(Traw-p,1) ylag(p+1:Traw,:)];


% Form y matrix accordingly
% Delete first "p" rows to match the dimensions of X matrix
Y = y(p+1:Traw,:); % This is the final Y matrix used for the VAR

%Traw was the dimesnion of initial data. T is the number of actual 
%time series observations of Y and X (final Y & X)
T=Traw-p;

n = (1 + p*M)*M;  % n is the number of alpha parameter elements (is the
               %number of rows of the "stacked" column vector of parameters)              

% In this program we assume that the constants are unrestricted
% and that the lag coefficients are restricted
m = (n - M); % choose number of restrictions, n minus the # of intercepts
non = n - m; % unrestricted alphas (= M intercepts)
%----------------------------PRELIMINARIES---------------------------------
% Set some Gibbs - related preliminaries
nsave = 1000;  % Number of draws to save
nburn = 100;   % Number of burn-in-draws
ntot  = nsave + nburn; % Total number of draws
thin  = 1;   % Consider every thin-th draw (thin value)

% Set storage space in computer memory for parameter matrices
alpha_draws = zeros(n,nsave); % store alphas draws
psi_ii_sq = zeros(M,1); % vector of psi^2 drawn at each iteration
psi_ii_sq_draws = zeros(M,nsave); % store psi^2 draws
gammas_draws = zeros(m,nsave); % store gamma draws
omega_draws = zeros(.5*M*(M-1),nsave); % store omega draws
psi_mat_draws = zeros(M,M,nsave); % store draws of PSI matrix
alpha_mat_draws = zeros(n./M,M,nsave); % store draws of ALPHA matrix

% Set storage space in memory for cell arrays. These can be put
% individually before each "for" loop
S=cell(1,M);
s=cell(1,M-1);
omega=cell(1,M-1);
h=cell(1,M-1);
D_j=cell(1,M-1);
R_j = cell(1,M-1);
DRD_j=cell(1,M-1);
B=cell(1,M);
eta = cell(1,M-1);              
              
%---------------Prior hyperparameters for ssvs algorithm-------------------
% First get ML estimators, see eq. (8)
ALPHA_OLS = inv(X'*X)*(X'*Y);
SSE = (Y - X*ALPHA_OLS)'*(Y - X*ALPHA_OLS);
% Stack columns of ALPHA_M
alpha_OLS_vec=reshape(ALPHA_OLS,n,1);  % vector of "MLE" alphas (=vec(ALPHA_OLS))

tau_temp = [];
for irep = 1:M
    [~,prior_Vb] = Minnesota_SSBCVAR(Y,X,p,delta,lambda,theta,V_prior,T,M,n/M,irep);
    tau_temp = [tau_temp; prior_Vb(1:n/M)];
end
tau_temp(tau_temp==V_prior) = [];

% Variances for the "Phi mixture", see eq.(13)
tau_0 = 0.001;   %*ones(m,1);  % Set tau_[0i], tau_[1i]
tau_1 = tau_temp;%4;     %*ones(m,1);

% Variances for the "Eta mixture", see eq.(16)  
kappa_0 = 0.001;      %*ones(m,M-1); % Set kappa_[0ij], kappa_[1ij]
kappa_1 = V_prior;  %*ones(m,M-1);

% Hyperparameters for Phi_non ~ N_[n-m](f_non,M_non), see eq.(11)
f_non = 0*ones(non,1);
c = V_prior;
M_non = c*eye(non);

% Hyperparameters for Phi_m ~ N_[m](f_m,DRD), see eq.(11)
f_m = 0*ones(m,1); %Mean of restricted alphas 

f_tot = zeros((n/M),M); %Prior mean of all alphas(restr. & unrestr.)
for ged = 1:M
    nnn_9 = 1 + (m./M)*(ged-1);
    f_tot(:,ged) = [f_non(ged,:) ; f_m(nnn_9:(m./M)*ged)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This will change the prior mean on the 1st own lag, to be consistent with
% the other methods
for ged = 1:M
    f_tot(1+ged,ged) = delta;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_tot = reshape(f_tot,size(f_tot,1)*size(f_tot,2),1);

% Matrices of restrictions on Phi and Psi parameters, R & R_[j]
R = eye(m);   % Set here (square) R matrix of m restrictions
% Create R_[j] = R_1, R_2, R_3,... matrices of restrictions
for kk = 1:(M-1)	% Set matrix R_j of restrictions on psi
    R_j{kk} = eye(kk);
end

% Initialize Gamma and Omega vectors
gammas = ones(m,1);       % vector of Gamma

for kk_1 = 1:(M-1)
    omega{kk_1} = ones(kk_1,1);	% Omega_j
end

% Hyperparameters for Gamma ~ BERNOULLI(m,p_i), see eq. (14)
p_i = .1;

% Hyperparameters for Omega_[j] ~ BERNOULLI(j,q_ij), see eq. (17)
q_ij = .1;

% Hyperparameters for (Psi)^2 ~ GAMMA(a_i , b_i), see eq. (18)
a_i = .01;
b_i = .01;
%***************End of Preliminaries & PriorSpecification******************
%_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*


%========================== Start Sampling ================================
% tic; % Start the timer
%************************** Start the Gibbs "loop" ************************
% disp('Number of iterations');
for irep = 1:ntot
    if mod(irep,ntot/10)==0
        fprintf('%d %% \t',round(100*irep/ntot))
    end
    % STEP 1.: ----------------------------Draw "psi"----------------------
    % Draw psi|alpha,gamma,omega,DATA from the GAMMA dist.
    %----------------------------------------------------------------------

    % Get S_[j] - upper-left [j x j] submatrices of SSE
    % The following loop creates a cell array with elements S_1,
    % S_2,...,S_j with respective dimensions 1x1, 2x2,...,jxj
    for kk_2 = 1:M                         
        S{kk_2} = SSE(1:kk_2,1:kk_2);
    end

    % Set also SSE =(s_[i,j]) & get vectors s_[j]=(s_[1,j] , ... , s_[j-1,j])
    for kk_3 = 2:M
        s{kk_3 - 1} = SSE(1:(kk_3 - 1),kk_3);
    end
    
    % Parameters for Heta|omega ~ N_[j-1](0,D_[j]*R_[j]*D_[j]), see eq. (15)
    % Create and update h_[j] matrix
    % If omega_[ij] = 0 => h_[ij] = kappa0, else...
    for kk_4 = 1:M-1
        omeg = cell2mat(omega(kk_4));
        het = cell2mat(h(kk_4));
        for kkk = 1:size(omeg,1)
            if omeg(kkk,1) == 0
                het(kkk,1) = kappa_0;
            else
                het(kkk,1) = kappa_1;
            end
        end
        h{kk_4} = het;
    end

    % D_j = diag(h_[1j],...,h[j-1,j])
    for kk_5 = 1:M-1
        D_j{kk_5} = diag(cell2mat(h(kk_5)));
    end

    % Now create covariance matrix D_[j]*R_[j]*D_[j], see eq. (15)
    for kk_6 = 1:M-1
        DD = cell2mat(D_j(kk_6));
        RR = cell2mat(R_j(kk_6));
        DRD_j{kk_6} = (DD*RR*DD);
    end

    % Create B_[i] matrix
    for rr = 1:M
        if rr == 1
            B{rr} = b_i + 0.5*(SSE(rr,rr));
        elseif rr > 1
            s_i = cell2mat(s(rr-1));
            S_i = cell2mat(S(rr-1));
            DiRiDi = cell2mat(DRD_j(rr-1));
            B{rr} = b_i + 0.5*(SSE(rr,rr) - s_i'*inv(S_i + inv(DiRiDi))*s_i);
        end
    end

    % Now get B_i from cell array B, and generate (psi_[ii])^2
    B_i = cell2mat(B);
    for kk_7 = 1:M
        % If you have the Statistics toolbox, use "gamrnd" instead
	    psi_ii_sq(kk_7,1) = gamm_rnd(1,1,(a_i + 0.5*T),B_i(1,kk_7));
    end
    
    % STEP 2.: ----------------------------Draw "eta"----------------------
    % Draw eta|psi,alpha,gamma,omega,DATA from the [j-1]-variate NORMAL dist.
    %----------------------------------------------------------------------
    for kk_8 = 1:M-1
        s_i = cell2mat(s(kk_8));
        S_i = cell2mat(S(kk_8));
        DiRiDi = cell2mat(DRD_j(kk_8));
        miu_j = - sqrt(psi_ii_sq(kk_8+1))*(inv(S_i + inv(DiRiDi))*s_i);
        Delta_j = inv(S_i + inv(DiRiDi));
    
        eta{kk_8} = miu_j + chol(Delta_j)'*randn(kk_8,1);
    end
    
    % STEP 3.: --------------------------Draw "omega"----------------------
    % Draw omega|eta,psi,alpha,gamma,omega,DATA from BERNOULLI dist.
    %----------------------------------------------------------------------    
    omega_vec = []; %temporary vector to store draws of omega
    for kk_9 = 1:M-1
        omeg_g = cell2mat(omega(kk_9));
        eta_g = cell2mat(eta(kk_9));
        for nn = 1:size(omeg_g)  % u_[ij1], u_[ij2], see eqs. (32 - 33)
            u_ij1 = (1./kappa_0)*exp(-0.5*((eta_g(nn))^2)./((kappa_0)^2))*q_ij;
            u_ij2 = (1./kappa_1)*exp(-0.5*((eta_g(nn))^2)./((kappa_1)^2))*(1-q_ij);
            ost = u_ij1./(u_ij1 + u_ij2);
            omeg_g(nn,1) = bernoullirnd(ost);
            omega_vec = [omega_vec ; omeg_g(nn,1)]; %#ok<AGROW>
        end
        omega{kk_9} = omeg_g;
    end
    
    % STEP 4.: --------------------------Draw "alpha"------------------------
    % Draw alpha|gamma,Sigma,omega,DATA from NORMAL dist.
    %----------------------------------------------------------------------    
    
    % Create PSI matrix from individual elements of "psi_ii_sq" and "eta"
    PSI_ALL = zeros(M,M);
    for nn_1 = 1:M
        PSI_ALL(nn_1,nn_1) = sqrt(psi_ii_sq(nn_1,1));
    end

    for nn_2 = 1:M-1
        eta_gg = cell2mat(eta(nn_2));
        for nnn = 1:size(eta_gg,1)
            PSI_ALL(nnn,nn_2+1) = eta_gg(nnn);
        end
    end
    
    % Hyperparameters for Phi_m|gamma ~ N_[m](0,D*R*D), see eq.(12)
    h_i = zeros(m,1);   % h_i is tau_0 if gamma=0 and tau_1 if gamma=1
    for nn_3 = 1:m
        if gammas(nn_3,1) == 0
           h_i(nn_3,1) = tau_0;
        elseif gammas(nn_3,1) == 1
           h_i(nn_3,1) = tau_1(nn_3);
        end
    end
    D=sparse(1:m,1:m,h_i); % Create D. Here D=diag(h_i) will also do
    DRD = D*R*D;   % Prior covariance matrix for Phi_m
    psi_xx = kron((PSI_ALL*PSI_ALL'),(X'*X));
    temp1 = zeros(size(M_non,1),size(DRD,1));
    temp2 = zeros((n./M),M);
    M_vec = diag(M_non);
    DRD_vec = diag(DRD);
    for sed = 1:M
        nnn_9 = 1 + (m./M)*(sed-1);
        temp2(:,sed) = [M_vec(sed,:) ; DRD_vec(nnn_9:(m./M)*sed)];
    end
    temp2 = reshape(temp2,size(temp2,1)*size(temp2,2),1);
    temp2 = sparse(1:n,1:n,temp2);
    Delta_alpha = inv(psi_xx + inv(temp2));
    
    miu_alpha = Delta_alpha*((psi_xx)*alpha_OLS_vec + inv(temp2)*f_tot);
    
    alphas = miu_alpha + chol(Delta_alpha)'*randn(n,1);
    
    alpha_mat = reshape(alphas,n./M,M);
    alpha_temp = alpha_mat(2:(n/M),:);
    reshape(alpha_temp,size(alpha_temp,1)*size(alpha_temp,2),1);
    % STEP 5.: --------------------------Draw "gamma"----------------------
    % Draw gamma|alpha,psi,eta,omega,DATA from BERNOULLI dist.
    %----------------------------------------------------------------------    
    for nn_6 = 1:m
        u_i1 = (1./tau_0)*exp(-0.5*(alpha_temp(nn_6)./(tau_0))^2)*p_i;
        u_i2 = (1./tau_1(nn_6)).*exp(-0.5*(alpha_temp(nn_6)./(tau_1(nn_6)))^2).*(1-p_i);
        gst = u_i1./(u_i1 + u_i2);
        gammas(nn_6,1) = bernoullirnd(gst);
    end
    
    % Save new Sum of Squared Errors (SSE)
    SSE = (Y - X*alpha_mat)'*(Y - X*alpha_mat);

    % Store matrices
    if irep>nburn
        alpha_draws(:,irep-nburn) = alphas;
        psi_ii_sq_draws(:,irep-nburn) = psi_ii_sq;
        psi_mat_draws(:,:,irep-nburn) = PSI_ALL;
        alpha_mat_draws(:,:,irep-nburn) = alpha_mat;
        gammas_draws(:,irep-nburn) = gammas;
        omega_draws(:,irep-nburn) = omega_vec;
    end
    
end
% ==========================Finished Sampling==============================
% =========================================================================
%clc;
% disp('Finished sampling successfully')

% Do thining in case of high correlation
thin_val = 1:thin:(nsave);

alpha_draws = alpha_draws(:,thin_val);
psi_ii_sq_draws = psi_ii_sq_draws(:,thin_val);
psi_mat_draws = psi_mat_draws(:,:,thin_val);
alpha_mat_draws = alpha_mat_draws(:,:,thin_val);
gammas_draws = gammas_draws(:,thin_val);
omega_draws = omega_draws(:,thin_val);

% Find average of restriction indices Gamma (note that the constant is not
% included!)
gammas = mean(gammas_draws,2);
gamma_mat = zeros(p*M,M);
for nn_5 = 1:M
    nnn_1 = 1 + p*M*(nn_5-1);
    gamma_mat(:,nn_5) = gammas(nnn_1:p*M*nn_5);
end

gamma_mat = [ones(1,M); gamma_mat];

% Find average of restriction indices Omega
omega = mean(omega_draws,2);
omega_mat = zeros(M,M);
for nn_5 = 1:M-1
    ggg = omega(((nn_5-1)*(nn_5)/2 + 1):(nn_5*(nn_5+1)/2),:);
    omega_mat(1:size(ggg,1),nn_5+1) = ggg;
end

alpha_mat = squeeze(mean(alpha_mat_draws,3));
psi_mat = squeeze(mean(psi_mat_draws,3));
sigma_mat = inv(psi_mat*psi_mat');

% Matrices in companion form   
By = [alpha_mat(2:end,:)'; eye(M*(p-1)) , zeros(M*(p-1),M)];   
Sy = zeros(M*p,M*p);
Sy(1:M,1:M) = sigma_mat;
miu = zeros(M*p,1);
miu(1:M,:) = alpha_mat(1,:)';

% -------| STEP 3: Prediction   
Y_pred = zeros(ndraws,M,forc_h); % Matrix to save prediction draws
    
% Now do prediction using standard formulas (see Lutkepohl, 2005)    
VAR_MEAN = 0;
VAR_VAR = 0;
    
X_FORE = [Y(end,:) X(end,2:M*(p-1)+1)];
BB = speye(M*p);
for ii = 1:forc_h % not very efficient, By^(ii-1) can be defined once 
    VAR_MEAN =  VAR_MEAN + BB*miu;
    FORECASTS = VAR_MEAN + (BB*By)*X_FORE';
    if ndraws > 1
        VAR_VAR = VAR_VAR + BB*Sy*BB';
        Y_pred(:,:,ii) = (repmat(FORECASTS(1:M),1,ndraws) +  chol(VAR_VAR(1:M,1:M))'*randn(M,ndraws))';
    else
        Y_pred(:,:,ii) = repmat(FORECASTS(1:M),1,ndraws);
    end
    BB = BB*By;
end

% Store predictive draws/mean
Y_forc = repmat(mm,ndraws,1,forc_h) + repmat(ss,ndraws,1,forc_h).*Y_pred;
Y_forc = permute(Y_forc,[1 3 2]);

fprintf('\n');