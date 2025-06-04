% AUTO_BVAR.m   Fast implementation of Gibbs sampler for Bayesian Vector
%               Autoregressions, using automatic hierarchical shrinkage
%               priors
% 
% Written by Dimitris Korobilis, March 2020
% University of Glasgow

clear all;
% close all;
clc;

% Add path of data and functions
addpath('functions');
addpath('data');

%-------------------------------PRELIMINARIES--------------------------------------
ngibbs     = 1000;          % Number of Gibbs sampler iterations
nburn      = 0.2*ngibbs;   % Number of iterations to discard
p          = 1;            % p is number of lags in the VAR part                         

prior = 3;              % 1: Normal-iGamma prior (Student t)
                        % 2: SSVS with Normal(0,tau_0) and Normal-iGamma components
                        % 3: Horseshoe prior


%----------------------------- END OF PRELIMINARIES --------------------------------
tic;
%----------------------------------LOAD DATA----------------------------------------   
[Y,PHI,C,S] = simvardgpcon(200,30,1,2);
%[Y,ynames,tcode,fcode] = load_data('MEDIUM');
%Y = zscore(Y);

% ===================================| VAR EQUATION |==============================
[y,X,T,M,KK,K,~,~] = prepare_BVAR_matrices(Y,p);

% Some useful matrices for the time-varying parameters
TKK       = T*KK;   
H         = speye(TKK,TKK) - sparse(KK+1:TKK,1:(T-1)*KK,ones(1,(T-1)*KK),TKK,TKK);
Hinv      = speye(TKK,TKK)/H;
J         = tril(ones(T,T));
bigG      = full(SURform(X)*Hinv);

est_meth = 2*(T>TKK) + 1*(T<=TKK);
% ==============| Define priors
Q   = cell(M,1);   miu = cell(M,1); lambda = cell(M,1); tau = zeros(M,1);

for ieq = 1:M
    Q{ieq,1}    = .01*ones(TKK+ieq-1,1);
    miu{ieq,1}  = zeros(TKK+ieq-1,1);  
    if prior == 1
        % Student_T shrinkage prior
        b0 = 0.01;
    elseif prior == 2       
        % SSVS prior
        c0 = (0.01)^2;
        lambda{ieq,1} = 4*ones(TKK+ieq-1,1);    % "local" shrinkage parameters
        pi0 = 0.25;                            % "global" shrinkage parameter
        b0 = 0.01;
    elseif prior == 3   
        % Horseshoe prior
        lambda{ieq,1} = 0.1*ones(TKK+ieq-1,1);  % "local" shrinkage parameters
        tau(ieq,1) = 0.1*ones(1,1);            % "global" shrinkage parameter
    end
end

% Initialize matrices
BETA = zeros(KK,M,T); 
A = eye(M);
resid = zeros(T,M);
sigma_t = 0.1*ones(T,M);
h = ones(T,M);   
sig = 0.1*ones(M,1);

%========|STORAGE MATRICES for MCMC:
beta_save  = zeros(ngibbs,KK,M,T);
A_save = zeros(ngibbs,M,M);
OMEGA_save = zeros(ngibbs,M,M,T);
%======================= BEGIN MCMC ESTIMATION =======================
tic;
for iter = 1: (ngibbs + nburn)
    if mod(iter,100)==0   
        disp(['This is iteration ' num2str(iter)])         
        disp([num2str(100*(iter/(ngibbs+nburn))) '% completed'])
        toc
    end
    
    %% Draw VAR equation-by-equation
    X_new = bigG;
    for ieq = 1:M                  
        % =====| Step 1: Sample VAR coefficients and covariances                
        u = (1./sqrt(sigma_t(:,ieq)));
        ys = y(:,ieq).*u;                       % Standardize y for GLS formula
        xs = X_new.*u;                          % Also standardize x           
        betaD = randn_gibbs(ys,xs,miu{ieq,1},Q{ieq,1},T,TKK+(ieq-1),est_meth);  % sample regression coefficients
        beta = Hinv*betaD(1:TKK); beta_mat = reshape(beta,KK,T);
        BETA(:,ieq,:) = beta_mat;
        A(ieq,1:ieq-1) = betaD(TKK+1:end);
        
        %--draw prior variance of BETA and A
        if prior == 1
            [Q{ieq,1},~,miu{ieq,1}] = student_T_prior(betaD,b0);
        elseif prior == 2       
            [Q{ieq,1},~,miu{ieq,1},lambda{ieq,1}] = ssvs_prior(betaD,c0,lambda{ieq,1},b0,pi0);
        elseif prior == 3
            [Q{ieq,1},~,miu{ieq,1},lambda{ieq,1},tau(ieq)] = horseshoe_prior(betaD,TKK+(ieq-1),lambda{ieq,1},tau(ieq));
        end
        Q{ieq,1}(ieq+1) = 1; % Prior variance on AR(1) lag
        Q{ieq,1}(1) = 10;    % Prior variance on intercept
        
        % =====| Step 2: Sample regression variance
        yhat = y(:,ieq) - X_new*betaD;                           % regression residuals
        ystar  = log(yhat.^2 + 1e-6);                           % log squared residuals        
        [h(:,ieq), ~] = SVRW(ystar,h(:,ieq),sig(ieq),4);        % log stochastic volatility using Chan's filter   
        sigma_t(:,ieq)  = exp(h(:,ieq));                        % convert log-volatilities to variances
        r1 = 1 + T - 1;   r2 = 0.01 + sum(diff(h(:,ieq)).^2)';  % posterior moments of variance of log-volatilities
        sig(ieq) = 1./gamrnd(r1./2,2./r2);                      % sample variance of log-volatilities from Inverse Gamma
                        
        % Construct residual to be used in next VAR equation
        resid(:,ieq) = (y(:,ieq) - X_new*betaD);
        X_new = [X_new, resid(:,ieq)];
    end
    
    OMEGA = zeros(M,M,T);
    for t = 1:T 
        OMEGA(:,:,t) = A*diag(sigma_t(t,:))*A';
    end
    
    if iter > nburn
        % Save draws
        beta_save(iter-nburn,:,:,:) = BETA;
        A_save(iter-nburn,:,:,:) = A;
        OMEGA_save(iter-nburn,:,:,:) = OMEGA;
    end    
end

toc;