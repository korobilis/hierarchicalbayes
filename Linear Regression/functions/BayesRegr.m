function [beta_draws,sigma2_draws,gamma_draws,pi0_draws] = BayesRegr(y,X,nsave,nburn,prior)
% BayesRegr: Function that does flexible estimation of Bayesian "mean" regression (used in Monte Carlo study)
%  INPUTS
%    y      LHS variable of interest
%    X      RHS matrix
%  nsave    Number of Gibbs draws to store after convergence
%  nburn    Number of initial Gibbs draws to discard
%  prior    Shrinkage prior to use. Choices are:
%           1: Normal-iGamma prior (Student t)
%           2: Bayesian Lasso prior (Park and Casella 2008) 
%           3: Horseshoe prior (Makalic and Schmidt 2015)
%           4: Horseshoe prior (Slice sampler of Johndrow etal 2020)
%           5: SSVS, variances tau0 and tau1 selected (as in Narisetty and He, 2014, Annals of Statistics)
%           6: SSVS, tau0 = c x kappa, tau1= kappa, where c = 1e-4, and kappa ~ iGamma	(SSVS - SBL)
%           7: SSVS, tau0 = c x kappa, tau1= kappa, where c = 1e-4, and kappa ~ Laplace (SSVS - LASSO)
% 			8: Spike and Slab Lasso, (tau0,tau1) ~ Laplace (as in Rockova and George, 2018, Annals of Statistics)
%           9: Spike and Slab Horseshoe
%           10: Spike and Slab variable selection (e.g. Kuo and Mallick, 1997)
%  OUTPUTS
%  beta_draws  Samples from posterior of regression coefficients
% sigma_draws  Samples from posterior of regression variance
%==========================================================================
% Written by Dimitris Korobilis, University of Glasgow
% Modified by Kenichi Shimizu
% This version: 11 July 2021
%==========================================================================

[n,p] = size(X);

% ==============| Define priors
% prior for sigma2
c=0.1;d=0.1;
% prior for beta
Q    = .01*ones(p,1); % Initialize prior variances for beta 
if prior >=1 && prior <=4       % continuous priors 
    prior_type='CTS';
elseif prior >=5 && prior <= 9  % SSVS priors 
    prior_type='SSVS';
elseif prior ==10               % Spike and Slab of Kuo and Malick (1998)
    prior_type='SS' ;
end 

switch upper(prior_type)
    case 'CTS'
        b0=NaN;
        tau=NaN;
        kappa=NaN;
        xi=NaN;
        nu=NaN;
        lambda=NaN;
       if prior == 1            % Student_T shrinkage prior    
        method='student-t';
        b0 = 0.01;
       elseif prior == 2  
        method='lasso';
        tau=ones(p,1);
        kappa=3;
       elseif prior == 3        % Horseshoe prior  
        method='horseshoe-slice';
        lambda = 0.1*ones(p,1);  % "local" shrinkage parameters, one for each Tp+1 parameter element of betaD
        tau = 0.1;               % "global" shrinkage parameter for the whole vector betaD    
       elseif prior ==4 
        method='horseshoe-mixture';
        kappa=1;
        xi=1;
        nu=1; 
       end 
    case 'SSVS'
        tau1   = 1*ones(p,1);
        tau0   = (1/n)*ones(p,1);
        pi0    = 0.2;
        kappa  = NaN;
        xi     = NaN;
        nu     = NaN;
       if prior == 5   % Simple SSVS
        method = 'ssvs_normal';
        fvalue = tpdf(sqrt(2.1*log(p+1)),5);
        tau1   = max(100*tau0,pi0*tau0/((1 - pi0)*fvalue));
       elseif prior == 6
        method = 'ssvs_student';
        kappa  = 0.01;
       elseif prior == 7
        method = 'ssvs_lasso';
        kappa  = 3;
       elseif prior == 8
        method = 'sns_lasso';
        kappa  = 3;
       elseif prior == 9
        method = 'ssvs_horseshoe';
        kappa  = 1;
        xi     = 1;
        nu     = ones(p,1);            
       end 
    case 'SS'
        method  = 'sns_normal';
        tau    = 9*ones(p,1);
        pi0    = 0.2;
        kappa  = 1;
        lambda = ones(p,1);
        xi     = 1;
        nu     = ones(p,1);
end 


   
       

% ==============| Initialize vectors
beta  = rand(p,1);
gamma = ones(p,1);
% Storage space for Gibbs draws
beta_draws   = zeros(nsave,p);
sigma2_draws = zeros(nsave,1);
pi0_draws    = zeros(nsave,1);
gamma_draws  = zeros(nsave,p);

% =========| GIBBS sampler starts here
iter = 500;             % Print every "iter" iteration
fprintf('Iteration 0000')
for irep = 1:(nsave+nburn)
    % Print every "iter" iterations on the screen
    if mod(irep,iter) == 0
        fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%4d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',irep)
    end

    % Sample stochastic volatility sigma2
    Qinv=diag(1./Q);
    sse=sum( (y-X*beta).^2 );
    sigma2=1/gamrnd( c+0.5*n+0.5*p , 1./(d + 0.5*sse + 0.5*beta'*Qinv*beta) );
    
    % Sample regression coefficients beta
    beta = randn_gibbs(y,X,Q,sigma2,n,p);
 
    %--draw prior variance 
    switch upper(prior_type)
        case 'CTS'
            [Q,kappa,xi,nu]=cts_prior(beta,sigma2,b0,tau,kappa,xi,nu,lambda,method);
        case 'SSVS'
            [Q,gamma,tau0,tau1,pi0,kappa,xi,nu] = ssvs_prior(beta,sigma2,tau0,tau1,pi0,kappa,xi,nu,method);
        case 'SS'
            [Q,gamma,tau,pi0,kappa,lambda,xi,nu] = sns_prior(y,X,sigma2,beta,gamma,tau,pi0,kappa,lambda,xi,nu,method);
    end 
    
    
    if irep > nburn
        switch upper(prior_type)
        case 'CTS'
            beta_draws(irep-nburn,:) = beta;
        case {'SSVS','SS'} 
            beta_draws(irep-nburn,:) = beta.*gamma;
            pi0_draws(irep-nburn,1) = pi0;
            gamma_draws(irep-nburn,:) = gamma';
        end
        sigma2_draws(irep-nburn,1) = sigma2;      
    end    
end
fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8)
end


% 
% function [Beta] = randn_gibbs(y,X,lambda,n,p)
% % Sample Gaussian posterior efficiently
% if n<p
%     U = bsxfun(@times,lambda,X');
%     %% step 1 %%
%     u = normrnd(0,sqrt(lambda));
%     v = X*u + randn(n,1);
%     %% step 2 %%
%     v_star = ((X*U) + eye(n))\(y-v);
%     Beta = (u + U*v_star);
% else  
%     Q_star=X'*X;
%     Dinv = diag(1./lambda);       
%     L=chol((Q_star + Dinv),'lower');
%     v=L\(y'*X)';
%     mu=L'\v;
%     u=L'\randn(p,1);
%     Beta = mu+u;
% end
%     
% end
