
% Last modified: Jun 16, 2021
% Kenichi Shimizu: University of Glasgow 

% Run MCMC of a generalized version of 
% Antonelli et al. (2019, Bayesian Analysis) 
% using a given spike and slab prior 

% i=1:n
% y_i = beta0 + betat*T_i + X_i'*beta + N(0,sigma2)
% X_i is p by 1 
% T_i is a scalar treatment variable 
% betat is the treatment effect 
% gamma is p by 1 "inclusion indicator"
% theta is a scalar "inclusion probability" 
% w is p by 1 "weights" (needs to be estimated via EB)

% The hierarchical model 
% y_i           ~ N(beta0 + betat*T_i + X_i'*beta, sigma^2) 
% beta_j        ~ gamma_j * N(0,tau1_j^2) + (1-gamma_j) * N(0,tau0_j^2) 
% gamma_j       ~ theta^{w_j * gamma_j} * (1-theta^w_j)^{1-gamma_j}
% theta         ~ Beta(a,b) 
% sigma^2       ~ InvGamma(a,b) 
% beta0, betat  ~ N(0,K) 
% tau1_j,tau0_j ~ depends on your choice of prior 

% 1. SSVS, variances tau0 and tau1 selected (as in Narisetty and He, 2014, Annals of Statistics)
% 2. SSVS, tau0 = c x kappa, tau1= kappa, where c = 1e-4, and kappa ~ iGamma
% 3. Spike and Slab Lasso, (tau0,tau1) ~ Laplace (as in Rockova and George, 2018, Annals of Statistics)
% 4. Spike and Slab Horseshoe, (tau0,tau1) ~ Horseshoe

% BetaAll = [beta0; betat; beta] is a (p+2) by 1 vector 
% X =[X_1' ; ... ; X_n'] is a n by p matrix 
% T =[T_1,...,T_n]; is a n by 1 vector 
% Xall =[ones(n,1),T,X] is a n by (p+2) matrix 

function [BetaAllPost,GammaPost,Sigma2Post]=...
    NaiveBayes(y,X,T,...
nsave,nburn,thin,...
    prior)

[n,p]=size(X);
Xall=[ones(n,1),T,X];

BetaAllPost=zeros(p+2,nsave);
GammaPost=zeros(p,nsave);
Sigma2Post=zeros(1,nsave);

if prior >=3             % Spike and Slab class of priors
    tau1   = 1*ones(p,1);        
    tau0  = (1/n)*ones(p,1);
    pi0   = 0.2;
    kappa  = NaN;
    xi     = NaN;
    nu     = NaN;
    if prior == 3   % Simple SSVS
        method = 'ssvs_normal';
        fvalue = tpdf(sqrt(2.1*log(p+1)),5);
        tau1   = max(100*tau0,pi0*tau0/((1 - pi0)*fvalue));
    elseif prior == 4
        method = 'ssvs_student';
        kappa  = 0.01;
    elseif prior == 5
        method = 'ssvs_lasso';
        kappa  = 3;
    elseif prior == 6 
        method = 'sns_lasso';
        kappa  = 3;
     elseif prior == 7
        method = 'ssvs_horseshoe';
        kappa  = 1;
        xi     = 1;
        nu     = ones(p,1);
     end
end

% Initialize prior variances for the slopes
Q    = ones(p,1);

BetaAllPost(:,1)=0.1*randn(p+2,1);
GammaPost(:,1)=zeros(p,1);
Sigma2Post(1)=1;

c=0.001;% Fixed 
d=0.001;% Fixed 
K=10000; % Fixed 

for irep = 2:(nsave+nburn) 
% Update sigma2 
Dinv=diag(1./[K;K;Q]);
sse=sum( (y-Xall*BetaAllPost(:,irep-1)).^2 );
Sigma2Post(irep)=1/gamrnd( c+0.5*n+0.5*p, 1/(d + 0.5*sse + 0.5*BetaAllPost(:,irep-1)'*Dinv*BetaAllPost(:,irep-1)) );

% Update BetaAll 
BetaAllPost(:,irep) = randn_gibbs(y,Xall,[K;K;Q],Sigma2Post(irep),n,p+2);

% Update Q
[Q,GammaPost(:,irep),tau0,tau1,pi0,kappa,xi,nu] = ssvs_prior(BetaAllPost(3:end,irep),Sigma2Post(irep),tau0,tau1,pi0,kappa,xi,nu,method);
end 


keep=nburn+1:thin:nburn+nsave;

BetaAllPost=BetaAllPost(:,keep);
Sigma2Post=Sigma2Post(keep);

end 
