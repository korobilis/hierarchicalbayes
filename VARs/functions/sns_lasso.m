function  [Q,invQ,miu,tau0,tau1] = sns_lasso(beta,tau0,tau1,pi0)

p = length(beta);

%% SSVS prior
l_0 = (1-pi0).*normpdf(beta,0,sqrt(tau0));   
l_1 = pi0.*normpdf(beta,0,sqrt(tau1));
temp  = min((l_1./l_0),10000000);    
gamma = bernoullimrnd(p,temp./(1+temp));
    
% Update lam0 and lam1
lam1 = 0.01;
lam0 = gamrnd(p + 1,(0.5*sum(tau0) + 3));

% Update tau0 and tau1
tau0 =  min(1./random('InverseGaussian',sqrt(lam0*beta.^2),lam0,p,1),1e+6);
tau1 =  min(1./random('InverseGaussian',sqrt(lam1*beta.^2),lam1,p,1),1e+6);

Q = (1-gamma).*tau0 + gamma.*tau1;
invQ = 1./Q;

miu = zeros(length(beta),1);