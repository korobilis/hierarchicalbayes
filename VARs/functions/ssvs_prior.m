function  [Q,invQ,miu,tau] = ssvs_prior(beta,c0,tau,b0,pi0)

%% SSVS prior
l_0 = (1-pi0).*normpdf(beta,0,sqrt(c0));   
l_1 = pi0.*normpdf(beta,0,sqrt(tau));
temp  = min((l_1./l_0),10000000);    
gamma = bernoullimrnd(length(beta),temp./(1+temp));
    
% update tau  
rho1 = 1 + 1./2;
rho2 =  b0 + (beta.^2)./2;    %%%%%%%%%% PLAY WITH THE VALUE OF THE PRIOR HERE (CHANGE FROM 0.00001 to 1)
tau = 1./gamrnd(rho1./1,1./rho2);   
tau = max(tau,c0);       
Q = tau;  
Q(gamma==0) = c0;
invQ = 1./Q;

miu = zeros(length(beta),1);