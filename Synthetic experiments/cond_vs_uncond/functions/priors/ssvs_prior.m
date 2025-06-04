
function  [Q,gamma,tau0,tau1,pi0,kappa,xi,nu] = ssvs_prior(beta,sigma2,tau0,tau1,pi0,kappa,xi,nu,method)

p = length(beta);

% SSVS/SNS class of priors
% 1) Update model probabilities
l_0 = (1-pi0).*normpdf(beta,0,sqrt(tau0));   
l_1 = pi0.*normpdf(beta,0,sqrt(tau1));
temp  = min((l_1./l_0),10000000);    
gamma = binornd(1,temp./(1+temp));
pi0 = betarnd(1 + sum(gamma==1),1 + sum(gamma==0));

% 2) Update tau0 and tau1
switch lower(method)
    case 'ssvs_normal'
        % tau0 and tau1 are fixed, following Narissetti and He (2014), Annals of Statistics 42(2), 789-817.                
    case 'ssvs_student'        
        % update tau1        
        rho1 = 1 + 1/2;
        rho2 = kappa + (beta.^2)./2;
        tau1 = 1./gamrnd(rho1,1./rho2);   
        tau0 = tau1.*1e-3; %June 30 2021, added
    case 'ssvs_lasso'        
        % Update lam0 and lam1        
        lam1 = gamrnd(sum(gamma) + 1,(0.5*sum(tau1.*(gamma)) + kappa));		
        tau1 =  min(1./random('InverseGaussian',sqrt((lam1*sigma2)./(beta.^2)),lam1,p,1),1e+6);
        tau0 = tau1.*1e-3;	
    case 'sns_lasso'        
        % Update lam0 and lam1        
        lam0 = gamrnd(sum(1-gamma) + 1,(0.5*sum(tau0.*(1-gamma)) + kappa));
        lam1=0.1; %June 30 2021, added
        tau0 =  min(1./random('InverseGaussian',sqrt((lam0*sigma2)./(beta.^2)),lam0,p,1),1e+6);
        tau1 =  min(1./random('InverseGaussian',sqrt((lam1*sigma2)./(beta.^2)),lam1,p,1),1e+6);   
    case 'ssvs_horseshoe'    
        rate = 1./nu + (beta.^2)/(2*kappa*sigma2) ;
        lambda = 1./gamrnd(1,1./rate);    % random inv gamma with shape=1, rate=rate
        
        rate  = 1/xi + (1/(2*sigma2))*sum((beta.^2./lambda).*gamma);
        kappa = 1/gamrnd((sum(gamma)+1)/2, 1/rate);    % inv gamma w/ shape=(p+1)/2, rate=rate
        
        rate = 1+1./lambda;
        nu   = 1./gamrnd(1,1./rate);    % random inv gamma with shape=1, rate=1+1/lambda_sq_12
     
        rate = 1+1/kappa;
        xi   = 1/gamrnd(1,1/rate);    % inv gamma w/ shape=1, rate=1+1/tau_sq
        
        tau1 = lambda.*kappa;
        tau0 = 1e-3;
      
        
end
Q = (1-gamma).*tau0 + gamma.*tau1;      
        
end
