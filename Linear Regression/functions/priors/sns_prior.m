function [Q,gamma,tau,pi0,kappa,lambda,xi,nu] = sns_prior(y,X,sigma2,beta,gamma,tau,pi0,kappa,lambda,xi,nu,method);

p = length(beta);
sigma = sqrt(sigma2);

% SSVS/SNS class of priors
% 1) Update model probabilities
for j = randperm(p)
    theta = beta.*gamma;
    theta_star = theta; theta_star_star = theta;
    theta_star(j) = beta(j); theta_star_star(j) = 0;
    l_0       = lnormpdf(0, sum((y - X*theta_star_star).^2), sigma);
    l_1       = lnormpdf(0, sum((y - X*theta_star).^2), sigma);
    pip       = 1./( 1 + ((1-pi0).*pi0).* exp(l_0 - l_1) );
    gamma(j,1)= binornd(1,pip);
end

pi0 = betarnd(1 + sum(gamma==1),1 + sum(gamma==0));

% 2) Update tau0 and tau1
% switch lower(method)
%     case 'sns_normal'
%         
%     case 'sns_horseshoe'
%         % sample lambda_sq and nu
%         rate = beta.^2/(2*kappa) + 1./nu;
%         lambda = 1./gamrnd(1,1./rate);    % random inv gamma with shape=1, rate=rate
%         nu = 1./gamrnd(1,1./(1+1./lambda));    % random inv gamma with shape=1, rate=1+1/lambda_sq_12
%         % sample tau_sq and xi	
%         rate = 1/xi + sum(beta.^2./(2*lambda));
%         kappa = 1/gamrnd((p+1)/2, 1/rate);    % inv gamma w/ shape=(p+1)/2, rate=rate
%         xi = 1/gamrnd(1,1/(1+1/kappa));    % inv gamma w/ shape=1, rate=1+1/tau_sq
%         
%         tau = lambda.*kappa;		        
% end
Q = tau;     
        
end

