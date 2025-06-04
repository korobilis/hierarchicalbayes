function [Q,invQ,miu,lambda_sq,tau_sq,xi,nu] = horseshoe_prior(beta,n,tau_sq,xi,nu)

%% Horseshoe prior
% sample lambda and nu
rate = (beta.^2)/(2*tau_sq) + 1./nu;
lambda_sq = min(1e+6,1./gamrnd(1,1./rate));    % random inv gamma with shape=1, rate=rate
nu = 1./gamrnd(1,1./(1 + 1./lambda_sq));    % random inv gamma with shape=1, rate=1+1/lambda_sq_12
% sample tau and xi	
rate = 1/xi + sum((beta.^2)./(2*lambda_sq));
tau_sq = min(1e+6,1/gamrnd((n+1)/2, 1/rate));    % inv gamma w/ shape=(p*(p-1)/2+1)/2, rate=rate
xi = 1/gamrnd(1,1/(1 + 1/tau_sq));    % inv gamma w/ shape=1, rate=1+1/tau_sq
%% update estimate of Q and Q^{-1}
Q = lambda_sq.*tau_sq;
invQ = 1./Q;
%% estimate of prior mean
miu = zeros(length(beta),1);
