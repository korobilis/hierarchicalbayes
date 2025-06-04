function [Q,invQ,miu,lambda,tau] = horseshoe_prior(beta,p,lambda,tau)

%% Horseshoe prior
%% update lambda_j's in a block using slice sampling %%  
eta = 1./(lambda.^2); 
upsi = unifrnd(0,1./(1+eta));
tempps = beta.^2/(2*tau^2); 
ub = (1-upsi)./upsi;

% now sample eta from exp(tempv) truncated between 0 & upsi/(1-upsi)
Fub = 1 - exp(-tempps.*ub); % exp cdf at ub 
Fub(Fub < (1e-4)) = 1e-4;  % for numerical stability
up = unifrnd(0,Fub); 
eta = -log(1-up)./tempps; 
lambda = 1./sqrt(eta);

%% update tau %%
tempt = sum((beta./lambda).^2)/2; 
et = 1/tau^2; 
utau = unifrnd(0,1/(1+et));
ubt = (1-utau)/utau; 
Fubt = gamcdf(ubt,(p+1)/2,1/tempt); 
Fubt = max(Fubt,1e-8); % for numerical stability
ut = unifrnd(0,Fubt); 
et = gaminv(ut,(p+1)/2,1/tempt); 
tau = 1/sqrt(et);

%% update estimate of Q and Q^{-1}
Q = (lambda.*tau).^2;
invQ = 1./Q;

%% estimate of prior mean
miu = zeros(length(beta),1);
