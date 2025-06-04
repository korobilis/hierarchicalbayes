function [Beta,lambda,tau,nu,xi,sigma_sq,h,sig] = horseshoeSV(y,X,XX,lambda,tau,nu,xi,h,sig,ieq,type)
%% Function to impelement Horseshoe shrinkage prior in Bayesian Linear Regression with stochastic volatiliy.
%% Unlike function horseshoe.m this prior is independent of \sigma^{2}, since this parameter is now time-varying.

[n,p]=size(X);

%% paramters %%
Beta=zeros(p,1); 

%% matrices %%
I_n=eye(n); 
l=ones(n,1);

lambda_star=tau*lambda;
if ieq ~= -999
    lambda_star(ieq+1) = 1;
    lambda_star(1) = 10;
end
%% update beta %%
switch type
    case 1 % new method
        U=bsxfun(@times,(lambda_star),X');
        %% step 1 %%
        u=normrnd(0,lambda_star);
        v=X*u + randn(n,1);
        %% step 2 %%
        v_star=(X*U+I_n)\(y-v);
        Beta=(u+U*v_star);
        
    case 2 % Rue
        Q_star=XX;
        L=chol((Q_star + diag(1./lambda_star)),'lower');
        v=L\(y'*X)';
        mu=L'\v;
        u=L'\randn(p,1);
        Beta=mu+u;
end

% % sample lambda and nu
% rate = (Beta.^2)/(2*tau) + 1./nu;
% lambda = min(1e+6,1./gamrnd(1,1./rate));    % random inv gamma with shape=1, rate=rate
% nu = 1./gamrnd(1,1./(1 + 1./lambda));    % random inv gamma with shape=1, rate=1+1/lambda_sq_12
% % sample tau and xi	
% rate = 1/xi + sum((Beta.^2)./(2*lambda));
% tau = min(1e+6,1/gamrnd((n+1)/2, 1/rate));    % inv gamma w/ shape=(p*(p-1)/2+1)/2, rate=rate
% xi = 1/gamrnd(1,1/(1 + 1/tau));    % inv gamma w/ shape=1, rate=1+1/tau_sq


%% update lambda_j's in a block using slice sampling %%  
eta = 1./(lambda.^2); 
upsi = unifrnd(0,1./(1+eta));
tempps = Beta.^2/(2*tau^2); 
ub = (1-upsi)./upsi;

% now sample eta from exp(tempv) truncated between 0 & upsi/(1-upsi)
Fub = 1 - exp(-tempps.*ub); % exp cdf at ub 
Fub(Fub < (1e-4)) = 1e-4;  % for numerical stability
up = unifrnd(0,Fub); 
eta = -log(1-up)./tempps; 
lambda = 1./sqrt(eta);

%% update tau %%
tempt = sum((Beta./lambda).^2)/2; 
et = 1/tau^2; 
utau = unifrnd(0,1/(1+et));
ubt = (1-utau)/utau; 
Fubt = gamcdf(ubt,(p+1)/2,1/tempt); 
Fubt = max(Fubt,1e-8); % for numerical stability
ut = unifrnd(0,Fubt); 
et = gaminv(ut,(p+1)/2,1/tempt); 
tau = 1/sqrt(et);

%% update sigma_sq %%
yhat = y-X*Beta;
ystar  = log(yhat.^2 + 1e-6);        
[h, ~] = SVRW(ystar,h,sig,4);  % log stochastic volatility
sigma_sq  = exp(h);
r1 = 0.1 + n - 1;   r2 = 0.1 + sum(diff(h).^2)';
sig = 1./gamrnd(r1./2,2./r2);   % sample state variance of log(sigma_t.^2)


