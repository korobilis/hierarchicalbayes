function [Beta,lambda,tau,sigma_sq] = horseshoe(y,X,XX,lambda,tau,sigma_sq,ieq,type)
%% Function to impelement Horseshoe shrinkage prior (http://faculty.chicagobooth.edu/nicholas.polson/research/papers/Horse.pdf)
%% in Bayesian Linear Regression. %%
%% Written by Antik Chakraborty (antik@stat.tamu.edu) and Anirban Bhattacharya (anirbanb@stat.tamu.edu)

%% Model: y=X\beta+\epslion, \epsilon \sim N(0,\sigma^2) %%
%%        \beta_j \sim N(0,\sigma^2 \lambda_j^2 \tau^2) %%
%%        \lambda_j \sim Half-Cauchy(0,1), \tau \sim Half-Cauchy (0,1) %%
%%        \pi(\sigma^2) \sim 1/\sigma^2 %%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% This function employs the algorithm proposed in "Fast Sampling with Gaussian Scale-Mixture priors in High-Dimensional Regression" by
%% Bhattacharya et. al. (2015). The global local scale parameters are updated via a Slice sampling scheme given in the online supplement 
%% of "The Bayesian Bridge" by Polson et. al. (2011). Two different algorithms are used to compute posterior samples of the p*1 vector
%% of regression coefficients \beta. Type=1 corresponds to the proposed method in Bhattacharya et. al. and Type=2 corresponds to an algorithm provided
%% in Rue (2001). We recomend our algorithm when p>n.%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input: y=response, a n*1 vector %%
%%        X=matrix of covariates, dimension n*p %%
%%        type= 1 if sampling from posterior of beta is done by Proposed Method %%
%%              2 if sampling is done by Rue's algorithm %%


%% Output: 
%%         pMean= posterior mean of Beta, a p by 1 vector%%
%%         pMeadian=posterior median of Beta, a p by 1 vector %%
%%         pLambda=posterior mean of local scale parameters, a p by 1 vector %%
%%         pSigma=posterior mean of Error variance %% 
%%         betaout=posterior samples of beta %%


[n,p]=size(X);

%% paramters %%
Beta=zeros(p,1); 

%% matrices %%
I_n=eye(n); 
l=ones(n,1);
if type==2
   Q_star=XX;
end

lambda_star=tau*lambda;
if ieq ~= -999
    lambda_star(ieq+1) = 1;
    lambda_star(1) = 20;
end
%% update beta %%
switch type

    case 1 % new method
        U=bsxfun(@times,(lambda_star.^2),X');
        %% step 1 %%
        u=normrnd(0,lambda_star);
        v=X*u + randn(n,1);
        %% step 2 %%
        v_star=(X*U+I_n)\((y/sqrt(sigma_sq))-v);
        Beta=sqrt(sigma_sq)*(u+U*v_star);
        
    case 2 % Rue   
        L=chol((1/sigma_sq)*(Q_star+diag(1./lambda_star.^2)),'lower');
        v=L\((y'*X)./sigma_sq)';
        mu=L'\v;
        u=L'\randn(p,1);
        Beta=mu+u;  
end

%% update lambda_j's in a block using slice sampling %%  
eta = 1./(lambda.^2); 
upsi = unifrnd(0,1./(1+eta));
tempps = Beta.^2/(2*sigma_sq*tau^2); 
ub = (1-upsi)./upsi;

% now sample eta from exp(tempv) truncated between 0 & upsi/(1-upsi)
Fub = 1 - exp(-tempps.*ub); % exp cdf at ub 
Fub(Fub < (1e-4)) = 1e-4;  % for numerical stability
up = unifrnd(0,Fub); 
eta = -log(1-up)./tempps; 
lambda = 1./sqrt(eta);

%% update tau %%
tempt = sum((Beta./lambda).^2)/(2*sigma_sq); 
et = 1/tau^2; 
utau = unifrnd(0,1/(1+et));
ubt = (1-utau)/utau; 
Fubt = gamcdf(ubt,(p+1)/2,1/tempt); 
Fubt = max(Fubt,1e-8); % for numerical stability
ut = unifrnd(0,Fubt); 
et = gaminv(ut,(p+1)/2,1/tempt); 
tau = 1/sqrt(et);

%% update sigma_sq %%
switch type    
    case 1 % new method
        E_1=max((y-X*Beta)'*(y-X*Beta),(1e-10)); % for numerical stability
        E_2=max(sum(Beta.^2./((tau*lambda)).^2),(1e-10));
    case 2 % Rue 
        E_1=max((y-X*Beta)'*(y-X*Beta),(1e-8));  % for numerical stability
        E_2=max(sum(Beta.^2./((tau*lambda)).^2),(1e-8));
end
sigma_sq=1/gamrnd((n + p)/2, 2/(E_1+E_2));

