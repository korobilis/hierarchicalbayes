function [y,x,beta,sigma2,disturbance] = GenRegr(n,p,options)
%===================================================================================================================
% Function to generate various regression models to be used as DGPs for
% Monte Carlo exercise
% INPUT:
%        n  Number of observations
%        p  Number of predictors
%  options  Structure with additional options
%
%  options.sparse  A 0/1 variable (0: dense regression; 1: sparse regression)
%  options.q       Proportion of p variables that are significant (if options.sparse = 1), value between 0.00 and 1.00
%  options.corr    0: no correlation; 1: spatial correlation; 2: random correlation structure
%  options.rho     Correlation coefficient (if options.corr = 1)
%  options.distr   Error distribution; 0: Gaussian, N(0,1^2); 
%                                      1: Skewed, 1/5 N(-22/25,1^2) + 1/5 N(-49/125,(3/2)^2) + 3/5 N(49/250,(5/9)^2)
%                                      2: Kurtotic, 2/3 N(0,1^2) + 1/3 N(0,(1/10)^2)
%                                      3: Outlier, 1/10 N(0,1^2) + 9/10 N(0,(1/10)^2)
%                                      4: Bimodal, 1/2 N(-3/2,(1/2)^2) + 1/2 N(3/2,(1/2)^2)
%                                      5: Trimodal, 9/20 N(-6/5,(3/5)^2) + 9/20 N(6/5,(3/5)^2) + 1/10 N(0,(1/4)^2)
%  options.hetero  Heteroskedasticity; (0: Homeskedastic error, 1: Heteroskedastic error)
%  options.sigma   Value of variance parameter
%===================================================================================================================

if nargin==0
    % If no input in the function, just use a generic sparse DGP
    n = 100;
    p = 50;
    x = randn(n,p);
    beta = round(betarnd(1,1,p,1))*(4*rand(p,1)-2);
    sigma2 = 0.1;
    y = x*beta + sqrt(sigma2)*randn(n,1);
end

% Generate error distribution (before multiplying with variance)
probs = rand(n,1);  
if options.distr == 0        % Gaussian
    disturbance = Normal(0,1^2,n,1);
elseif options.distr == 1    % Skewed
    disturbance = (probs<=1/5).*Normal(-22/25,1^2,n,1) + (probs<=2/5 & probs>1/5).*Normal(-49/125,(3/2)^2,n,1) + (probs>2/5).*Normal(49/250,(5/9)^2,n,1);
elseif options.distr == 2    % Kurtotic
    disturbance = (probs>1/3).*Normal(0,1^2,n,1) + (probs<=1/3).*Normal(0,(1/10)^2,n,1);
elseif options.distr == 3    % Outlier
    disturbance = (probs<=1/10).*Normal(0,1^2,n,1) + (probs>1/10).*Normal(0,(1/10)^2,n,1);
elseif options.distr == 4    % Bimodal
    disturbance = (probs<=1/2).*Normal(-3/2,(1/2)^2,n,1) + (probs>1/2).*Normal(3/2,(1/2)^2,n,1);
elseif options.distr == 5    % Trimodal
    disturbance = (probs<=9/20).*Normal(-6/5,(3/5)^2,n,1) + (probs>9/20 & probs<=18/20).*Normal(6/5,(3/5)^2,n,1) + (probs>18/20).*Normal(0,(1/4)^2,n,1);
else
    error('Wrong choice of options.distr');
end

% Generate predictors x
if options.corr == 0      % Uncorrelated predictors
    x = randn(n,p);
elseif options.corr == 1  % Spatially correlated predictors
    C = toeplitz(options.rho.^(0:p-1)',options.rho.^(0:p-1)');    
    x = randn(n,p)*chol(C);
elseif options.corr == 2  % Randomly correlated predictors    
    % create random correlation matrix
    temp = tril(rand(p),-1);
    A = eye(p) + temp + temp';
    C = A'*A;
    C = (diag(diag(sqrt(C)))\C)/diag(diag(sqrt(C)));    
    x = randn(n,p)*chol(C);
else
    error('Wrong choice of options.corr');
end

% Generate coefficients
if options.sparse == 0
    beta = 4*rand(p,1)-2;  % U(-2,2)
elseif options.sparse == 1
    beta = 4*rand(p,1)-2;  % U(-2,2)
    beta = beta.*binornd(1,options.q,p,1); % create sparse vector
else
    error('Wrong choice of options.sparse');
end




sigma2=(options.sigma)^2;

c=sqrt( (sigma2/(beta'*beta))*(options.R2/(1-options.R2)) );
beta=c*beta;


% Generate variance
if options.hetero == 0
    sigma2 = options.sigma;
elseif options.hetero == 1
    sigma2 = (rand + (x.^2)*rand(p,1)); 
    reweight = mean(sigma2)/options.sigma;
    sigma2 = sigma2/reweight;
else
    error('Wrong choice of options.hetero');
end

% Generate regression
y = x*beta + sqrt(sigma2).*disturbance;

end

%**************************************************************************
% Function to generate from Gaussian
function [x] = Normal(a,b,n,p)
% Note that a is the mean, and b is the variance
x = a + sqrt(b)*randn(n,p);
end
%**************************************************************************