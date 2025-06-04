% Function to demonstrate Skinny Gibbs algorithm of 
% Naveen N. Narisetty, Juan Shen & Xuming He (2018): Skinny Gibbs: A
% Consistent and Scalable Gibbs Sampler for Model Selection, Journal of the 
% American Statistical Association, DOI: 10.1080/01621459.2018.1482754
%==========================================================================
% Written by Dimitris Korobilis, University of Glasgow
% dikorobilis@googlemail.com
% This version: 14 May 2021
%==========================================================================

clear all; close all; clc;
format bank;
tic;

addpath('functions')

%----------------------Generate artificial data----------------------------
T = 200; % Set sample size
p = 2000;  % Set maximum number of predictors
x = randn(T,p);%*(tril((2*rand(p,p)-1),-1) + tril((2*rand(p,p)-1),-1)' + eye(p));
% Set the coefficients only of relevant predictors
b1_sim = 0.9;
b2_sim = 1.1;
b4_sim = 0.7;
b7_sim = 0.8;
b100_sim = -4.2;
sigma_sim = 0.5; % standard deviation of the regression
%now you are ready to simulate y
y = b1_sim*x(:,1) + b2_sim*x(:,2) + b4_sim*x(:,4) + b7_sim*x(:,7) + b100_sim*x(:,100) + sigma_sim*randn(T,1);
%--------------------------------------------------------------------------

T = size(x,1); % time series observations
p = size(x,2); % maximum nubmer of predictors

% ----------------Gibbs related preliminaries
nsave = 2000;            % Number of draws to store
nburn = 2000;            % Number of draws to discard
ntot = nsave + nburn;    % Total number of draws

beta_draws = zeros(nsave,p);   % Storate matrix for regression coefficients
gamma_draws = zeros(nsave,p);  % Storage matrix for variable selection indicators
sigma2_draws = zeros(nsave,1); % Storage matrix for regression variance

% ----------------Set priors
% Prior probability of inclusion for each predictor
pi_0 = 0.01;

% Prior variances for spike (tau_0) and slab (tau_1) components
% 1) Spike component
b0 = 1;
tau_0 = b0./T;
% 2) Slab component
fvalue = tpdf(sqrt(2.1*log(p+1)),5);
tau_1 = max(100*tau_0,pi_0*tau_0/((1 - pi_0)*fvalue))*ones(p,1);  

% Define here for computational convenience some quantities we use in each iteration of the MCMC
tau_0_inv = 1./tau_0;     tau_1_inv = 1./tau_1;
stau_0 = sqrt(tau_0);     stau_1 = sqrt(tau_1);
v_i_sqrt = sqrt(1./(T + tau_0_inv));
xx = diag(x'*x);

% Initialize parameters
beta = 0*ones(p,1); %beta_OLS;
gamma = 0*round(rand(p,1));
sigma2 = 0.1; %sigma2_OLS;

%==========================================================================
%====================| GIBBS ITERATIONS START HERE |=======================
disp('Now you are running Skinny Gibbs')
for irep = 1:ntot
    if mod(irep,100)==0
        disp(irep)
    end
    
    % 1. Update beta from Normal
    index_A = find(gamma==1);  p_A = length(index_A);
    index_I = find(gamma==0);  p_I = p - p_A; 
    beta_A = randn_gibbs(y,x(:,index_A),tau_1(index_A),sigma2,T,p_A);
    beta_I = normrnd(0,v_i_sqrt*ones(p_I,1));    
    beta(index_A) = beta_A; beta(index_I) = beta_I;
    
    % 2. Update gamma from Bernoulli
    l_0 = (1-pi_0)*normpdf(beta,0,stau_0);
    l_1 = pi_0*normpdf(beta,zeros(p,1),stau_1);
    xg = (x.*repmat((beta.*gamma)',T,1));
    sxg =  sum(xg,2);
    sse = y - sxg + xg;
    correction = sum(((x.*repmat(beta',T,1))./sigma2).*sse)' + .5*(xx.*(1 - 1./sigma2)).*(beta.^2);
    temp  = min((l_1./l_0).*exp(correction),1e+10);
    gamma = bernoullimvrnd(temp./(1+temp),p);
    
    % 3. Update sigma2 from Inverse Gamma
    c1 = p_A + T;
    PSI = (y-x(:,index_A)*beta_A)'*(y-x(:,index_A)*beta_A);
    c2 = (beta_A'/diag(tau_1(index_A)))*beta_A + PSI;%sum((beta_A.^2)./tau_1) + PSI;
    sigma2 = 1/gamrnd(c1/2,2/c2); 

    % Save draws
    if irep > nburn
        beta_draws(irep-nburn,:) = beta;
        gamma_draws(irep-nburn,:) = gamma;
        sigma2_draws(irep-nburn,:) = sigma2;
    end
end
toc;

% Now print some results
clc;
disp(['TRUE VALUE OF BETA    '   '   POSTERIOR MEAN OF BETA'])
disp('    ')
beta_true = zeros(1,p);
beta_true(1,[1 2 4 7 100]) = [b1_sim b2_sim b4_sim b7_sim b100_sim];
disp([beta_true' mean(beta_draws)'])
