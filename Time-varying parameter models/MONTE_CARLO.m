%% Fast estimation of TVP regression model using the Gibbs sampler
% ********************************************************************************
% This code estimates a time-varying parameter regression of the form
%               y(t) = x(t) b(t) + e(t)
%               b(t) = b(t-1)    + u(t)
% where e(t) ~ N(0,s2(t)) and u(t) ~ N(0,Q). Estimation is achieved by applying 
% the following transformation:
%                  y = Z Db      + e
%                 Db =             u
% where Db =[b(1)',Db(2)',...,Db(T)']', with D the diff operator, and Z is the matrix
%                _                                    _
%               |  x(1)    0      0    ...   0     0   |
%               |  x(2)   x(2)    0    ...   0     0   |
%          Z  = |  ...    ...    ...   ...  ...   ...  |.
%               | x(T-1) x(T-1) x(T-1) ... x(T-1)  0   |
%               |  x(T)   x(T)   x(T)  ...  x(T)  x(T) |
%                -                                    -
% This model estimates equation y = Z Db + e, treating Db = u as the
% following prior
%                     Db ~ N(0,Q)
% and we subsequently allow Q to be diagonal and have various hyperprior
% distributions.
% ********************************************************************************
% Written by Dimitris Korobilis, 12 November 2019
% ********************************************************************************

clear all;
close all;
clc;

addpath('functions')
addpath('MODELS')
%-------------------------------PRELIMINARIES--------------------------------------
nMC  = 1;                 % Number of DGPs to simulate

ngibbs     = 1000;         % Number of Gibbs sampler iterations
nburn      = 0.2*ngibbs;    % Number of iterations to discard
nthin      = 10;            % Thin to reduce correlation in Gibbs chain

prior = 1;            % 1: Normal-iGamma prior (Student t)
                      % 2: SSVS with Normal(0,tau_0) and Normal-iGamma components
                      
T = 200;              % Number of time series observations
p = 4;                % Number of predictors
rho = 0;              % Correlation among predictors
%----------------------------- END OF PRELIMINARIES --------------------------------
tic;
%% ----------------------------------PREP DATA----------------------------------------   
s = ones(T,p); s(round(T/3):end,1)=0; s(round(T/2):end,3)=0; s(1:round(T/4),4)=0; if p>4; s(:,5:end)=0;end
theta0 = 4.*rand(1,p)-2;  mu = s(1,1).*theta0;
param.s = s; param.theta0=theta0; param.sigma0 = 0.2; param.rho = rho; param.mu = mu; 
 
%% Start Monte Carlo
BETA   = zeros(T,p,nMC,5);    SIGMA  = zeros(T,nMC,5);
DIAGNB = zeros(T*p,8,nMC,4);  DIAGNS = zeros(T,8,nMC,4);
IFB    = zeros(T*p,nMC,4);    IFS    = zeros(T,nMC,4);
for iMC = 1:nMC
    disp(['This is iteration ' num2str(iMC)]);
    % Generate artificial data
    [y,X,s_sim,theta_sim,beta_sim,sigma_sim] = sptvpsv_reg_dgp(T,p,param);
    
    % 1) KR1 algorithm
    tic;
    [beta_save1,sigma_save1] = KR1(y,X,ngibbs,nburn,prior);
    time_kr1 = toc;
    % 2) KR2 algorithm
    tic;
    [beta_save2,sigma_save2] = KR2(y,X,ngibbs,nburn,prior,1);
    time_kr2 = toc;
    % 3) CJ algorithm
    tic;
    [beta_save3,sigma_save3] = CJ(y,X,ngibbs,nburn,prior);
    time_cj = toc;
    % 4) CK algorithm
    tic;
    [beta_save4,sigma_save4] = CK(y,X,ngibbs,nburn,prior);
    time_ck = toc;
    % 6) Tibshirani Lasso
    [beta_save5,sigma_save5] = tvp_lasso(y,X);
    
    TIMES =  [time_kr1; time_kr2; time_cj; time_ck].*1000;

    % Do thinning
    beta_save1 = beta_save1(1:nthin:end,:,:);   sigma_save1 = sigma_save1(1:nthin:end,:);
    beta_save2 = beta_save2(1:nthin:end,:,:);   sigma_save2 = sigma_save2(1:nthin:end,:);
    beta_save3 = beta_save3(1:nthin:end,:,:);   sigma_save3 = sigma_save3(1:nthin:end,:);
    beta_save4 = beta_save4(1:nthin:end,:,:);   sigma_save4 = sigma_save4(1:nthin:end,:);
    
    % Save all coefficients
    BETA(:,:,iMC,1) = beta_sim;           SIGMA(:,iMC,1) = sigma_sim;
    BETA(:,:,iMC,2) = mean(beta_save1);   SIGMA(:,iMC,2) = mean(sigma_save1);
    BETA(:,:,iMC,3) = mean(beta_save2);   SIGMA(:,iMC,3) = mean(sigma_save2);
    BETA(:,:,iMC,4) = mean(beta_save3);   SIGMA(:,iMC,4) = mean(sigma_save3);
    BETA(:,:,iMC,5) = mean(beta_save4);   SIGMA(:,iMC,5) = mean(sigma_save4);
    BETA(:,:,iMC,6) = beta_save5;         SIGMA(:,iMC,6) = sigma_save5;
    
    DIAGNB(:,:,iMC,1) = momentg(beta_save1(:,:));  DIAGNS(:,:,iMC,1) = momentg(sigma_save1);
    DIAGNB(:,:,iMC,2) = momentg(beta_save2(:,:));  DIAGNS(:,:,iMC,2) = momentg(sigma_save2);
    DIAGNB(:,:,iMC,3) = momentg(beta_save3(:,:));  DIAGNS(:,:,iMC,3) = momentg(sigma_save3);
    DIAGNB(:,:,iMC,4) = momentg(beta_save4(:,:));  DIAGNS(:,:,iMC,4) = momentg(sigma_save4);

    IFB(:,iMC,1) = 1./DIAGNB(:,6,iMC,1);   IFS(:,iMC,1) = 1./DIAGNS(:,6,iMC,1);
    IFB(:,iMC,2) = 1./DIAGNB(:,6,iMC,2);   IFS(:,iMC,2) = 1./DIAGNS(:,6,iMC,2);
    IFB(:,iMC,3) = 1./DIAGNB(:,6,iMC,3);   IFS(:,iMC,3) = 1./DIAGNS(:,6,iMC,3);
    IFB(:,iMC,4) = 1./DIAGNB(:,6,iMC,4);   IFS(:,iMC,4) = 1./DIAGNS(:,6,iMC,4);    
end
IFB = squeeze(mean(IFB,2));
IFS = squeeze(mean(IFS,2));

models = {'KR1';'KR2';'CJ09';'CK94'};

% Plot some convergence diagnostics here
figure;
for i = 1:4
    subplot(2,2,i)
    plot(IFB(:,i),'LineWidth',2);
    grid on;
    legend(models(i));
end
suptitle('Average (over MC iterations) inefficiency factors for parameters \beta_{t}')

figure;
for i = 1:4
    subplot(2,2,i)
    plot(IFS(:,i),'LineWidth',2);
    grid on;
    legend(models(i));
end
suptitle('Average (over MC iterations) inefficiency factors for parameter \sigma_{t}^{2}')


figure; for i=1:4; subplot(2,2,i); plot([squeeze(BETA(:,i,1,1)), squeeze(BETA(:,i,1,2)), squeeze(BETA(:,i,1,6))],'LineWidth',2); legend({'True','KR1','LASSO'}); end

%save(sprintf('%s_%g_%g_%g.mat','MONTE_CARLO',T,p,rho),'-mat');