%MONTE_CARLO_regression.m    Monte Carlo comparison of Bayesian estimators
%                            for regressions with high-dimensional predictors
%
% Based on Korobilis, Shimizu, Trinh (2021) 
% Bayesian penalized regression methods for high-dimensional econometric inference
%==========================================================================
% Written by Dimitris Korobilis
% Modified by Kenichi Shimizu
% University of Glasgow
% This version: 11 July 2021
%==========================================================================

clear all; clc; close all;
rng(123)
addpath(genpath('functions'))

% Generate Normal, linear, dense, homoskedastic regression model
n = 500;
p = 5;
options.sparse = 1;
options.q      = 0.1;
options.corr   = 0;
options.distr  = 0;
options.hetero = 0;
options.sigma  = 0.1;

[y,X,beta_true,sigma2_true] = GenRegr(n,p,options);

% Estimate regression
[beta_draws1,sigma2_draws1] = BayesRegr(y,X,2000,1000,1);
[beta_draws2,sigma2_draws2] = BayesRegr(y,X,2000,1000,2);
[beta_draws3,sigma2_draws3] = BayesRegr(y,X,2000,1000,3);
[beta_draws4,sigma2_draws4] = BayesRegr(y,X,2000,1000,4);


[beta_draws5,sigma2_draws5,gamma_draws5,pi5] = BayesRegr(y,X,2000,1000,5);
[beta_draws6,sigma2_draws6,gamma_draws6,pi6] = BayesRegr(y,X,2000,1000,6);
[beta_draws7,sigma2_draws7,gamma_draws7,pi7] = BayesRegr(y,X,2000,1000,7);
[beta_draws8,sigma2_draws8,gamma_draws8,pi8] = BayesRegr(y,X,2000,1000,8);
[beta_draws9,sigma2_draws9,gamma_draws9,pi9] = BayesRegr(y,X,2000,1000,9);
[beta_draws10,sigma2_draws10,gamma_draws10,pi10] = BayesRegr(y,X,2000,1000,10);

% print posterior means vs true coefficients
b1 = squeeze(mean(beta_draws1))';
b2 = squeeze(mean(beta_draws2))';
b3 = squeeze(mean(beta_draws3))';
b4 = squeeze(mean(beta_draws4))';
b5 = squeeze(mean(beta_draws5))';
b6 = squeeze(mean(beta_draws6))';
b7 = squeeze(mean(beta_draws7))';
b8 = squeeze(mean(beta_draws8))';
b9 = squeeze(mean(beta_draws9))';
b10 = squeeze(mean(beta_draws10))';

disp([beta_true b1 b2 b3 b4 b5 b6 b7 b8 b9 b10])

sigma21 = squeeze(mean(sigma2_draws1))';
sigma22 = squeeze(mean(sigma2_draws2))';
sigma23 = squeeze(mean(sigma2_draws3))';
sigma24 = squeeze(mean(sigma2_draws4))';
sigma25 = squeeze(mean(sigma2_draws5))';
sigma26 = squeeze(mean(sigma2_draws6))';
sigma27 = squeeze(mean(sigma2_draws7))';
sigma28 = squeeze(mean(sigma2_draws8))';
sigma29 = squeeze(mean(sigma2_draws9))';
sigma210 = squeeze(mean(sigma2_draws10))';

disp([sigma2_true sigma21 sigma22 sigma23 sigma24 sigma25 sigma26 sigma27 sigma28 sigma29 sigma210])

gamma5 = squeeze(mean(gamma_draws5))';
gamma6 = squeeze(mean(gamma_draws6))';
gamma7 = squeeze(mean(gamma_draws7))';
gamma8 = squeeze(mean(gamma_draws8))';
gamma9 = squeeze(mean(gamma_draws9))';
gamma10 = squeeze(mean(gamma_draws10))';


disp([ (beta_true~=0) NaN*ones(p,1) NaN*ones(p,1) NaN*ones(p,1) NaN*ones(p,1) gamma5 gamma6 gamma7 gamma8 gamma9 gamma10])





