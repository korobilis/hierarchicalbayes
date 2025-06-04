% Code written for 
% Bayesian Approaches to Shrinkage and Sparse Estimation: A guide for applied econometricians

% The code estimates a hierarchical Bayes version of
% Antonelli, Parmigiani, and Dominici (2019). 
% High-dimensional confounding adjustment using continuous spike and slab priors. Bayesian Analysis


%==========================================================================
% Written  by Kenichi Shimizu 
% University of Glasgow
% This version: November 3, 2021
%==========================================================================


clear
rng('default')
addpath( genpath('BayesMethods') );
addpath( genpath('FreqMethods') );
%     
n=100;p=200;
options.corr=0;% 0 or 1
options.htrs=0;% 0 or 1

options.q=0.04; % 0.04 or 0.4 % q=0.04 is more or less what Antonelli etal do

options.R2_t=0.8;
options.R2_y=0.8;


nsave=5000;nburn=0;thin=1;% For Bayes methods

% Generate data 
[y,T,X]=genData(n,p,options);

% Naive Outcome LASSO and Post Selection LASSO
[beta_naive,beta_postselect]=NaiveOutcomeLasso(y,X,T);
EstTE_NaiveLasso=beta_naive(2);
EstTE_PostLasso=beta_postselect(2);
Xincluded_NiaveLasso=(beta_naive(3:end)~=0);

% Double Post Selection of Belloni etal (2014, ReStu) 
[BetaAll_DoublePostSelection,se_TE]=DoublePostSelection(y,[ones(n,1),T,X]);
EstTE_DoublePostSelection=BetaAll_DoublePostSelection(2);
LbTE_DoublePostSelection=EstTE_DoublePostSelection-1.96*se_TE;
UbTE_DoublePostSelection=EstTE_DoublePostSelection+1.96*se_TE;
Xincluded_DoublePostSelection=(BetaAll_DoublePostSelection(3:end)~=0);

% Naive Bayesian regulalizations 
prior=3;
[BetaAllPost,GammaPost,Sigma2Post]=NaiveBayes(y,X,T,nsave,nburn,thin,prior);
EstTE_NaiveBayes=mean( BetaAllPost(2,:) );
LbTE_NaiveBayes=quantile( BetaAllPost(2,:),0.025 );
UbTE_NaiveBayes=quantile( BetaAllPost(2,:),0.975 );
Xincluded_NaiveBayes=mean(GammaPost,2);

% High-dimensional cofounding adjustment of Antonelli et al (2019)
% under various spike-and-slab priors 
activeX=HDCofoundingLasso(T,X);
prior=3;
[BetaAllPost,GammaPost,Sigma2Post,ThetaPost,wj_final]=HDCofounding(y,X,T,activeX,nsave,nburn,thin,prior);
EstTE_HDCofounding=mean( BetaAllPost(2,:) );
LbTE_HDCofounding=quantile( BetaAllPost(2,:),0.025 );
UbTE_HDCofounding=quantile( BetaAllPost(2,:),0.975 );
Xincluded_HDCofounding=mean(GammaPost,2);
 


