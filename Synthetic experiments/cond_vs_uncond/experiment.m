clear 
rng(123)
addpath(genpath('functions'))

% Generate Normal, linear, dense, homoskedastic regression model
n = 100;
p = 300;
%options.sparse = 1;
%options.q      = 0.4;
options.corr   = 0;options.rho=0.4;
%options.distr  = 0;
options.hetero = 0;
options.sigma  = 3;
options.R2=0.4;

% Estimate regression
nsave=1000;
nburn=0;

Nsim=100;
PE_conj=zeros(4,Nsim);% Prediction error 
FN_conj=zeros(4,Nsim);
FP_conj=zeros(4,Nsim);
TP_conj=zeros(4,Nsim);

PE_indep=zeros(4,Nsim);
FN_indep=zeros(4,Nsim);
FP_indep=zeros(4,Nsim);
TP_indep=zeros(4,Nsim);

bias_indep=zeros(4,Nsim);
mse_indep=zeros(4,Nsim);
bias_conj=zeros(4,Nsim);
mse_conj=zeros(4,Nsim);

parfor sim=1:Nsim
    %sim
[y,X,beta_true,sigma2_true] = GenRegr_sep2021(n,p,options);
     
% Conjugate case 
conjugate=1;
sigma2_draws_conj=zeros(nsave,4);
beta_draws_conj=zeros(nsave,p,4);
[beta_draws_conj(:,:,1),sigma2_draws_conj(:,1)] = BayesRegr(y,X,nsave,nburn,1,conjugate);
[beta_draws_conj(:,:,2),sigma2_draws_conj(:,2)] = BayesRegr(y,X,nsave,nburn,2,conjugate);
[beta_draws_conj(:,:,3),sigma2_draws_conj(:,3)] = BayesRegr(y,X,nsave,nburn,3,conjugate);
[beta_draws_conj(:,:,4),sigma2_draws_conj(:,4)] = BayesRegr(y,X,nsave,nburn,4,conjugate);

sigma2_conj(:,sim)=mean(sigma2_draws_conj,1)';
for j=1:4
    PE_conj(j,sim)=norm( X*mean(beta_draws_conj(:,:,j),1)'-X*beta_true );
end 
% Post processing signal vs noise 
for j=1:4
res=twoMeansVS(beta_draws_conj(:,:,j)');
% False Nagative 
FN_conj(j,sim)=sum(res==0 & beta_true~=0);
% False Positive 
FP_conj(j,sim)=sum(res~=0 & beta_true==0);
% True Positive 
TP_conj(j,sim)=sum(res~=0 & beta_true~=0);
bias_conj(j,sim)= mean( abs( mean(beta_draws_conj(:,1:6,j),1)'-beta_true(1:6)) );
mse_conj(j,sim)= mean( ( mean(beta_draws_conj(:,1:6,j),1)'-beta_true(1:6)).^2 );
bias_conj_all(j,sim)= mean( abs( mean(beta_draws_conj(:,:,j),1)'-beta_true) );
mse_conj_all(j,sim)= mean( ( mean(beta_draws_conj(:,:,j),1)'-beta_true).^2 );
bias_conj_noise(j,sim)= mean( abs( mean(beta_draws_conj(:,7:end,j),1)'-beta_true(7:end)) );
mse_conj_noise(j,sim)= mean( ( mean(beta_draws_conj(:,7:end,j),1)'-beta_true(7:end)).^2 );
end 




% Indep case 
conjugate=0;
sigma2_draws_indep=zeros(nsave,4);
beta_draws_indep=zeros(nsave,p,4);
[beta_draws_indep(:,:,1),sigma2_draws_indep(:,1)] = BayesRegr(y,X,nsave,nburn,1,conjugate);
[beta_draws_indep(:,:,2),sigma2_draws_indep(:,2)] = BayesRegr(y,X,nsave,nburn,2,conjugate);
[beta_draws_indep(:,:,3),sigma2_draws_indep(:,3)] = BayesRegr(y,X,nsave,nburn,3,conjugate);
[beta_draws_indep(:,:,4),sigma2_draws_indep(:,4)] = BayesRegr(y,X,nsave,nburn,4,conjugate);

sigma2_indep(:,sim)=mean(sigma2_draws_indep,1)';
for j=1:4
    PE_indep(j,sim)=norm( X*mean(beta_draws_indep(:,:,j),1)'-X*beta_true );
end 
% Post processing signal vs noise 
for j=1:4
res=twoMeansVS(beta_draws_indep(:,:,j)');
% False Nagative 
FN_indep(j,sim)=sum(res==0 & beta_true~=0);
% False Positive 
FP_indep(j,sim)=sum(res~=0 & beta_true==0);
% True Positive 
TP_indep(j,sim)=sum(res~=0 & beta_true~=0);

bias_indep(j,sim)= mean( abs( mean(beta_draws_indep(:,1:6,j),1)'-beta_true(1:6)) );
mse_indep(j,sim)= mean( ( mean(beta_draws_indep(:,1:6,j),1)'-beta_true(1:6)).^2 );

bias_indep_all(j,sim)= mean( abs( mean(beta_draws_indep(:,:,j),1)'-beta_true) );
mse_indep_all(j,sim)= mean( ( mean(beta_draws_indep(:,:,j),1)'-beta_true).^2 );

bias_indep_noise(j,sim)= mean( abs( mean(beta_draws_indep(:,7:end,j),1)'-beta_true(7:end)) );
mse_indep_noise(j,sim)= mean( ( mean(beta_draws_indep(:,7:end,j),1)'-beta_true(7:end)).^2 );
end 

end 

fprintf('sigma2')
[mean(sigma2_conj,2),mean(sigma2_indep,2)]
fprintf('Bias')
[mean(bias_conj,2),mean(bias_indep,2)]
fprintf('MSE')
[mean(mse_conj,2),mean(mse_indep,2)]
fprintf('False Nagative')
[mean(FN_conj,2),mean(FN_indep,2)]
fprintf('False Positive')
[mean(FP_conj,2),mean(FP_indep,2)]
fprintf('True Positive')
[mean(TP_conj,2),mean(TP_indep,2)]
fprintf('Prediction Error')
[mean(PE_conj,2),mean(PE_indep,2)]
% fprintf('Bias-all')
% [mean(bias_conj_all,2),mean(bias_indep_all,2)]
% fprintf('MSE-all')
% [mean(mse_conj_all,2),mean(mse_indep_all,2)]
% fprintf('Bias-noise')
% [mean(bias_conj_noise,2),mean(bias_indep_noise,2)]
% fprintf('MSE-noise')
% [mean(mse_conj_noise,2),mean(mse_indep_noise,2)]
