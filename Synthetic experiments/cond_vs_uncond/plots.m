clear 
n = 100;
p = 1000;
%options.sparse = 1;
%options.q      = 0.1;
options.corr   = 0;options.rho=0.9;
%options.distr  = 0;
options.hetero = 0;
options.sigma  = 3;
%options.R2=0.8;
% Estimate regression
nsave=2000;
nburn=1000;

[y,X,beta_true,sigma2_true] = GenRegr_sep2021(n,p,options);
     
% Conjugate case 
conjugate=1;
sigma2_draws_conj=zeros(nsave,4);
beta_draws_conj=zeros(nsave,p,4);
[beta_draws_conj(:,:,1),sigma2_draws_conj(:,1)] = BayesRegr(y,X,2000,1000,1,conjugate);
[beta_draws_conj(:,:,2),sigma2_draws_conj(:,2)] = BayesRegr(y,X,2000,1000,2,conjugate);
[beta_draws_conj(:,:,3),sigma2_draws_conj(:,3)] = BayesRegr(y,X,2000,1000,3,conjugate);
[beta_draws_conj(:,:,4),sigma2_draws_conj(:,4)] = BayesRegr(y,X,2000,1000,4,conjugate);
figure 
for j=1:4
subplot(2,2,j)
for i=1:6 %plot first 6 slopes 
[f,xi] = ksdensity(beta_draws_conj(:,i,j)); 
plot(xi,f);
hold on 
end 
end 
title('conjugate')
figure
for j=1:4
subplot(2,2,j)
[f,xi] = ksdensity(sigma2_draws_conj(:,j)); 
plot(xi,f);
end 
title('conjugate')



% Indep case 
conjugate=0;
sigma2_draws_indep=zeros(nsave,4);
beta_draws_indep=zeros(nsave,p,4);
[beta_draws_indep(:,:,1),sigma2_draws_indep(:,1)] = BayesRegr(y,X,2000,1000,1,conjugate);
[beta_draws_indep(:,:,2),sigma2_draws_indep(:,2)] = BayesRegr(y,X,2000,1000,2,conjugate);
[beta_draws_indep(:,:,3),sigma2_draws_indep(:,3)] = BayesRegr(y,X,2000,1000,3,conjugate);
[beta_draws_indep(:,:,4),sigma2_draws_indep(:,4)] = BayesRegr(y,X,2000,1000,4,conjugate);
figure 
for j=1:4
subplot(2,2,j)
for i=1:6
[f,xi] = ksdensity(beta_draws_indep(:,i,j)); 
plot(xi,f);
hold on 
end 
end 
title('indep')

figure
for j=1:4
subplot(2,2,j)
[f,xi] = ksdensity(sigma2_draws_indep(:,j)); 
plot(xi,f);
end 
title('indep')
