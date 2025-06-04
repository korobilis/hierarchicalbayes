% i=1:n
% y_i = beta0 + betat*T_i + X_i'*beta + N(0,sigma2)
% X_i is p by 1 
% T_i is a scalar treatment variable 
% betat is the treatment effect 
% gamma is p by 1 "inclusion indicator"
% theta is a scalar "inclusion probability" 
% w is p by 1 "weights" (needs to be estimated via EB)
% lambda0 (needs to be estimated via EB)
% lambda1=0.1

% theta ~ Beta(a,b)
% sigma2 ~ InvGamma(c,d)
% beta0, betat ~ Normal(0,K) 

% BetaAll = [beta0; betat; beta] is a (p+2) by 1 vector 
% X =[X_1' ; ... ; X_n'] is a n by p matrix 
% T =[T_1,...,T_n]; is a n by 1 vector 
% Xall =[ones(n,1),T,X] is a n by (p+2) matrix 

% tau2 is p by 1 
% Dtau = diag(K,K,tau2_1,...,tau2_p) 


function [BetaAllPost,GammaPost,Sigma2Post,ThetaPost,Tau2Post,Lambda0Post,...
    wj_final,Lambda0Est]=...
    Antonelli2019(y,X,T,activeX,...
    nsave,nburn,thin)
[n,p]=size(X);
Xall =[ones(n,1),T,X];

% Estimation starts here 
%%%%%%% 
Lambda1=0.1;% Fixed as in the paper
w=ones(p,1);%Initial value of weights



% % 0. Run LASSO 
% if T_type=='binary'
% % using 10-fold cross-validation 
% [B,FitInfo] = lassoglm(X,T,'binomial','CV',10);   
% elseif  T_type=='continuous'
% % using 10-fold cross-validation 
% [B,FitInfo] = lasso(X,T,'CV',10);
% end 
% % the sparsest model within one standard error of the minimum MSE
% idxLambda1SE = FitInfo.Index1SE;
% activeX = (B(:,idxLambda1SE)~=0);



% 1. Run Empirical Bayes to calculate weights 
Lambda0start=10;%Initial value of lambda0
%nsave=1000;nburn=0;thin=1;
fprintf('\nRunning initial empirical Bayes estimates to calculate weights: ');
tic;
[BetaAllPost,GammaPost,Sigma2Post,ThetaPost,Tau2Post,Lambda0Post]=...
    SnSTE(nsave,nburn,thin,...
    p,y,Xall,...
    Lambda0start,Lambda1,...
    w,1);
toc;
% figure; 
% subplot(1,2,1);plot(1:length(Lambda0Post),Lambda0Post);title('\lambda_0')
% subplot(1,2,2);plot(1:length(ThetaPost),ThetaPost);title('\theta')

%keep=nsave*0.3:nsave;
ThetaEst=mean(ThetaPost);
Lambda0Est=mean(Lambda0Post);
fprintf(['\nLambda0=',num2str(Lambda0Est)]);

% Choose wj_final as in Section 2.4
wj_vec=0:1/(2000-1):1;
pStar=zeros(length(wj_vec),1);
for ii=1:length(wj_vec)
pStar(ii) = getPstar(0, ThetaEst^wj_vec(ii), Lambda1, Lambda0Est);
end 
%figure;plot(wj_vec,pStar);xlabel('wj');
% The smallest value of wj such that inclusion prob for beta_j=0 is less
% than 0.1
wj_final = wj_vec( find(pStar<0.1,1) );
fprintf(['\nw final=',num2str(wj_final)]);


if sum(activeX)==0
w= ones(p,1);
else 
w= ones(p,1);
w(activeX) = wj_final;
end 
    
% 2. Now estimate lambda0 conditional on the weights
Lambda0start=Lambda0Est;
%nsave=1000;nburn=0;thin=1;

fprintf('\nComputing empirical Bayes estimates of Lambda0 conditional on weights: ');
tic;
[BetaAllPost,GammaPost,Sigma2Post,ThetaPost,Tau2Post,Lambda0Post]=...
    SnSTE(nsave,nburn,thin,...
    p,y,Xall,...
    Lambda0start,Lambda1,...
    w,1);
toc;
% figure;plot(1:length(Lambda0Post),Lambda0Post);title('\lambda_0')
Lambda0Est=mean(Lambda0Post);
fprintf(['\nLambda0=',num2str(Lambda0Est)]);

% 3. Do final analysis 
fprintf('\nRunning final analysis now: ')
%nsave=1000;nburn=1000;thin=10;
tic;
[BetaAllPost,GammaPost,Sigma2Post,ThetaPost,Tau2Post,Lambda0Post]=...
    SnSTE(nsave,nburn,thin,...
    p,y,Xall,...
    Lambda0Est,Lambda1,...
    w,0);
toc;

end 


function pStar = getPstar(beta, theta, lambda1, lambda0) 
  part1 = theta*lambda1*exp(-lambda1*abs(beta));
  part0 = (1-theta)*lambda0*exp(-lambda0*abs(beta));
  pStar=(part1 / (part1 + part0));
end 
