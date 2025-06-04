
% Last modified: Jun 2, 2021
% Kenichi Shimizu: University of Glasgow 

% Run MCMC of Antonelli et al. (2019, Bayesian Analysis) 


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

function [BetaAllPost,GammaPost,Sigma2Post,ThetaPost,Tau2Post,Lambda0Post]=...
    SnSTE(nsave,nburn,thin,...
    p,y,Xall,...
    Lambda0,Lambda1,...
    w,EmpiricalBayes)

n=length(y);
BetaAllPost=zeros(p+2,nsave);
GammaPost=zeros(p,nsave);
Sigma2Post=zeros(1,nsave);
ThetaPost=zeros(1,nsave);
Tau2Post=zeros(p,nsave);
Lambda0Post=zeros(1,nsave);

BetaAllPost(:,1)=0.1*randn(p+2,1);
GammaPost(:,1)=zeros(p,1);
Sigma2Post(1)=1;
ThetaPost(1)=0.02;
Tau2Post(:,1)=gamrnd(1,1,p,1);
Lambda0Post(1)= Lambda0;


a=1;    % Fixed as in the paper
b=0.2*p;% Fixed as in the paper
c=0.001;% Fixed as in the paper
d=0.001;% Fixed as in the paper
K=10000; % Fixed as in the paper
accTheta = 0;


iter = 100;             % Print every "iter" iteration
%fprintf('Iteration 0000')
for irep = 2:(nsave+nburn)
    % Print every "iter" iterations on the screen
    if mod(irep,iter) == 0
        fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%4d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',irep);
    end    
    
Dinv=diag(1./[K;K;Tau2Post(:,irep-1)]);
   
% Update sigma2 
sse=sum( (y-Xall*BetaAllPost(:,irep-1)).^2 );
Sigma2Post(irep)=1/gamrnd( c+0.5*(n-1)+0.5*(p+2), 1/(d + 0.5*sse + 0.5*BetaAllPost(:,irep-1)'*Dinv*BetaAllPost(:,irep-1)) );

% Update theta 

% Proposal density: q(\theta1 | \theta2 )  = unif( \theta1 ; \theta2-0.02, \theta2+0.02 )

BoundaryLow2 = max(0, ThetaPost(irep-1) - 0.02);
BoundaryAbove2 = min(1, ThetaPost(irep-1) + 0.02);
ThetaNew=BoundaryLow2 + rand*(BoundaryAbove2 - BoundaryLow2);
BoundaryLow1 = max(0, ThetaNew - 0.02);
BoundaryAbove1 = min(1, ThetaNew + 0.02);
    
logAR=...
getLogTheta(ThetaNew,           a, b, w, GammaPost(:,irep-1))...
+log( unifpdf(ThetaPost(irep-1),BoundaryLow1, BoundaryAbove1) )...
-getLogTheta(ThetaPost(irep-1), a, b, w, GammaPost(:,irep-1))...
-log( unifpdf(ThetaNew,         BoundaryLow2, BoundaryAbove2) );

if logAR > log(rand)
      ThetaPost(irep) = ThetaNew;
      accTheta = accTheta + 1;
else 
      ThetaPost(irep)  = ThetaPost(irep-1);
end

% Update BetaAll
u = sqrt(1./(Sigma2Post(irep)));
ynew = y.*u;
Xallnew = Xall.*u;  
BetaAllPost(:,irep) = randn_gibbs(ynew,Xallnew,[K;K;Tau2Post(:,irep-1)],n,p+2);
    
% 
% if EmpiricalBayes==0
%  Lambda0=Lambda0;
% elseif EmpiricalBayes==1
%  Lambda0=Lambda0Post(irep-1); 
% end 

% Update gamma 
lam1_div_sigma=Lambda1/sqrt(Sigma2Post(irep));
lam0_div_sigma=Lambda0Post(irep-1)/sqrt(Sigma2Post(irep));
% RandOrder=randsample(1:p,p);
%     for ii=1:p
%     j=RandOrder(ii);  

    l1=lam1_div_sigma*( ThetaPost(irep-1).^w ).*exp(-lam1_div_sigma*abs( BetaAllPost(3:end,irep) ) ) ;
    l0=lam0_div_sigma*( (1-ThetaPost(irep-1)).^w ).*exp(-lam0_div_sigma*abs( BetaAllPost(3:end,irep) ));
    tempProb=1./(1+l0./l1);
    GammaPost(:,irep)=(rand(p,1)<tempProb)';    

% 
%     for j=1:p
%     tempProb=lam1_div_sigma*( ThetaPost(irep-1)^w(j) )*exp(-lam1_div_sigma*abs( BetaAllPost(j+2,irep) ) ) /...
%             (lam1_div_sigma*( ThetaPost(irep-1)^w(j) )*exp(-lam1_div_sigma*abs( BetaAllPost(j+2,irep)) )+...
%             lam0_div_sigma*( (1-ThetaPost(irep-1))^w(j) )*exp(-lam0_div_sigma*abs( BetaAllPost(j+2,irep) ) ) );
%     GammaPost(j,irep)=binornd(1,tempProb);
%     end     
    
% Update tau 
% RandOrder=randsample(1:p,p);
%     for ii=1:p
%     j=RandOrder(ii); 

    tempLambda =GammaPost(:,irep)*Lambda1 + (1-GammaPost(:,irep))*Lambda0Post(irep-1);
    lambdaPrime = tempLambda.^2; 
    muPrime = sqrt(Sigma2Post(irep)*lambdaPrime./ BetaAllPost(3:end,irep).^2);
    Tau2Post(:,irep)=   1./random('InverseGaussian',muPrime,lambdaPrime,p,1)';
    
% 
%     for j=1:p
%     tempLambda =GammaPost(j,irep)*Lambda1 + (1-GammaPost(j,irep))*Lambda0Post(irep-1);
%     lambdaPrime = tempLambda^2; 
%     muPrime = sqrt(lambdaPrime * Sigma2Post(irep) / BetaAllPost(j+2,irep)^2);
%     Tau2Post(j,irep)= 1/random('InverseGaussian',muPrime,lambdaPrime,1,1);
%     end
%     
    
if EmpiricalBayes==0
    Lambda0Post(irep) =  Lambda0;
else 
% Update lambda0 as in Section 3.1
    if mod(irep,50)==0 && irep>500 %Take averages of 501:550, 551:600, etc
      wut1=sum( GammaPost(:,irep-49:irep) );
      wut2=mean( Tau2Post(:,irep-49:irep).*( GammaPost(:,irep-49:irep)==0 ),2 );
      Lambda0Post(irep) = sqrt( 2*(p - mean(wut1)) / sum(wut2) );
    else 
      Lambda0Post(irep) =  Lambda0Post(irep-1);
%       if irep > 7000 
%         sign1 = sign(Lambda0Post(irep) - Lambda0Post(1));
%         sign2 = sign(Lambda0Post(irep) - Lambda0Post(irep-6000));
%         if sign1 ~= sign2 
%             break
%         end 
%       end 
    end    
    
end  


end 


keep=nburn+1:thin:nburn+nsave;

BetaAllPost=BetaAllPost(:,keep);
GammaPost=GammaPost(:,keep);
Sigma2Post=Sigma2Post(keep);
ThetaPost=ThetaPost(keep);
Tau2Post=Tau2Post(:,keep);
Lambda0Post=Lambda0Post(keep);

end 
