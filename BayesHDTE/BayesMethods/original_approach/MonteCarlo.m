clear
% MC experiment on methods 
% 1. Antonelli 
% 2. Naive outcome Lasso
% 3. Post selection Lasso
% 4. Belloni Double Post Selection
% 5. Usual OLS 

%rng(123)





pc = parcluster('local');       % get parallel cluster object
delete(pc.Jobs);
parpool(pc,20)

% Addpath Penalized ToolBox for Naive Outcome LASSO 
addpath('PenalizedToolBox');
addpath('PenalizedToolBox/models');addpath('PenalizedToolBox/internals');addpath('PenalizedToolBox/penalties')
addpath('PenalizedToolBox/helpfiles');addpath('PenalizedToolBox/jss');addpath('PenalizedToolBox/jss/jsstable')

n=100;p=200;
sigmaX=0.3;
Nsim=60;
EstTE_Bayes=zeros(1,Nsim);
EstTE_OLS=zeros(1,Nsim);
EstTE_NaiveLasso=zeros(1,Nsim);
EstTE_PostLasso=zeros(1,Nsim);
EstTE_Belloni=zeros(1,Nsim);

IncProb_Bayes=zeros(p,Nsim);
IncProb_NaiveLasso=zeros(p,Nsim);
IncProb_Belloni=zeros(p,Nsim);

w_finalAll=zeros(1,Nsim);
Lam0_finalAll=zeros(1,Nsim);
FirstStageInclusion=zeros(p,Nsim);

ub_Bayes=zeros(1,Nsim);
lb_Bayes=zeros(1,Nsim);
ub_Belloni=zeros(1,Nsim);
lb_Belloni=zeros(1,Nsim);


seedForSimulation = randi(10*Nsim,1,Nsim);


parfor sim=1:Nsim
  rng(seedForSimulation(1, sim), 'twister');

    sim

[y,T,X]=genData(n,p,sigmaX);

% Antonelli etal (2019) 
[BetaAllPost,GammaPost,Sigma2Post,ThetaPost,Tau2Post,Lambda0Post,...
    activeX,wj_final,Lambda0_final]=Antonelli2019(y,X,T,T_type);

FirstStageInclusion(:,sim)=activeX;
w_finalAll(sim)=wj_final;
Lam0_finalAll(sim)=Lambda0_final;
IncProb_Bayes(:,sim)=mean(GammaPost,2);
EstTE_Bayes(sim)=mean(BetaAllPost(2,:));
lb_Bayes(sim)=quantile(BetaAllPost(2,:),0.025);
ub_Bayes(sim)=quantile(BetaAllPost(2,:),0.975);


% Naive Outcome LASSO and Post Selection LASSO
[beta_naive,beta_postselect]=NaiveOutcomeLasso(y,X,T);
EstTE_NaiveLasso(sim)=beta_naive(2);
EstTE_PostLasso(sim)=beta_postselect(2);
IncProb_NaiveLasso(:,sim)=(beta_naive(3:end)~=0);

% Usual OLS 
% BetaAll_ols=inv(Xall'*Xall)*Xall'*y;
% EstTE_OLS(sim)=BetaAll_ols(2);

% Belloni etal (2014) Double Post Selection
[BetaAll_belloni,se_TE]=Belloni2014(y,Xall);
EstTE_Belloni(sim)=BetaAll_belloni(2);
IncProb_Belloni(:,sim)=(BetaAll_belloni(3:end)~=0);

lb_Belloni(sim)=EstTE_Belloni(sim)-1.96*se_TE;
ub_Belloni(sim)=EstTE_Belloni(sim)+1.96*se_TE;

end 
n
p
sigmaX

BiasBayes=100*(EstTE_Bayes-1)/1;
BiasNaiveLasso=100*(EstTE_NaiveLasso-1)/1;
BiasPostLasso=100*(EstTE_PostLasso-1)/1;
BiasBelloni=100*(EstTE_Belloni-1)/1;
BiasOLS=100*(EstTE_OLS-1)/1;

MSEBayes=mean( (EstTE_Bayes-1).^2 );
MSENaiveLasso=mean( (EstTE_NaiveLasso-1).^2 );
MSEPostLasso=mean( (EstTE_PostLasso-1).^2 );
MSEBelloni=mean( (EstTE_Belloni-1).^2 );
MSEOLS=mean( (EstTE_OLS-1).^2 );

CovBayes=(lb_Bayes<1 & 1<ub_Bayes);
CovBelloni=(lb_Belloni<1 & 1<ub_Belloni);

ILBayes=ub_Bayes-lb_Bayes;
ILBelloni=ub_Belloni-lb_Belloni;





fprintf('Bias: (1)Bayes, (2)NaiveOutcomeLASSO, (3)PostLASSO, (4)DoublePostSelec')
disp([mean( BiasBayes ),mean( BiasNaiveLasso ),mean(BiasPostLasso ), mean( BiasBelloni )])
% fprintf('Bias: (4)OLS')
% disp( mean( BiasOLS )  )

fprintf('MSE: (1)Bayes, (2)NaiveOutcomeLASSO, (3)PostLASSO, (4)DoublePostSelec')
disp([MSEBayes, MSENaiveLasso,MSEPostLasso,MSEBelloni  ])
% fprintf('MSE: (4)OLS')
% disp( MSEOLS   )


fprintf('Coverage: (1)Bayes, (4)DoublePostSelec')
disp([mean( CovBayes ),mean( CovBelloni )])

fprintf('Interval length: (1)Bayes, (4)DoublePostSelec')
disp([mean( ILBayes ),mean( ILBelloni )])



fprintf('BiasBayes Lam0 weight BiasNaiveLASSO BiasPostLASSO')
[ BiasBayes',Lam0_finalAll' , w_finalAll', BiasNaiveLasso',BiasPostLasso']

fprintf('IncProbBayes FirstStage IncProbNaiveLASSO DoublePostSelec')
100*[mean(IncProb_Bayes,2),mean(FirstStageInclusion,2),mean(IncProb_NaiveLasso,2),mean(IncProb_Belloni,2)]

delete(gcp('nocreate')) 
