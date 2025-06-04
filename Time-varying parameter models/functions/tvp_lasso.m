%%%%*********************************************************
%%%% function to get penalized TVP estimator
%%%%*********************************************************
function [beta,sigma] = tvp_lasso(y,X)

[T,p] = size(X);
Tp        = T*p;   
H         = speye(Tp,Tp) - sparse(p+1:Tp,1:(T-1)*p,ones(1,(T-1)*p),Tp,Tp);
Hinv      = speye(Tp,Tp)/H;
bigG      = SURform(X)*Hinv;

Lambda = logspace(-5,-1,15);
bigG = bigG'; 
CVMdl = fitrlinear(bigG,y-mean(y),'ObservationsIn','columns','KFold',5,'Lambda',Lambda,...
    'Learner','leastsquares','Solver','sparsa','Regularization','lasso');
mse = kfoldLoss(CVMdl);
[~,pos] = min(mse);
Mdl = fitrlinear(bigG,y-mean(y),'ObservationsIn','columns','Lambda',Lambda,...
    'Learner','leastsquares','Solver','sparsa','Regularization','lasso');
beta = reshape(Hinv*Mdl.Beta(:,pos),p,T)';
sigma = (y-sum(X.*beta,2))'*(y-sum(X.*beta,2))/(T-size(X,2));

end