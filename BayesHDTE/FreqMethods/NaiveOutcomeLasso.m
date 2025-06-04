% 1. Naive Outcome Lasso
% i.e. LASSO of y on (T,X) with no penalty on the slope on T
% 2. Post Selection Lasso
% i.e. after 1, just regress Y on T and the selected X's

% We use the Matlab ToolBox "penalized" McIlhagga
% Journal of Statistical Software (2016) 

% Specifically, we use "Adaptive LASSO" 
% The naive outcome LASSO is equivalent to the adaptive LASSSO
% with "weight" being zero for T and 1 for X 
% Note that in this toolbox, the weight on the intercept is 
% always zero, so we don't specify it in "weight" 

function [beta_naive,beta_postselect]=NaiveOutcomeLasso(y,X,T)
[n,p]=size(X);
weight=ones(p+1,1);weight(1)=0;
model = glm_gaussian(y, [T,X]);
%fit = penalized(model, @p_lasso, "penaltywt", weight);
cv = cv_penalized(model, @p_lasso, "folds", 10, "penaltywt", weight);
%mat=cv.fit.beta;SE=cv.cvse;Lam=cv.lambda;cv.cve(cv.minlambda);figure;plot(cv.lambda,cv.cve,'o')
% Lam(cv.minlambda)
% Difference between error and the min_error
MinCVE=cv.cve(cv.minlambda);
diff_err=abs( cv.cve-MinCVE(1) );%in case of ties, take (1)
% "1 standard error" for the min_error model
oneSE=cv.cvse(cv.minlambda);oneSE=oneSE(1);%in case of ties, take (1)
% Which models are within 1se of the min_error? 
withinoneSEmin=(diff_err<oneSE);
% Effective num. of parameters 
EffSize=cv.p;
% Which model is the sparsest model within 1se of the min_error
ind=find( EffSize==min( EffSize(withinoneSEmin) ) );
ind=ind(1);%In case there are ties

beta_mat=cv.fit.beta;
% Naive outcome lasso
beta_naive=beta_mat(:,ind);
%te_naive=beta_naive(2);

% Post selection lasso
selected=(beta_naive(3:end)~=0);
Xselected=[ones(n,1),T,X(:,selected)];
beta_postselect=inv(Xselected'*Xselected)*Xselected'*y;
%te_postselect=beta_postselect(2);

end 


