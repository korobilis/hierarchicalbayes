function [Beta] = randn_chol(y,X,lambda,p)
%% matrices %%
Q_star=X'*X;
Dinv = (1./lambda)*eye(p);       
L=chol((Q_star + Dinv),'lower');
v=L\(y'*X)';
mu=L'\v;
u=L'\randn(p,1);
Beta = mu+u;