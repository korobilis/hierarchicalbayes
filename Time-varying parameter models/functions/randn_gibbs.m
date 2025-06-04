function [Beta] = randn_gibbs(y,X,mm,lambda,n)


    U = bsxfun(@times,lambda,X');
    %% step 1 %%
    u = normrnd(mm,sqrt(lambda));	
    v = X*u + randn(n,1);
    %% step 2 %%
    v_star = ((X*U) + eye(n))\(y-v);
    Beta = (u + U*v_star);

% U=bsxfun(@times,(lambda.^2),X');
% %% step 1 %%
% u = normrnd(0,lambda);
% v = X*u + randn(n,1);
% %% step 2 %%
% v_star = (X*U+eye(n))\(y-v);
% Beta = (u+U*v_star);


% %% matrices %%
% Q_star=X'*X;
% Dinv = diag(1./lambda);       
% L=chol((Q_star + Dinv),'lower');
% v=L\(y'*X)';
% mu=L'\v;
% u=L'\randn(p,1);
% Beta = mu+u;