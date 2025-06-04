
 
% Modified by Kenichi on June 25, 2021

function [Beta] = randn_gibbs(y,X,lambda,sigma2,n,p)
% Sample Gaussian posterior efficiently
if n<=p
   % U = bsxfun(@times,lambda,X'); June 25
    U = bsxfun(@times,sigma2*lambda,X');    
    % step 1 %
    %u = normrnd(0,sqrt(lambda));June 25
    %there are some very rare cases with a very small negative number for
    %tau's, so we take real(). The results seem to be unchanged with or without real()

    u = normrnd(0,real( sqrt(sigma2*lambda) ) );
   
    v = X*u + randn(n,1);
    % step 2 %
    v_star = ((X*U) + eye(n))\(y-v);
    Beta = (u + U*v_star);
else  
    Q_star=X'*X;
    Dinv = diag(1./lambda);       
    L=chol((Q_star + Dinv),'lower');
    v=L\(y'*X)';
    mu=L'\v;
    u=L'\randn(p,1);
    %Beta = mu+u; June 25
    Beta = mu+sqrt(sigma2)*u;
end
    
end




