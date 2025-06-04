function [Beta] = randn_gibbs(y,X,mm,lambda,n,p,type)

switch type
    case 1 % new method    
        U = bsxfun(@times,lambda,X');
        %% step 1 %%
        u = normrnd(mm,sqrt(lambda));
        %u = sqrt(lambda).*randn(p,1);	
        v = X*u + randn(n,1);
        %% step 2 %%
        v_star = ((X*U) + eye(n))\(y-v);
        Beta = (u + U*v_star);    

    case 2 % new method
        Q_star=X'*X;
        L=chol((Q_star + diag(1./lambda)),'lower');   
        v=L\(y'*X)';
        mu=L'\v;
        u=L'\randn(p,1);
        Beta=mu+u;
end