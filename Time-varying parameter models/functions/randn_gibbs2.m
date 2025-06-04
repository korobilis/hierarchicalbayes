function [Beta] = randn_gibbs2(y,X,XX,lambda,n,p)


    U = X';%bsxfun(@times,lambda,X');
    %U = lambda.*X';
    %% step 1 %%
    u = sqrt(lambda).*randn(p,1);
    v = X*u + randn(n,1);
    %% step 2 %%
    v_star = ((XX) + eye(n))\(y-v);
    Beta = (u + U*v_star);