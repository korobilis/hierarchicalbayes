function [beta] = FFBS_fast(y,x,beta,W)

[T,p] = size(x);

% sample initial condition beta(0)
invD = diag(1./((1/10)*ones(p,1) + W(:,1)));
b0 = invD*(W(:,1).*beta(:,1)) + sqrt(invD)*randn(p,1);
invD = eye(p); idx = 1:p+1:p*p;
for t = randperm(T)
    previous = (t>1)*beta(:,max(1,t-1)) + (t==1)*b0;
    next = (t<T)*beta(:,min(T,t+1));
    D = W(:,t) + (t<T)*W(:,min(T,t+1));
    xrow = x(t,:)./D';
    invD(idx) = 1./D;
    Sigma = invD - (xrow'*xrow)/(1 + xrow*x(t,:)');
    
    % sample beta(t)
    mu = x(t,:)'*y(t,:) + W(:,t).*previous + (t<T)*W(:,min(T,t+1)).*next;    
    beta(:,t) = Sigma*mu + chol(Sigma)'*randn(p,1);
end
