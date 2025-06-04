function [beta] = FFBS_fast2(y,x,W)

[T,p] = size(x);
beta  = zeros(p,T);
b     = zeros(p,T);

invD = diag(1./((1/10)*ones(p,1) + W(:,1)));
b0 = invD*(W(:,1).*beta(:,1)) + sqrt(invD)*randn(p,1);

y_til = y;
for t = 1:T        
    % purge previous coefficient increments
    y_til(t,:) = y(t,:) - (t>1)*x(t,:)*sum(beta(:,1:max(1,t-1)),2);
    D = W(:,t);
    xrow = x(t,:)./D';
    Sigma = diag(1./D) - (xrow'*xrow)/(1 + xrow*x(t,:)');
    mu = x(t,:)'*y_til(t,:);% + (t==1)*(W(:,t).*b0 + W(:,min(T,t+1)).*beta(:,min(T,t+1)));
    % sample beta(t)
    b(:,t) = Sigma*mu + (t==1)*b0;
end

y_til = y;
for t = T-1:-1:1
    y_til(t,:) = y(t,:) - x(t,:)*sum(beta(:,1:t+1),2);
    D = W(:,t);
    xrow = x(t,:)./D';
    Sigma = diag(1./D) - (xrow'*xrow)/(1 + xrow*x(t,:)');
    mu = -x(t,:)'*y_til(t,:);
    % sample beta(t)
    beta(:,t+1) = Sigma*mu + chol(Sigma)'*randn(p,1);       
end