function [beta] = reg_fast(y,x,beta,W)


[T,p] = size(x);
%beta = zeros(p,T);

y_til = y;% - x(1,:)*(sqrt(diag(W(:,1)))*randn(p,1));
for t = randperm(T)
    D = W(:,t);
    xrow = x(t,:)./D';
    Sigma = diag(1./D) - (xrow'*xrow)/(1 + xrow*x(t,:)');
    mu = x(t,:)'*y_til(t,:);
    % sample beta(t)
    beta(:,t) = Sigma*mu + chol(Sigma)'*randn(p,1);
    if t<T
        bb = beta(:,1:t);
        y_til(t+1,:) = y(t+1,:) - repmat(x(t+1,:),1,t)*bb(:);
    end
end
%beta = cumsum(beta,2);