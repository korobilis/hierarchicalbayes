function [beta] = FFBS(y,x,W,v)

[T,p] = size(x);
n = size(y,2);
beta = zeros(p,T);

a = zeros(p,T);
m = zeros(p,T);
R = zeros(p,p,T);
C = repmat(4*eye(p),1,1,T);
f = zeros(n,T);
q = zeros(n,T);
e = zeros(n,T);
A = zeros(p,T);

% Forward filtering
for t = 1:T
    % Update time t prior
    a(:,t) = m(:,max(1,t-1));
    R(:,:,t) = C(:,:,max(1,t-1)) + diag(W(:,t));
    % One step ahead predictive distributions
    f(:,t) = x(t,:)*a(:,t);
    q(:,t) = x(t,:)*R(:,:,t)*x(t,:)' + v(t,:);
    e(:,t) = y(t,:)' - f(:,t);
    % Time t posterior
    A(:,t) = (R(:,:,t)*x(t,:)')/q(:,t);
    m(:,t) = a(:,t) + A(:,t)*e(:,t);
    C(:,:,t) = R(:,:,t) - A(:,t)*A(:,t)'*q(:,t);
end

B = zeros(p,p,T);
aT = zeros(p,T);    aT(:,T) = m(:,T);
RT = zeros(p,p,T);  RT(:,:,T) = C(:,:,T);

% Backwards sampling
beta(:,T) = aT(:,T) + chol(RT(:,:,T))'*randn(p,1); % Sample time T
D = eye(p);
for t=T-1:-1:1
    B(:,:,t) = C(:,:,t)/R(:,:,t+1);
    aT(:,t) = m(:,t) + B(:,:,t)*(beta(:,t+1) - a(:,t+1));
    RT(:,:,t) = C(:,:,t)*(D - B(:,:,t)');
    %RT(:,:,t) = C(:,:,t) - B(:,:,t)*R(:,:,t+1)*B(:,:,t)';
    
    beta(:,t) = aT(:,t) + chol(RT(:,:,t))'*randn(p,1); % Sample time t
end