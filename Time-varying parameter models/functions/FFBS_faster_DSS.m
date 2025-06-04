function [beta,gamma,gamma0] = FFBS_faster_DSS(y,x,beta,gamma,gamma0,phi1,lambda0,lambda1,Theta)

[T,p] = size(x);

% define these scalar operations once, to reduce heavy notation
phi12 = phi1^2;
pl1   = phi1/lambda1;
pl2   = phi12/lambda1;

%% Sample beta
% sample initial condition beta(0)
invD = diag( 1./( ((1-phi12)*gamma0/lambda1 + (1-gamma0)/lambda0) + pl2*gamma(:,1)) );
b0 = invD*( (pl1*gamma(:,1)).*beta(:,1)) + sqrt(invD)*randn(p,1);

% Sart sampling, preferably in random order
invsD = eye(p); idx = 1:p+1:p*p;
for t = randperm(T)
    previous = (t>1)*beta(:,max(1,t-1)) + (t==1)*b0;
    next = (t<T)*beta(:,min(T,t+1));
    D = (gamma(:,t)/lambda1 + (1-gamma(:,t))/lambda0) + (t<T)*(pl2*gamma(:,min(T,t+1)));
	xrow = x(t,:)./D';
    sD   = sqrt(D);
	sumx = xrow*x(t,:)';	
	c = -1/(1 + sumx);	
	Xtilde = xrow'*(x(t,:)./sD');
	d = (sqrt(1+c*sumx)-1)/sumx;
    term = d*Xtilde;
    invsD(idx) = 1./sD;
	M = invsD + term;
    
    % sample beta(t)
    mu = x(t,:)'*y(t,:) + (pl1*gamma(:,t)).*previous + (t<T)*(pl1*gamma(:,min(T,t+1))).*next;
    beta(:,t) = M*(M'*mu + randn(p,1));
end


%% sample gamma
% Sample initial condition gamma(0);
thetas = theta_beta(b0,lambda1,lambda0,phi1,Theta);
gamma0 = bernoullimvrnd(thetas,p);

for t = 1:T
    thetas = (t>1)*theta_beta(beta(:,max(1,t-1)),lambda1,lambda0,phi1,Theta) ...
        + (t==1)*theta_beta(b0,lambda1,lambda0,phi1,Theta);        
        
    ps     = (t>1)*pstar_beta(beta(:,t),beta(:,max(1,t-1)),lambda1,lambda0,phi1,thetas) ...
        + (t==1)*pstar_beta(beta(:,t),b0,lambda1,lambda0,phi1,thetas);
    gamma(:,t) = bernoullimvrnd(ps,p);
end

end

function [result] = theta_beta(beta,lambda1,lambda0,phi1,Theta)
    
    num = lnormpdf(beta,0,sqrt(lambda1/(1-phi1^2)));
    den = lnormpdf(beta,0,sqrt(lambda0));
    
    result = 1./(1+((1-Theta)./Theta).*exp(den-num));
end
    
function [result] = pstar_beta(beta,beta_previous,lambda1,lambda0,phi1,theta)

	mu  = phi1*beta_previous;
	num = lnormpdf(beta,mu,sqrt(lambda1));
	den = lnormpdf(beta,0,sqrt(lambda0));
	
	result = 1./(1+((1-theta)./theta).*exp(den-num));
end

