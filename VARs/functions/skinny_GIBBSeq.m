function [beta,gamma,sigma2]  = skinny_GIBBSeq(y,x,xx,gamma,sigma2,T,p,pi0,stau_0,stau_1,tau_1,v_i_sqrt)

% Prepare some hyperparameters here (consider having them once as input, not defined in
% each draw)
beta = zeros(p,1);

% 1. Update beta from Normal
index_A = find(gamma==1);  p_A = length(index_A);
index_I = find(gamma==0);  p_I = p - p_A;   
xs = x(:,index_A)./sqrt(sigma2);
ys = y./sqrt(sigma2);  
beta_A = randn_gibbs(ys,xs,tau_1,p_A,T);
beta_I = normrnd(0,v_i_sqrt*ones(p_I,1));
beta(index_A) = beta_A; beta(index_I) = beta_I;

% 2. Update gamma from Bernoulli
l_0 = (1-pi0)*normpdf(beta,0,stau_0);
l_1 = pi0*normpdf(beta,0,stau_1);
xg = (x.*repmat((beta.*gamma)',T,1));
sxg =  sum(xg,2);
sse = y - sxg + xg;
correction = sum(((x.*repmat(beta',T,1))./sigma2).*sse)' + .5*(xx.*(1-1./sigma2)).*(beta.^2);
temp  = min((l_1./l_0).*exp(correction),1000000);    
gamma = bernoullimvrnd(temp./(1+temp),p);

% 3. Update sigma2 from Inverse Gamma
c1 = p_A + T;
er2 = (y-x*beta);
PSI = er2'*er2;
c2 = sum((beta_A.^2)./tau_1) + PSI; %(beta_A'/(tau_1*eye(p_A)))*beta_A + PSI;
sigma2 = 1/gamrnd(c1/2,2/c2);