function [Q,miu,lambda] = multipleshrinkage_prior(beta,miu,lambda,theta,a0,b0,a1,b1,c,d,clus,p_star)

Tp = length(beta);
m  = zeros(p_star,1);
V  = zeros(p_star,1);
pi = zeros(p_star,1);
miu_bar = zeros(p_star,1);
lambda_bar = zeros(p_star,1);
% Update prior location and scale parameters  
for kk = 1:p_star
    if kk == 1            
        m(kk,1) = sum(clus==kk);  % Number of predictors that fall in the 1st bin
        % a. Update the pair (miu_bar[1],tau_bar[1])
        if m(kk,1) ~= 0                
            miu_bar(kk,1) = 0;
            lambda_bar(kk,1) = 1e+10;%Draw_Gamma( m(kk,1)/2 + a0, sum(((beta(clus==kk) - miu(clus==kk)).^2))/2 + b0 );
        elseif m(kk,1) == 0        
            miu_bar(kk,1) = 0;
            lambda_bar(kk,1) = gamrnd(a0,1./b0);
        end
        % b. Update pi[1]= V_t
        V(kk,1) = betarnd(m(kk,1) + 1, Tp - sum(m(1:kk,1)) + theta);
        pi(kk,1) = V(kk,1);       
    else
        m(kk,1) = sum(clus==kk);
        % a. Update the pairs
        % (miu_bar[2],tau_bar[2]),...,(miu_bar[p_star],tau_bar[p_star])
        if m(kk,1) ~= 0 
            V_miu_bar_t = inv(1/d + sum(1./lambda(clus==kk)));
            E_miu_bar_t = V_miu_bar_t*(c/d + sum(beta(clus==kk)./lambda(clus==kk)));
            miu_bar(kk,1) = E_miu_bar_t + sqrt(V_miu_bar_t)*randn;
            lambda_bar(kk,1) = gamrnd( m(kk,1)/2 + a1,1./(sum(((beta(clus==kk) - miu(clus==kk)).^2))/2  + b1) );
        elseif m(kk,1) == 0        
            miu_bar(kk,1) = c + sqrt(d)*randn;
            lambda_bar(kk,1) = gamrnd(a1,1./b1);
        end        
        % b. Update pi_t from prod(1 - pi[h])V_t for h=1,...,kk-1
        V(kk,1) = betarnd(m(kk,1) + 1, Tp - sum(m(1:kk,1)) + theta);
        pi(kk,1) = prod( (1 - pi(1:kk-1,1)) )*V(kk,1);
    end
end

% Make sure pi is a properly defined probability density  
pi = pi./sum(pi);
    
U = rand(Tp,1);
for j = 1:Tp        
    q = pi.*(normpdf(beta(j,1),miu_bar,1./lambda_bar));   
    q = q./sum(q);
    clus(j,1) = min(find(cumsum(q)>=U(j,1)));         %#ok<MXFND>
    prob_incl(j,1) = q(1,1);
end

% Update the estimate of the prior mean
miu = miu_bar(clus);
lambda = lambda_bar(clus);
Q = 1./lambda;
