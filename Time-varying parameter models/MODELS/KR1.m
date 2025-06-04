function [beta_save,sigma_save] = KR1(y,X,ngibbs,nburn,prior)

[T,p] = size(X);

% Some useful matrices for the time-varying parameters
Tp        = T*p;   
H         = speye(Tp,Tp) - sparse(p+1:Tp,1:(T-1)*p,ones(1,(T-1)*p),Tp,Tp);
Hinv      = speye(Tp,Tp)/H;
bigG      = full(SURform(X)*Hinv);

% Priors for TVPs
Q    = .01*ones(Tp,1);
miu  = zeros(Tp,1);
if prior == 1
    % Student_T shrinkage prior
    b0 = 0.00001;
elseif prior == 2
    % SSVS prior
    c0 = (0.01)^2;
    tau = 4*ones(Tp,1);
    pi0 = 0.25;
    b0 = 0.01;
elseif prior == 3          
    % Horseshoe prior
    tau_sq = 1;
    nu = ones(Tp,1);
    xi = 1;
end

% initialize volatility parameters
isigma_t = 0.1*ones(T,1);
h        = ones(T,1); 
sig      = 0.1;

%% ========|STORAGE MATRICES:
beta_save  = zeros(ngibbs,T,p);
sigma_save = zeros(ngibbs,T);
%% ======================= BEGIN MCMC ESTIMATION =======================
disp('Run Gibbs sampler for KR1 algorithm');
for iter = 1: (ngibbs + nburn)
%     if mod(iter,(ngibbs + nburn)/100) == 0
%         if mod(iter,(ngibbs + nburn)/10) == 0
%             fprintf('%d %% \r',round(100*(iter/(ngibbs + nburn))));
%         else
%             fprintf('%d %% \t',round(100*(iter/(ngibbs + nburn))));
%         end
%     end
    
    %% Sample beta_t
    W = Q;
    xs = isigma_t.*bigG;
    ys = isigma_t.*y;
    betaD = randn_gibbs(ys,xs,miu,W,T);
    beta = Hinv*betaD;

    %% sample Q
    if prior == 1
        [Q,~,miu] = student_T_prior(betaD,b0);
    elseif prior == 2
        [Q,~,miu,tau] = ssvs_prior(betaD,c0,tau,b0,pi0);
    elseif prior == 3
        [Q,~,miu,~,tau_sq,xi,nu] = horseshoe_prior(betaD,Tp,tau_sq,xi,nu);
    end

    %% Sample sigma2_t
    yhat = y - bigG*betaD;
    ystar  = log(yhat.^2 + 1e-6);        
    [h, ~] = SVRW(ystar,h,sig,4);  % log stochastic volatility
    isigma_t = exp(-0.5*h);
    sigma_t  = exp(h);
    r1 = 1 + T - 1;   r2 = 0.01 + sum(diff(h).^2)';
    sig = 1./gamrnd(r1./2,2./r2);   % sample state variance of log(sigma_t.^2)
         
    
    %% Save draws
    if iter > nburn
        beta_save(iter-nburn,:,:) = reshape(beta,p,T)';
        sigma_save(iter-nburn,:) = sigma_t;
    end   
end
end
