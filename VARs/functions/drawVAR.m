function [beta,BETA,A,sigma_t,h,miu,Q,lambda0,lambda1,tau] = drawVAR(y,X,T,M,KK,sigma_t,h,sig,miu,Q,b0,c0,pi0,lambda0,lambda1,tau,prior,est_meth,sv_reg)

BETA = zeros(KK,M);
A = eye(M);
resid = zeros(T,M);
X_new = X;

% sample equation-by-equation
for ieq = 1:M                  
    % =====| Step 1: Sample VAR coefficients and covariances                
    u = (1./sqrt(sigma_t(:,ieq)));
    ys = y(:,ieq).*u;                       % Standardize y for GLS formula
    xs = X_new.*u;                          % Also standardize x           
    beta = randn_gibbs(ys,xs,miu{ieq,1},Q{ieq,1},T,KK+ieq-1,est_meth);  % sample regression coefficients
    BETA(:,ieq) = beta(1:KK);
    A(ieq,1:ieq-1) = beta(KK+1:end);
           
    %--draw prior variance of BETA and A
    if prior == 1
        [Q{ieq,1},~,miu{ieq,1}] = student_T_prior(beta,b0);
    elseif prior == 2       
        [Q{ieq,1},~,miu{ieq,1},lambda0{ieq,1}] = ssvs_prior(beta,c0,lambda0{ieq,1},b0,pi0);
    elseif prior == 3
        [Q{ieq,1},~,miu{ieq,1},lambda0{ieq,1},tau(ieq)] = horseshoe_prior(beta,KK+ieq-1,lambda0{ieq,1},tau(ieq));
    elseif prior == 4
        [Q{ieq,1},~,miu{ieq,1},lambda0{ieq,1},lambda1{ieq,1}] = sns_lasso(beta,lambda0{ieq,1},lambda1{ieq,1},pi0);
    end
    % impose restrictions on intercept and AR(1) coefficient
    Q{ieq,1}(ieq+1) = 0.5; % Prior variance on AR(1) lag
    Q{ieq,1}(1) = 100;    % Prior variance on intercept
        
    % =====| Step 2: Sample regression variance
    if sv_reg == 1   % Sample Stochastic Volatility
        yhat = y(:,ieq) - X_new*beta;                           % regression residuals
        ystar  = log(yhat.^2 + 1e-6);                           % log squared residuals        
        [h(:,ieq), ~] = SVRW(ystar,h(:,ieq),sig(ieq),4);        % log stochastic volatility using Chan's filter   
        sigma_t(:,ieq)  = exp(h(:,ieq));                        % convert log-volatilities to variances
        r1 = 1 + T - 1;   r2 = 0.01 + sum(diff(h(:,ieq)).^2)';  % posterior moments of variance of log-volatilities
        sig(ieq) = 1./gamrnd(r1./2,2./r2);                      % sample variance of log-volatilities from Inverse Gamma
    elseif sv_reg == 0   % Sample constant regression variance          
        a1 = 0.1 + T/2;    sse = (y(:,ieq) - X_new*beta).^2;
        a2 = 0.1 + sum(sse);       
        sigma2 = 1./gamrnd(a1,1./a2);                           % Sample from inverse-Gamma
        sigma_t(:,ieq) = repmat(sigma2,T,1);                    % Convert to a T x n_q matrix (but replicate same value for all t=1,...,T)   
    end
    
    % Construct residual to be used in next VAR equation
    resid(:,ieq) = (y(:,ieq) - X_new*beta);
    X_new = [X_new, resid(:,ieq)];
end