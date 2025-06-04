% b0 for student-t
% tau,kappa for lasso
% kappa,xi,nu for horsehoshoe-simple 

function [Q,kappa,xi,nu]=cts_prior(beta,sigma2,b0,tau,kappa,xi,nu,lambda,method)
p=length(beta);

switch lower(method)
    case 'student-t'
        invQ = gamrnd(1 + 1/2, 1./(b0 + (beta.^2)./(2*sigma2)) );
        Q    = 1./invQ;
    case 'lasso'
        lam  = gamrnd(p + 1,1./(0.5*sum(tau) + kappa));		
        tau  =  min(1./random('InverseGaussian',sqrt((lam*sigma2)./(beta.^2)),lam,p,1),1e+6);
        Q    = tau;
    case 'horseshoe-slice'
        % Horseshoe prior
        % update lambda_j's in a block using slice sampling %%  
        eta = 1./(lambda.^2); 
        upsi = unifrnd(0,1./(1+eta));
        %tempps = beta.^2/(2*tau^2); 
        tempps = beta.^2/(2*sigma2*tau^2); 
        ub = (1-upsi)./upsi;
        % now sample eta from exp(tempv) truncated between 0 & upsi/(1-upsi)
        Fub = 1 - exp(-tempps.*ub); % exp cdf at ub 
        Fub(Fub < (1e-4)) = 1e-4;  % for numerical stability
        up = unifrnd(0,Fub); 
        eta = -log(1-up)./tempps; 
        lambda = 1./sqrt(eta);
        % update tau %%
        tempt = sum((beta./lambda).^2)/(2*sigma2); 
        et = 1/tau^2; 
        utau = unifrnd(0,1/(1+et));
        ubt = (1-utau)/utau; 
        Fubt = gamcdf(ubt,(p+1)/2,1/tempt); 
        Fubt = max(Fubt,1e-8); % for numerical stability
        ut = unifrnd(0,Fubt); 
        et = gaminv(ut,(p+1)/2,1/tempt); 
        tau = 1/sqrt(et);
        % update estimate of Q and Q^{-1}
        Q = (lambda.*tau).^2;
        
    case 'horseshoe-mixture'
        rate = 1./nu + (beta.^2)/(2*kappa*sigma2);
        lambda = 1./gamrnd(1,1./rate);    % random inv gamma with shape=1, rate=rate
        
        rate  = 1/xi + (1/(2*sigma2))*sum(beta.^2./lambda);
        kappa = 1/gamrnd((p+1)/2, 1/rate);    % inv gamma w/ shape=(p+1)/2, rate=rate
        
        rate = 1+1./lambda;
        nu   = 1./gamrnd(1,1./rate);    % random inv gamma with shape=1, rate=1+1/lambda_sq_12
     
        rate = 1+1/kappa;
        xi   = 1/gamrnd(1,1/rate);    % inv gamma w/ shape=1, rate=1+1/tau_sq
              
        Q=kappa*lambda;
end 

end 