function [alpha_vec,y_til,lambda_cov,tau_cov] = drawA(y,resid,lambda_cov,tau_cov,sigma_sq,ieq,M)

residx = resid./sigma_sq;
rr = residx(:,1:ieq-1)'*residx(:,1:ieq-1);
[alpha,lambda_cov,tau_cov,~] = horseshoe(residx(:,ieq),residx(:,1:ieq-1),rr,lambda_cov,tau_cov,1,-999,2);
y_til = y(:,ieq) - resid(:,1:ieq-1)*alpha;
alpha_vec = [alpha', 1, zeros(1,M-ieq)];