function Beta = randn_gibbs_sep2021(y,X,Q,n,p,sigma2,conjugate)

if conjugate==1
    if p<n
    XX=X'*X;
    Qinv = diag(1./Q);       
    L=chol((XX + Qinv),'lower');%L*L'=(XX + Qinv)
    v=L\(y'*X)';
    mu=L'\v;
    u=L'\randn(p,1);
    Beta = mu+sqrt(sigma2)*u; 
    else
    Phi=X/sqrt(sigma2);  
    alpha=y/sqrt(sigma2);
    D  = sigma2*Q;

    DPhi = bsxfun(@times,D,Phi');%same as diag(D)*Phi' 
    u = normrnd(zeros(p,1),sqrt(D));
    v = Phi*u + randn(n,1);
    w_star = ( (Phi*DPhi) + eye(n) )\(alpha - v); %Phi*DPhi same as (Phi*diag(D))*Phi'
    Beta = u + DPhi*w_star;
    end 
elseif conjugate==0 %independent priors on beta and sigma2
    if p<n
    XX=X'*X;
    Qinv = diag(1./Q);       
    L=chol((XX/sigma2 + Qinv),'lower');%L*L'=(XX/sigma2 + Qinv)
    v=L\(y'*X)';
    mu=L'\v;
    u=L'\randn(p,1);
    Beta = mu/sigma2+u; 
    else
    Phi=X/sqrt(sigma2);  
    alpha=y/sqrt(sigma2);
    D  = Q;

    DPhi = bsxfun(@times,D,Phi');
    u = normrnd(zeros(p,1),sqrt(D));
    v = Phi*u + randn(n,1);
    w_star = ( (Phi*DPhi) + eye(n) )\(alpha - v);
    Beta = u + DPhi*w_star;
    end    
end 

% 
% clear 
% n=100;
% p=10;
% X=randn(n,p);
% beta=randn(p,1);
% y=X*beta+randn(n,1);
% 
% sigma2=2;
% Q=rand(p,1);
% 
% Nsim=1000;
% 
% % Test for conjugate case 
% Beta1=zeros(p,Nsim);
% Beta2=zeros(p,Nsim);
% for sim=1:Nsim
%     XX=X'*X;
%     Qinv = diag(1./Q);       
%     L=chol((XX + Qinv),'lower');%L*L'=(XX + Qinv)
%     v=L\(y'*X)';
%     mu=L'\v;
%     u=L'\randn(p,1);
%     Beta1(:,sim) = mu+sqrt(sigma2)*u; 
% end 
% for sim=1:Nsim
%     Phi=X/sqrt(sigma2);  
%     alpha=y/sqrt(sigma2);
%     D  = sigma2*Q;
%     
%    DPhi = bsxfun(@times,D,Phi');
%     u = normrnd(zeros(p,1),sqrt(D));
%     v = Phi*u + randn(n,1);
%     w_star = ( (Phi*DPhi) + eye(n) )\(alpha - v);
%     Beta2(:,sim) = u + DPhi*w_star;
% end 
% [mean(Beta1,2),mean(Beta2,2)]
% 
% % Test for indep case 
% Beta3=zeros(p,Nsim);
% Beta4=zeros(p,Nsim);
% for sim=1:Nsim
%     XX=X'*X;
%     Qinv = diag(1./Q);       
%     L=chol((XX/sigma2 + Qinv),'lower');%L*L'=(XX/sigma2 + Qinv)
%     v=L\(y'*X)';
%     mu=L'\v;
%     u=L'\randn(p,1);
%     Beta3(:,sim) = mu/sigma2+u; 
% end 
% 
% for sim=1:Nsim
%     Phi=X/sqrt(sigma2);  
%     alpha=y/sqrt(sigma2);
%     D  = Q;
% 
%     DPhi = bsxfun(@times,D,Phi');
%     u = normrnd(zeros(p,1),sqrt(D));
%     v = Phi*u + randn(n,1);
%     w_star = ( (Phi*DPhi) + eye(n) )\(alpha - v);
%     Beta4(:,sim) = u + DPhi*w_star;
% end 
% 
% [mean(Beta3,2),mean(Beta4,2)]
% 
% 
