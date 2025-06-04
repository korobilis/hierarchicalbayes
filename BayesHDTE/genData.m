
function [y,T,X,IndX]=genData(n,p,options)
options.sigmaX=1;options.rho=0.5;
% 1. Generate predictors x
if options.corr == 0      % Uncorrelated predictors
    X = options.sigmaX*randn(n,p);
elseif options.corr == 1  % Spatially correlated predictors
    C = toeplitz(options.rho.^(0:p-1)',options.rho.^(0:p-1)');    
    X = randn(n,p)*chol(C);
% elseif options.corr == 2  % Randomly correlated predictors    
%     % create random correlation matrix
%     temp = tril(rand(p),-1);
%     A = eye(p) + temp + temp';
%     C = A'*A;
%     C = (diag(diag(sqrt(C)))\C)/diag(diag(sqrt(C)));    
%     X = randn(n,p)*chol(C);
 else
    error('Wrong choice of options.corr');
end

% 2. Define slopes 
beta0=0;% intercept (in the outcome equation)
alpha=1;% treatment effect 

% p times q must be divisible by 8 
% First p*q/4 X's are strong coufounders (corr strongly with both T and y)
% Second p*q/4 X's are weak coufounders  (corr strongly with T, weakly with y)
% Third p*q/4 X's are instruments        (corr strongly with T, no corr with y)
% Last p*q/4 X's are strong predictors   (no corr with T, strongly corr with y)
 

NumImX=p*options.q;% Number of important X's, must be divisible by 8 
if mod(NumImX,8)~=0
    fprintf('p times q must be divisible by 8')
end 

IndX=zeros(p,4);
IndX(0*NumImX/4+1:1*NumImX/4,1)=1;%strong cofounders
IndX(1*NumImX/4+1:2*NumImX/4,2)=1;%weak cofounders
IndX(2*NumImX/4+1:3*NumImX/4,3)=1;%instruments 
IndX(3*NumImX/4+1:4*NumImX/4,4)=1;%strong predictors 
IndX=logical(IndX);

psi_ = zeros(p,1);
%psi_(1:6)=[1, -1, 1, -1, 1, -1]';
beta_ = zeros(p,1);
%beta_(1:8)=[1,-1,0.3,-0.3,0,0,1,-1]';
for j=1:p
    if IndX(j,1)==1 %strong cofounders
    psi_(j)= 1*(mod(j,2)~=0)+-1*(mod(j,2)==0);%1 if j odd, -1 if j even 
    beta_(j)=1*(mod(j,2)~=0)+-1*(mod(j,2)==0);%1 if j odd, -1 if j even 
    elseif IndX(j,2)==1 %weak cofounders
    psi_(j)= 1*(mod(j,2)~=0)+-1*(mod(j,2)==0);%1 if j odd, -1 if j even 
    beta_(j)=0.3*(mod(j,2)~=0)+-0.3*(mod(j,2)==0);%0.3 if j odd, -0.3 if j even 
    elseif IndX(j,3)==1 %instruments
    psi_(j)= 1*(mod(j,2)~=0)+-1*(mod(j,2)==0);%1 if j odd, -1 if j even 
    beta_(j)=0;
    elseif IndX(j,4)==1 %strong predictors
    psi_(j)=0;
    beta_(j)=1*(mod(j,2)~=0)+-1*(mod(j,2)==0);%1 if j odd, -1 if j even 
    end 
end
%beta_(9:end)=0.1*randn(p-8,1);
beta_(NumImX+1:end)=0.1*randn(p-NumImX,1);


if options.corr == 0      % Uncorrelated predictors
    %theta'*((options.sigmaX)^2)*theta
    %Signal=norm(options.sigmaX*theta)^2;
     Signal_t=norm(options.sigmaX*psi_)^2;
     Signal_y=norm(options.sigmaX*beta_)^2;
elseif options.corr == 1  % Spatially correlated predictors
    %theta'*(chol(C)'*chol(C))*theta
    %Signal=norm(chol(C)*theta)^2;
    Signal_t=norm(chol(C)*psi_)^2;
    Signal_y=norm(chol(C)*beta_)^2;
end

sigma2_t=1;sigma2_y=1;
%Determine c_t and c_y as in Belloni etal (2014, ReStud) 'additional simulations,
%'last 13 simulations'
c_t=sqrt( (sigma2_t/Signal_t) * (options.R2_t/(1-options.R2_t)) );
c_y=sqrt( (sigma2_y/Signal_y) * (options.R2_y/(1-options.R2_y)) );

psi  = c_t*psi_;
beta = c_y*beta_;


BetaAll = [beta0; alpha; beta];

% 3. Generate erros 
if options.htrs==0 % Homosckedastic errors 
    v=sqrt(sigma2_t)*randn(n,1);
    epsilon=sqrt(sigma2_y)*randn(n,1);
    
    T=X*psi+v;
    Xall =[ones(n,1),T,X];
    y = Xall*BetaAll + epsilon;
   
elseif options.htrs==1 % Heteroskedastic errors 
   
    %sigma_t_htrs=(1+X*theta).^2;
    sigma_t_htrs=(1+X*psi_).^2;
   
    sigma_t_htrs= sqrt(sigma_t_htrs/mean(sigma_t_htrs) );
    v=sigma_t_htrs.*randn(n,1);
    T=X*psi+v;
  
    Xall =[ones(n,1),T,X];
    
    sigma_y_htrs=(1+Xall*[beta0; alpha; beta_]).^2;
    sigma_y_htrs= sqrt(sigma_y_htrs/mean(sigma_y_htrs) );
    
    
    epsilon=sigma_y_htrs.*randn(n,1);
    y = Xall*BetaAll + epsilon; 
end 

%TrueActiveX=(beta~=0);

end 