function [y,x,beta,sigma2] = GenRegr_sep2021(n,p,options)

% Generate predictors x
if options.corr == 0      % Uncorrelated predictors
    x = randn(n,p);
elseif options.corr == 1  % Spatially correlated predictors
    C = toeplitz(options.rho.^(0:p-1)',options.rho.^(0:p-1)');    
    x = randn(n,p)*chol(C);
else
    error('Wrong choice of options.corr');
end
x=zscore(x);

% Generate coefficients
beta=zeros(p,1);

% Nsignal=floor(p*(options.q));% number of signals 
% for j=1:Nsignal
%    if mod(j,2)==1%odd
%        beta(j)=-2;
%    elseif mod(j,2)==0%even 
%        beta(j)=2;
%    end 
% end 
beta(1:6)=[1.5 -1.5 2 -2 2.5 -2.5];

sigma2=(options.sigma)^2;
c=sqrt( (sigma2/(beta'*beta))*(options.R2/(1-options.R2)) );
beta=c*beta;


% Generate variance
if options.hetero == 0
    sigma2 = options.sigma;
elseif options.hetero == 1
    sigma2 = (rand + (x.^2)*rand(p,1)); 
    reweight = mean(sigma2)/options.sigma;
    sigma2 = sigma2/reweight;
else
    error('Wrong choice of options.hetero');
end

% Generate regression
y = x*beta + sqrt(sigma2).*randn(n,1);


end 
