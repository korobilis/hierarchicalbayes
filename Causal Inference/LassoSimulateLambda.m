%  The lambda( 1- ALPHA | X ) is the 1-ALPHA quantile of
%  
%    || S ||_{\infty} = 2 max_{j <= p} | n^{1/2} E_n[ e_t x_{tj} / s_j ] |
%    s_j = sqrt{ E_n[ x_{tj}^2 ] }
%
%

function [ lambda_final ] = LassoSimulateLambda ( X, sigma, NumSim, n, ALPHA )

NumSim = max(NumSim, n);

[ ~, NumColumns ] = size( X );

NormXX = zeros(NumColumns,1);
lambda = zeros(NumSim,1);

for j = 1 : NumColumns
    NormXX(j) = norm(X(:,j)/sqrt(n));
end    


for k = 1 : NumSim  
    error = sigma*randn(n,1);          
    lambda(k) = 2*max( abs(  ( X'*error )./NormXX ) );        
end

lambda_final = quantile(lambda, 1-ALPHA ); 

