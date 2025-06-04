function [Y,X,t,M,KK,K,numa,index_cov] = prepare_BVAR_matrices(Y,p)


% Number of observations and dimension of X and Y
t = size(Y,1); % t is the time-series observations of Y
M = size(Y,2); % M is the dimensionality of Y
KK = M*p+1;
K = KK*M;
numa = M*(M-1)/2; % Number of lower triangular elements of A_t (other than 0's and 1's)

% ===================================| VAR EQUATION |==============================
% Take lags, and correct observations
Ylag = mlag2(Y,p);
Ylag = Ylag(p+1:end,:);
t = t-p;

% Final VAR matrices/vectors
% 1/ VAR matrices for traditional matrix form
X = [ones(t,1) Ylag];
Y = Y(p+1:end,:);
    
% This is just to get index_cov which helps me construct the covariance
% matrix later using the vector of coefficient draws
COVM = randn(M,M);
COVM = COVM'*COVM;
COVM = chol(COVM)';
COVM = COVM/diag(diag(COVM));
index_cov = find(~tril(COVM));