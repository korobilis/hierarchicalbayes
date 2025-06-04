function [y,PHI,C,sigma] = simvardgpcon(T,N,L,DGP)
%--------------------------------------------------------------------------
%   PURPOSE:
%      Get matrix of Y generated from a VAR model
%--------------------------------------------------------------------------
%   INPUTS:
%     T     - Number of observations (rows of Y)
%     N     - Number of series (columns of Y)
%     L     - Number of lags
%
%   OUTPUT:
%     y     - [T x N] matrix generated from VAR(L) model
% -------------------------------------------------------------------------

randn('seed',sum(100*clock));
rand('seed',sum(100*clock));
%-----------------------PRELIMINARIES--------------------

if DGP == 1    % First simulation exercise
    if L~=1
        error('in this DGP we need only 1 lag by consruction')
    end
    
    PHI = [ones(1,N);0.9*eye(N)];

    PSI = eye(N);
    PSI(1,2:end) = 0.5;
    sigma = inv(PSI*PSI');
    C = chol(sigma);
elseif DGP == 2
    % Note: we allow AR(1) value to be random in the range (0.4, 0.6) and then set for a certain VAR equation to have
    % persistence: ar(1)/(1^2) + ar(1)/(2^2) + ar(1)/(3^2) + ar(1)/(4^2).... in the spirit of the Minnesota prior. For
    % ar(1) = 0.6 we get a total persistence of 0.8542 for a 4-lag specification (while asymptotically the persistence
    % converges to ~0.987). Notice that I allow the persistence to be random for each VAR equation
    % Regading correlations, I work with the matrix C where I randomly set some lower triangular elements to zero and
    % some to a value between (0,1). If we don't use SSVS in the comparison for this DGP, then I think it would be ok to
    % define things in terms of C.
    
    low_lim = 0.3;   high_lim = 0.5;
    lam     = (1./(N-1)); % this makes the prob of non-zero elements in the other lags a function of the VAR size; 
                          %  N=3 => lam=0.5; N=7 => lam=0.17; N=20 => lam=0.05; N=40 => lam=0.025; ....  
    sig2_sl = 0.1; % variance of non-zero coeffs in the other lags
    
    stable_var = 0;
    while stable_var == 0
        persistence_level = (high_lim - low_lim)*rand(N,1) + low_lim;
        correlation_level = rand(N*(N-1)/2,1);
        
        % Constant
        PHI = ones(1,N);
        
        % Fill in own lags
        for i = 1:L
            PHI = [PHI; (diag(persistence_level)./(i^2))]; %#ok<AGROW>
        end
        
        % Fill in other lags
        PHI_long = PHI(:);
        indx_other = PHI_long==0;
        PHI_long(indx_other) = double(rand(sum(indx_other),1)<lam) .* (sqrt(sig2_sl)*randn(sum(indx_other),1));
        PHI = reshape(PHI_long,N*L+1,N);
%         % Correlation and Covariance matrices
%         correlation_level(abs(correlation_level) < 0.25) = 0;
%         c = correlation_level;%.*round(rand(N*(N-1)/2,1));
%         C = tril(ones(N),-1);
%         C(C==1) = c;
%         C = C + eye(N);
        
        C = zeros(N,N);
        C(2:end,1) = 2*rand(N-1,1) - 1;
        C = C  + eye(N);

        S = diag(rand(N,1));
        sigma = C*S*C';
        
        % Check whether VAR is stable, writing VAR in companion form
        By = [PHI(2:end,:)'; eye(N*(L-1)) , zeros(N*(L-1),N)];
        if max(abs(eig(By)))>=1
            disp('Non stationary VAR - redrawing now...')
            stable_var = 0;
        else
            stable_var = 1;
        end
    end

end

%----------------------GENERATE--------------------------
% Set storage in memory for y
% First L rows are created randomly and are used as
% starting (initial) values
y =[0*rand(L,N) ; zeros(T-L+100,N)];

% Now generate Y from VAR (L,PHI,PSI)
for nn = L:T+100
    u = chol(sigma)'*randn(N,1);
    ylag = mlag2(y,L);
    y(nn,:) = [1 ylag(nn,:)]*PHI + u';
end
y = y(end-T+1:end,:);