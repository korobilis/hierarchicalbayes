function [Y_forc,B,SIGMA] = BVARMINN_CCM(data,p,delta,lambda,theta,V_prior,h,ndraws,stdata)
%% ************************************************************************************
% Triangular algorithm, Carriero Clark and Marcellino (2015), Large Vector Autoregressions
% with stochastic volatility and flexible priors.
% ************************************************************************************
% Model is:
%
%     Y(t) = Pi(L)Y(t-1) + v(t); Y(t) is Nx1;
%     v(t) = inv(A)*(LAMBDA(t)^0.5)*e(t); e(t) ~ N(0,I);
%                   _                                         _
%                  |    1          0       0       ...      0  |
%                  |  A(2,1)       1       0       ...      0  |
%         A =      |  A(3,1)     A(3,2)    1       ...      0  |
%                  |   ...        ...     ...      ...     ... |
%                  |_ A(N,1)      ...     ...   A(N,N-1)    1 _|
%
%    Lambda(t)^0.5 = diag[sqrt_h(1,t)  , .... , sqrt_h(N,t)];
%
%    ht=exp(Vol_states)
%    sqrt_ht=sqrt(ht)=sqrt(exp(Vol_states))=exp(Vol_states/2)
%    Vol_states=2*ln(sqrt_ht)
%
%    Vol_states(i,t)   = Vol_states(i,t-1) + eta(i,t),
%    eta(t) ~ N(0,PHI); with PHI full
% ------------------------------------------------------------------------------------

%% --------------------------- OPTIONS ----------------------------------
check_stationarity = 0;      % Truncate nonstationary draws? (1=yes)
M          = 1000;           % Final number of draws after burn in
burn_in    = 0.1*M;          % burn in
MCMCreps   = M + burn_in;    % total MCMC draws

%% --------------------------- PRIORS ----------------------------------
theta=[lambda theta V_prior 2];  % hyperparameters of Minnesota prior:
% [lambda1 lambda2 int lambda3], int is the
% prior on the intercept. lambda1, lambda2
% and lambda3 are as in equation (42) with
% lambda1 the overall shrinkage, lambda2 the
% cross srhinkage and lambda 3 the lag decay
% (quadratic if =2). Note lambda2~=1 implies
% the prior becomes asymmetric across eqation,
% so this would not be implementable in the
% standard conjugate setup.
Minn_pmean = delta;              % Prior mean of the 1-st own lag for each
% equation. For nonstationary variables, this
% is usually set to 1. For transformed
% stationary variables this is set to 0.


%% -------------------------Create data matrices-------------------------
% pointers
[T,N]=size(data);
if stdata == 1
    [data,mm,ss] = standard(data);
else
    mm = zeros(1,N);
    ss = ones(1,N);
end

% matrix X
lags=zeros(T,N*p);
for l=1:p; lags(p+1:T,(N*(l-1)+1):N*l)=data(p+1-l:T-l,1:N); end
X = [ones(T-p,1) lags(p+1:T,:)];
% trim Y
Y = data(p+1:end,:);
% update pointers
[T,K]=size(X);

%% -----------------Prior hyperparameters for bvar model

% Prior on conditional mean coefficients, use Minnesota setup
ARresid=[];
for i=1:N
    yt_0=[ones(T-1,1) Y(1:end-1,i)];
    yt_1=Y(2:end,i);
    ARresid(:,i)=yt_1-yt_0*(yt_0\yt_1);
end
AR_s2= diag(diag(ARresid'*ARresid))./(T-2);

Pi_pm=zeros(N*(K-1),1); Pi_pv=speye(N*(K-1)); co=0;
for i=1:N
    sigma_const(i)=AR_s2(i,i)*theta(3); % this sets the prior variance on the intercept
    for l=1:p;
        for j=1:N
            co=co+1;
            if (i==j)
                Pi_pv(co,co)=theta(1)/(l^theta(4)); % prior variance, own lags
                if l==1; Pi_pm(co)=Minn_pmean; end; % this sets the prior means for the first own lag coefficients.
            else
                Pi_pv(co,co)=(AR_s2(i,i)/AR_s2(j,j)*theta(1)*theta(2)/(l^theta(4))); % prior variance, cross-lags
            end
        end
    end
end

% Pai~N(vec(MU_pai),OMEGA_pai), equation 7.
OMEGA_pai   = diag(vec([sigma_const;reshape(diag(Pi_pv),K-1,N)])); % prior variance of Pai
MU_pai      = [zeros(1,N);reshape(Pi_pm,K-1,N)];                   % prior mean of Pai

% A~N(MU_A,inv(OMEGA_A_inv)), equation 8.
for i = 2:N;
    MU_A(1:i-1,i) = zeros(i-1,1);            %#ok<SAGROW> % prior mean of A
    OMEGA_A_inv(1:i-1,1:i-1,i) = 0*eye(i-1); %#ok<SAGROW> % prior precision of A
end;

%% >>>>>>>>>>>>>>>>>>>>>>>>>> Gibbs sampler <<<<<<<<<<<<<<<<<<<<<<<<<<<

% Storage arrays for posterior draws
PAI_all     = zeros(M,K,N);
SIGMA_all   = zeros(M,N,N);

% define some useful matrices out of the loop
PAI = zeros(K,N);                                         % pre-allocate space for PAI
comp=[eye(N*(p-1)),zeros(N*(p-1),N)];                     % companion form
iV=spdiags(1./diag(OMEGA_pai),0,N*K,N*K); iVb_prior=iV*vec(MU_pai);    % inverses of prior matrices

% initializations
A_         = eye(N);                                      % initialize A matrix
invA_      = inv(A_);                                     % initialize inv(A) matrix
PHI_       = 0.0001*eye(N);                               % Initialize PHI_, a draw from the covariance matrix W
sigma2     = ones(1,N);
%% start of MCMC loop

%tic;
for m = 1:MCMCreps ;
    %disp(m)
    if mod(m,MCMCreps/10) == 0;
        %disp(['percentage completed:' num2str(100*m/MCMCreps) '%']);
        fprintf('%d %% \t',round(100*(m/MCMCreps)));
        %toc;
    end
    
    %% STEP 2b: Draw from the conditional posterior of PAI, equation 10.
    stationary=0;
    while stationary==0
        % This is the only new step (triangular algorithm).
        PAI=triang(Y,X,N,K,T,invA_,sqrt(sigma2),iV,iVb_prior);
        if (check_stationarity==0 || max(abs(eig([PAI(2:K,:)' ; comp]))) < 1); stationary = 1; end;
    end
    RESID = Y - X*PAI; % compute the new residuals
    
    %% STEP 2c: Draw the covariances, equation 11.
    for ii = 2:N
        % weighted regression to get Z'Z and Z'z (in Cogley-Sargent 2005 notation)
        y_spread_adj=RESID(:,ii)./sqrt(sigma2(ii));
        X_spread_adj=[]; for vv=1:ii-1;  X_spread_adj=[X_spread_adj RESID(:,vv)./sqrt(sigma2(ii))]; end  %#ok<AGROW>
        ZZ=X_spread_adj'*X_spread_adj; Zz=X_spread_adj'*y_spread_adj;
        % computing posteriors moments
        Valpha_post = (ZZ + OMEGA_A_inv(1:ii-1,1:ii-1,ii))\eye(ii-1);
        alpha_post  = Valpha_post*(Zz + OMEGA_A_inv(1:ii-1,1:ii-1,ii)*MU_A(1:ii-1,ii));
        % draw and store
        alphadraw   = alpha_post+chol(Valpha_post,'lower')*randn(ii-1,1);
        A_(ii,1:ii-1)= -1*alphadraw';
    end
    invA_=A_\eye(N); % compute implied draw from A^-1, needed in step 2b.
    
    %% STEP 2d: Draw variances
    sse2 = (RESID*A_').^2;
    S1 = 0.01 + T;
    S2 = 0.01 + sum(sse2);
    sigma2 = 1./gamrnd(S1./2,2./S2);
    
    
    %SIGMA = invA_*diag(sigma2)*invA_;
    SIGMA = invA_*diag(sigma2)*invA_';
    
    %% Store the posterior draws
    if m > burn_in;
        PAI_all(m-burn_in,:,:)     = PAI;         SIGMA_all(m-burn_in,:,:) = SIGMA;
    end
    
end; %end of the Gibbs sampler

B = squeeze(mean(PAI_all,1)); SIGMA = squeeze(mean(SIGMA_all,1));

% Matrices in companion form
By = [B(2:end,:)'; eye(N*(p-1)) , zeros(N*(p-1),N)];
Sy = zeros(N*p,N*p);
Sy(1:N,1:N) = SIGMA;
miu = zeros(N*p,1);
miu(1:N,:) = B(1,:)';

% -------| STEP 3: Prediction
Y_pred = zeros(ndraws,N,h); % Matrix to save prediction draws

% Now do prediction using standard formulas (see Lutkepohl, 2005)
VAR_MEAN = 0;
VAR_VAR = 0;

X_FORE = [Y(end,:) X(end,2:N*(p-1)+1)];
BB = speye(N*p);
for ii = 1:h % not very efficient, By^(ii-1) can be defined once
    VAR_MEAN =  VAR_MEAN + BB*miu;
    FORECASTS = VAR_MEAN + (BB*By)*X_FORE';
    if ndraws > 1
        VAR_VAR = VAR_VAR + BB*Sy*BB';
        Y_pred(:,:,ii) = (repmat(FORECASTS(1:N),1,ndraws) +  chol(VAR_VAR(1:N,1:N))'*randn(N,ndraws))';
    else
        Y_pred(:,:,ii) = repmat(FORECASTS(1:N),1,ndraws);
    end
    BB = BB*By;
end

% Store predictive draws/mean
Y_forc = repmat(mm,ndraws,1,h) + repmat(ss,ndraws,1,h).*Y_pred;
Y_forc = permute(Y_forc,[1 3 2]);

