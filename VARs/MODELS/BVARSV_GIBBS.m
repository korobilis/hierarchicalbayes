function [Y_forc,BETA,SIGMA] = BVARSV_GIBBS(Y,p,h,ndraws,stdata)

ngibbs = 1000;
nburn = 0.1*ngibbs;

if stdata == 1
    [Y,mm,ss] = zscore(Y);
else
    mm = zeros(1,M);
    ss = ones(1,M);
end

% ===================================| VAR EQUATION |==============================
[y,X,M,T,KK,~,~] = prepare_BVAR_matrices(Y,p);
K = KK*M;

if T>KK
    est_meth = 2;
else
    est_meth = 1;
end

% OLS estimates:
%[beta_OLS,beta_OLS_vec,SIGMA_OLS] = getOLS(y,X,M,p);

% Initialize parameters
lambda = zeros(KK,M); tau = zeros(M,1); nu = zeros(KK,M); xi = zeros(M,1);
lambda_sq = cell(M,1); tau_sq = zeros(M,1); 
h = zeros(T,M); sigma_sq = zeros(T,M); sig = zeros(M,1); 
for ieq = 1:M
    lambda(:,ieq) = ones(KK,1);
    tau(ieq,1) = 1;
    nu(:,ieq) = ones(KK,1);
    xi(ieq,1) = 1;
    h(:,ieq) = ones(T,1);
    sigma_sq(:,ieq) = ones(T,1);
    sig(ieq) = 0.001;
    lambda_sq{ieq,1} = ones(ieq-1,1);   
    tau_sq(ieq,1) = 1;
end

y_til = y;
XX = X'*X;

A = rand(M,M);
A = triu(A,+1);
index_A = find(A~=0);

%========|STORAGE MATRICES:
beta_save  = zeros(ngibbs,KK,M);
OMEGA_save = zeros(ngibbs,M,M,T);
tic;
%======================= BEGIN MCMC ESTIMATION =======================
MCMCreps =(ngibbs + nburn);
for iter = 1:MCMCreps
%     if mod(iter,MCMCreps/10) == 0        
%         fprintf('%d %% \t',round(100*(iter/MCMCreps)));
%     end
    disp(iter); toc;
    % Draw coefficients BETA and variances sigma_sq
    BETA = zeros(KK,M); resid = zeros(T,M);
    for ieq = 1:M
        ys = y_til(:,ieq)./sqrt(sigma_sq(:,ieq));
        xs = X./sqrt(sigma_sq(:,ieq));
        xssx = xs'*xs;
        [BETA(:,ieq),lambda(:,ieq),tau(ieq),nu(:,ieq),xi(ieq),sigma_sq(:,ieq),h(:,ieq),sig(ieq)] = horseshoeSV(ys,xs,xssx,lambda(:,ieq),tau(ieq),nu(:,ieq),xi(ieq),h(:,ieq),sig(ieq),ieq,est_meth);
        resid(:,ieq) = (y(:,ieq) - X*BETA(:,ieq));%./sqrt(sigma_sq(:,ieq));
    end
    
%     % Check stationarity
%     B=[BETA(2:end,:)'; eye(M*(p-1)), zeros(M*(p-1),M)];  % Convert to companion form
%     counter = 0;
%     while max(abs(eig(B))) > 0.999
%         counter = counter + 1;
%         if counter>10 && iter>1; BETA = BETA_OLD; end     
%         BETA = zeros(KK,M);   % redraw if BETA not stationary
%         parfor ieq = 1:M       
%             [BETA(:,ieq),lambda(:,ieq),tau(ieq),nu(:,ieq),xi(ieq),sigma_sq(:,ieq),h(:,ieq),sig(ieq)] = horseshoeSV(ys,xs,xssx,lambda(:,ieq),tau(ieq),nu(:,ieq),xi(ieq),h(:,ieq),sig(ieq),ieq,est_meth);
%             resid(:,ieq) = (y(:,ieq) - X*BETA(:,ieq));%./sqrt(sigma_sq(:,ieq));
%         end  
%         B=[BETA(2:end,:)'; eye(M*(p-1)), zeros(M*(p-1),M)];  % convert to companion form
%     end
%     BETA_OLD = BETA;
    
    % Draw covariance elements A
    A = eye(M); y_til = y;
    for ieq = 2:M
        [A(ieq,:),y_til(:,ieq),lambda_sq{ieq,1},tau_sq(ieq)] = drawA(y,resid,lambda_sq{ieq,1},tau_sq(ieq),mean(sigma_sq(:,ieq)),ieq,M);
    end
    
    OMEGA = zeros(M,M,T);
    for t = 1:T 
        OMEGA(:,:,t) = A*diag(sigma_sq(t,:))*A';
    end
    
    if iter > nburn
        % Save draws
        beta_save(iter-nburn,:,:) = BETA;
        OMEGA_save(iter-nburn,:,:,:) = OMEGA;
    end
end

BETA = squeeze(median(beta_save));
SIGMA = squeeze(median(OMEGA_save(:,:,:,end)));

% Matrices in companion form   
By = [BETA(2:end,:)'; eye(M*(p-1)) , zeros(M*(p-1),M)];   
Sy = zeros(M*p,M*p);
Sy(1:M,1:M) = SIGMA;
miu = zeros(M*p,1);
miu(1:M,:) = BETA(1,:)';

% -------| STEP 3: Prediction   
Y_pred = zeros(ndraws,M,h); % Matrix to save prediction draws
    
% Now do prediction using standard formulas (see Lutkepohl, 2005)    
VAR_MEAN = 0;
VAR_VAR = 0;
    
X_FORE = [Y(end,:) X(end,2:M*(p-1)+1)];
BB = speye(M*p);
for ii = 1:h % not very efficient, By^(ii-1) can be defined once 
    VAR_MEAN =  VAR_MEAN + BB*miu;
    FORECASTS = VAR_MEAN + (BB*By)*X_FORE';
    if ndraws > 1
        VAR_VAR = VAR_VAR + BB*Sy*BB';
        Y_pred(:,:,ii) = (repmat(FORECASTS(1:M),1,ndraws) +  chol(VAR_VAR(1:M,1:M))'*randn(M,ndraws))';
    else
        Y_pred(:,:,ii) = repmat(FORECASTS(1:M),1,ndraws);
    end
    BB = BB*By;
end

% Store predictive draws/mean
Y_forc = repmat(mm,ndraws,1,h) + repmat(ss,ndraws,1,h).*Y_pred;
Y_forc = permute(Y_forc,[1 3 2]);

