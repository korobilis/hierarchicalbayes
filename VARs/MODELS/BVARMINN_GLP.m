function [Y_forc,B,SIGMA,res] = BVARMINN_GLP(data,p,delta,h,stdata,ndraws,algor)

if stdata == 1
    [data,mm,ss] = standard(data);
else
    mm = zeros(1,size(data,2));
    ss = ones(1,size(data,2));
end


% Run the Bayesian VAR
if algor == 1
    [res,Y,X,T,N] = bvarGLP(data,p,delta,'noc',0);
    B = res.postmax.betahat;
    SIGMA = res.postmax.sigmahat;
else
    [res,Y,X,T,N] = bvarGLP(data,p,delta,'noc',0,'mcmc',1,'Ndraws',1000,'Ndrawsdiscard',100,'MCMCfcast',0);
    B = mean(res.mcmc.beta,3);
    SIGMA = mean(res.mcmc.sigma,3);
end

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
