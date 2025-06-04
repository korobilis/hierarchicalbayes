% AUTO_BVAR.m   Fast implementation of Gibbs sampler for Bayesian Vector
%               Autoregressions, using automatic hierarchical shrinkage
%               priors
% 
% Written by Dimitris Korobilis, March 2020
% University of Glasgow

clear all;
% close all;
clc;

% Add path of data and functions
addpath('functions');
addpath('data');

%-------------------------------PRELIMINARIES--------------------------------------
ngibbs     = 1000;         % Number of Gibbs sampler iterations
nburn      = 0.2*ngibbs;   % Number of iterations to discard
p          = 2;            % p is number of lags in the VAR part                         

sv_reg = 1;             % 0: Constant variance; 1: Stochastic volatility

prior = 3;              % 1: Normal-iGamma prior (Student t)
                        % 2: SSVS with Normal(0,tau_0) and Normal-iGamma components
                        % 3: Horseshoe prior
                        % 4: Spike and Slab lasso
                        
nhor = 20;              % Horizon for impulse response analysis

%----------------------------- END OF PRELIMINARIES --------------------------------
tic;
%----------------------------------LOAD DATA----------------------------------------   
[Y,PHI,C,S] = simvardgpcon(200,40,2,2);

% load ydata.mat;
% dim = zeros(size(Y,2),1);
% for i=1:size(Y,2)
%    dim(i) = sum(isnan(Y(:,i))); 
% end
% Y = Y(max(dim)+1:end,:);
% Ydates = datesq(max(dim)+1:end);
% 
% Yraw = Y;
% for i = 1:size(Y,2)
%     Y(:,i) = transx(Yraw(:,i),tcode1(i));
%     %Y(:,i) = adjout(Yraw(:,i),4.5,4);
% end
% Y = Y(2:end,:); Ydates = Ydates(2:end,:); Ydates = Ydates(p+1:end);

% ===================================| VAR EQUATION |==============================
[y,X,T,M,KK,K,~,~] = prepare_BVAR_matrices(Y,p);

est_meth = 2*(T>KK) + 1*(T<=KK);
% ==============| Define priors
Q   = cell(M,1);   miu = cell(M,1); lambda0 = cell(M,1); lambda1 = cell(M,1); tau = zeros(M,1);
b0 = 0; c0 = 0; pi0 = 0;
for ieq = 1:M
    Q{ieq,1}    = .01*ones(KK+ieq-1,1);
    miu{ieq,1}  = zeros(KK+ieq-1,1);
    if prior == 1
        % Student_T shrinkage prior
        b0 = 0.01;
    elseif prior == 2     
        % SSVS prior
        c0 = (0.01)^2;
        lambda0{ieq,1} = 4*ones(KK+ieq-1,1);    % "local" shrinkage parameters
        pi0 = 0.25;                             % "global" shrinkage parameter
        b0 = 0.01;
    elseif prior == 3
        % Horseshoe prior
        lambda0{ieq,1} = 0.1*ones(KK+ieq-1,1); % "local" shrinkage parameters
        tau(ieq,1) = 0.1*ones(1,1);            % "global" shrinkage parameter
    elseif prior == 4 
        % Spike and slab lasso
        lambda0{ieq,1} = 0.1*ones(KK+ieq-1,1);  % "local" shrinkage parameters
        lambda1{ieq,1} = 4*ones(KK+ieq-1,1);
        pi0 = 0.25;
    end
end

% Initialize matrices
sigma_t = 0.1*ones(T,M);
h = ones(T,M);   
sig = 0.1*ones(M,1);

%========|STORAGE MATRICES for MCMC:
beta_save  = zeros(ngibbs,KK,M);
A_save = zeros(ngibbs,M,M);
OMEGA_save = zeros(ngibbs,M,M,T);
irf_save = zeros(ngibbs,nhor,M,M);
%======================= BEGIN MCMC ESTIMATION =======================
tic;
for iter = 1: (ngibbs + nburn)
    if mod(iter,50)==0   
        disp(['This is iteration ' num2str(iter)])         
        disp([num2str(100*(iter/(ngibbs+nburn))) '% completed'])
        toc
    end
    
    %% Draw VAR equation-by-equation
    [beta,BETA,A,sigma_t,h,miu,Q,lambda0,lambda1,tau] = drawVAR(y,X,T,M,KK,sigma_t,h,sig,miu,Q,b0,c0,pi0,lambda0,lambda1,tau,prior,est_meth,sv_reg);
    
    % Check for stationarity
    B = [BETA(2:end,:)'; eye(M*(p-1)) , zeros(M*(p-1),M)];
    rej_count = 0;
    while max(abs(eig(B))) > 0.999
        rej_count = rej_count + 1;
        if rej_count > 10; break; end
        [beta,BETA,A,sigma_t,h,miu,Q,lambda0,lambda1,tau] = drawVAR(y,X,T,M,KK,sigma_t,h,sig,miu,Q,b0,c0,pi0,lambda0,lambda1,tau,prior,est_meth,sv_reg);
        B = [BETA(2:end,:)'; eye(M*(p-1)) , zeros(M*(p-1),M)];
    end

    % assign covariance matrix
    OMEGA = zeros(M,M,T);
    for t = 1:T 
        OMEGA(:,:,t) = A*diag(sigma_t(t,:))*A';
    end
    
    % Post burnin storage
    if iter > nburn
        % Save draws
        beta_save(iter-nburn,:,:) = BETA;
        A_save(iter-nburn,:,:) = A;
        OMEGA_save(iter-nburn,:,:,:) = OMEGA;
        
        % Save GIRFs
        ar_lags = BETA(2:end,:)';
        ar0 = {ar_lags(:,1:M)};
        if p>1       
            for i = 2:p
                ar0 = [ar0 ar_lags(:,(i-1)*M+1:i*M)];
            end
        end
        [irf] = armairf(ar0,[],'InnovCov',squeeze(OMEGA(:,:,end)),'Method','generalized','NumObs',nhor);
        irf_save(iter-nburn,:,:,:) = irf;
    end
end

% % Take posterior mean and quantiles of GIRFs
% IRF = quantile(irf_save,[0.14, 0.5, 0.86],1);
% 
% %% PLOT GIRF-based shocks to GDP
% ticks=0:40:T; ticks(1) = 1; 
% ticklabels = Ydates(ticks);
% 
% figure1 = figure('Color',[1 1 1]); 
% set(gcf,'units','normalized','outerposition',[0 0 1 1]);
% for i = 1:M
%     subplot(7,7,i)
%     shadedplot(1:nhor,IRF(1,:,i,1),IRF(3,:,i,1),[0.7,0.7,0.7],[0.7,0.7,0.7])
%     hold all        
%     plot(squeeze(IRF(2,:,i,1))','black','Linewidth',2)
%     xlim([1 nhor])
%     grid on
%     title(['Response of ' cell2mat(names(i)) ' to ' cell2mat(names(1))])
% end
% saveas(gcf, fullfile('GIRF_GDP'), 'fig')
% saveas(gcf, fullfile('GIRF_GDP'), 'eps')

toc;