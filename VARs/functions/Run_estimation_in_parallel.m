function [Y_forc_mat,msfe,msfe_ALL,PL,B_mat,SIGMA_mat] = Run_estimation_in_parallel(Y,VAR_size,stdata,p,delta,theta,V_prior,RP_type,n_psi,apply_bcr,weight_scheme,cov_comp,sparsity,h,ndraws,series_to_eval,irep)

[~,M] = size(Y);
Y_forc_mat = NaN(h,size(Y,2),9);
B_mat      = NaN(size(Y,2)*p+1,size(Y,2),9);
SIGMA_mat  = NaN(size(Y,2),size(Y,2),9);

% =============================| ESTIMATION METHODS
% 1) BVAR with fast Gibbs sampling and Horseshoe prior
tic;
[bvarhors] = BVAR_GIBBS(Y,p,h,ndraws,stdata);
[~,msfe(:,:,1),msfe_ALL(:,:,1),PL(:,:,1)] = forecast_stats(Y(irep+1:irep+h,:),bvarhors,h,series_to_eval,ndraws);
Y_forc_mat(:,:,1) = squeeze(mean(bvarhors,1));
toc;

% 3) BVAR Minnesota with Banbura Giannone and Reichlin algorithm
tic;
% settings form BCVAR paper
grid = [0.5:.1:10 50 100];
lgrid = grid*sqrt(M*p);
bvarmin = BVAR_MINN(Y(1:irep,:),p,1,h,ndraws,lgrid,delta,stdata);
[~,msfe(:,:,2),msfe_ALL(:,:,2),PL(:,:,2)] = forecast_stats(Y(irep+1:irep+h,:),bvarmin,h,series_to_eval,ndraws);
Y_forc_mat(:,:,2) = squeeze(mean(bvarmin,1));
toc;


% % 2) BVAR with fast Gibbs sampling and spike and slab prior
% tic;
% [bvarskinny] = BVAR_skinnyGIBBS(Y,p,h,ndraws,stdata);
% [~,msfe(:,:,2),msfe_ALL(:,:,2),PL(:,:,2)] = forecast_stats(Y(irep+1:irep+h,:),bvarskinny,h,series_to_eval,ndraws);
% Y_forc_mat(:,:,2) = squeeze(mean(bvarskinny,1));
% toc;
% 
% % 3) BVAR Minnesota with Banbura Giannone and Reichlin algorithm
% tic;
% % settings form BCVAR paper
% grid = [0.5:.1:10 50 100];
% lgrid = grid*sqrt(M*p);
% bvarmin = BVAR_MINN(Y(1:irep,:),p,1,h,ndraws,lgrid,delta,stdata);
% [~,msfe(:,:,3),msfe_ALL(:,:,3),PL(:,:,3)] = forecast_stats(Y(irep+1:irep+h,:),bvarmin,h,series_to_eval,ndraws);
% Y_forc_mat(:,:,3) = squeeze(mean(bvarmin,1));
% toc;
% 
% % 4) BVAR Minnesota with Carriero Clark and Marcellino algorithm
% tic;
% lambda_ccm = 0.05^2; %As in CCM paper, see section 8.1 --- previously, we had lambda_ccm = 0.2^2;
% [minccm,B_mat(:,:,4),SIGMA_mat(:,:,4)] = BVARMINN_CCM(Y(1:irep,:),p,delta,lambda_ccm,theta,V_prior,h,ndraws,stdata);
% [~,msfe(:,:,4),msfe_ALL(:,:,4),PL(:,:,4)] = forecast_stats(Y(irep+1:irep+h,:),minccm,h,series_to_eval,ndraws);
% Y_forc_mat(:,:,4) = squeeze(mean(minccm,1));
% toc;
% 
% % 5) BVAR Minnesota with Giannone Lenza and Primiceri algorithm
% tic;
% [minglp,B_mat(:,:,5),SIGMA_mat(:,:,5),~] = BVARMINN_GLP(Y(1:irep,:),p,delta,h,stdata,ndraws,1);
% [~,msfe(:,:,5),msfe_ALL(:,:,5),PL(:,:,5)] = forecast_stats(Y(irep+1:irep+h,:),minglp,h,series_to_eval,ndraws);
% Y_forc_mat(:,:,5) = squeeze(mean(minglp,1));
% toc;
% 
% % 6) ***** DFM *****
% tic;
% maxfac = 2*round(sqrt(M));
% bvardfm = BDFM(Y(1:irep,:),p,1,h,ndraws,maxfac);
% [~,msfe(:,:,6),msfe_ALL(:,:,6),PL(:,:,6)] = forecast_stats(Y(irep+1:irep+h,:),bvardfm,h,series_to_eval,ndraws);
% Y_forc_mat(:,:,6) = squeeze(mean(bvardfm,1));
% toc;
% 
% % 7) ***** FAVAR *****
% if strcmp(VAR_size,'SMALL')
%     [bvarbnch,~,SIGMA_mat(:,:,7)] = BVAR_OLS(Y(1:irep,:),stdata,p,h,ndraws); 
%     [~,msfe(:,:,7),msfe_ALL(:,:,7),PL(:,:,7)] = forecast_stats(Y(irep+1:irep+h,:),bvarbnch,h,series_to_eval,ndraws);
%     Y_forc_mat(:,:,7) = squeeze(mean(bvarbnch,1));
% else
%     tic;
%     maxfac = round(sqrt(M));
%     bfavar1 = BFAVAR(Y(1:irep,:),p,1,h,ndraws,series_to_eval,maxfac);
%     [~,msfe(:,:,7),msfe_ALL(:,:,7),PL(:,:,7)] = forecast_stats(Y(irep+1:irep+h,:),bfavar1,h,series_to_eval,ndraws);
%     Y_forc_mat(:,:,7) = squeeze(mean(bfavar1,1));
%     toc;
% end
% 
% % 8) ***** 8-variable VAR(p)-OLS *****
% [bvarbnch,~,SIGMA_mat(series_to_eval,series_to_eval,8)] = BVAR_OLS(Y(1:irep,series_to_eval),stdata,p,h,ndraws);
% [~,msfe(:,:,8),msfe_ALL(:,series_to_eval,8),PL(:,series_to_eval,8)] = forecast_stats(Y(irep+1:irep+h,series_to_eval),bvarbnch,h,series_to_eval,ndraws);
% Y_forc_mat(:,series_to_eval,8) = squeeze(mean(bvarbnch,1));
% 
% % 9) ***** BCVAR *****
% bcvar = BCTRVAR_CONJ(Y(1:irep,:),p,1,h,ndraws,RP_type,n_psi,stdata,apply_bcr,weight_scheme,cov_comp,sparsity,series_to_eval);
% [~,msfe(:,:,9),msfe_ALL(:,:,9),PL(:,:,9)] = forecast_stats(Y(irep+1:irep+h,:),bcvar,h,series_to_eval,ndraws);
% Y_forc_mat(:,:,9) = squeeze(mean(bcvar,1));


%clc;
