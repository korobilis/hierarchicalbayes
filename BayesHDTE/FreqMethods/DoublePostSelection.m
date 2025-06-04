% June 10, 2021
% Kenichi Shimizu and Duong Trinh
% University of Glasgow

% This code implements 
% Double Post Selection approach proposed by
% Belloni etal (2014, Review of Economic Studies) 

% Codes are from the supp. folder of the journal website 

%Here X_i=[1, T_i, covariates_i'] 
function [EstBetaAll,se_TE]=DoublePostSelection(Y,X)
p=size(X,2)-2;
conf_lvl = 0.05; % Confidence level parameter for model selection penalty
psi = 0.75; % initial parameter for heteroskedastic estimation of loadings

%%%%%%% double selection
%%% 1.  Run D on X, select X1
%%% 2.  Run Y on X, select X2
%%% 3.  Run Y on D, X1 and X2.
[ betaLASSO_1, ~] = MC_TE_LassoHeteroskedastic_unpenalized( X(:,2), [ X(:,1) X(:,1) X(:,3:(p+2))], conf_lvl, psi, 5, 1 );
[ betaLASSO_2, ~] = MC_TE_LassoHeteroskedastic_unpenalized(Y, [ X(:,1) X(:,1) X(:,3:(p+2))],  conf_lvl, psi, 5, 1 );
                
IND_NEW = [];
for kk = 3 : 2+p 
   if ( abs(betaLASSO_1(kk)) > 1.0e-8 || abs(betaLASSO_2(kk)) > 1.0e-8 )
   IND_NEW = [ IND_NEW kk ]; %#ok<AGROW>
   end
end
NEWvecSECOND_9 = X(:, [ 1 2 IND_NEW ]);
[NEWpostSECOND_9, ~] = regress(Y, NEWvecSECOND_9);
%         NEWmatSECOND_9 = inv(NEWvecSECOND_9'*NEWvecSECOND_9);
%         seNEWSECOND_9 = sqrt(NEWmatSECOND_9(2,2));
%         sNEWSECOND_9 = 2 + max(size(IND_NEW));

[ se_hetNEWSECOND_9 ] = Heteroskedastic_se ( Y, X(:,2), X(:,[ 1 IND_NEW]), NEWpostSECOND_9(2) );
se_TE=se_hetNEWSECOND_9;%standard error for treatment effect 

EstBetaAll=zeros(p+2,1);
EstBetaAll(1:2)=NEWpostSECOND_9(1:2);
EstBetaAll(IND_NEW')=NEWpostSECOND_9(3:end);



end 

%
%

%
function [ se_het ] = Heteroskedastic_se ( Y, D, Z, alpha )

% ZtZinv = inv(Z'*Z);
% PZ = Z*ZtZinv*Z';

if ~isempty(Z)
    [ n dimZ] = size(Z);
    Dtilde = D - Z*(Z\D); %d~ = d-z*(z*'z*)^(-1)(z*'d)
    Ytilde = Y - Z*(Z\Y); %y~ = y-z*(z*'z*)^(-1)(z*'y)
else
    n = size(D,1);
    dimZ = 0;
    Dtilde = D;
    Ytilde = Y;
end

Etilde = Ytilde - Dtilde * alpha; %e~ = y~ - d~*alpha

P = (Dtilde.^2)/(Dtilde'*Dtilde);
Estar = Etilde./(1-P);

% se_het = sqrt( (n/(n-dimZ-1))* (Dtilde.^2)'*(Etilde.^2) / (Dtilde'*Dtilde)^2 ); %(n/(n-dim(z*)-1))*sum(d~_i2 e~_i2)/(sum(d~_i2))2
se_het = sqrt( (n/(n-dimZ-1))* ((n-1)/n)* ...
    ((Dtilde.^2)'*((Estar.^2)) / (Dtilde'*Dtilde)^2 ...
    - (1/n)*((Dtilde'*Estar)^2)/(Dtilde'*Dtilde)^2 ));

end

% Run Lasso without penalizing components in IND
% but also estimating sigma
%  psi is associated the initial penalty
function [ betahat, shat ] = MC_TE_LassoHeteroskedastic_unpenalized ( Y, X, conf_lvl, psi, MaxIt, IND )

[ NumRow, NumCol ] = size(X);
n = NumRow;
m = NumCol;

vv = zeros(NumCol,1);
XX = zeros(NumRow,NumCol);
for j = 1 : NumCol
    vv(j) = norm( X(:,j)/sqrt(n) );
    XX(:,j) = X(:,j) / vv(j) ;
end

                                 % matrix, variance, seed, simulations,
                                 % sample size, quanile
[ lambda ] = MC_TE_SimulateLambda ( XX, 1, 1, 2000, n, conf_lvl );
VecLAMBDA = lambda*ones(m,1);
if ( max(size(IND))>0)
    VecLAMBDA(IND) = 0*IND;
end
% Compute penalty loadings Appendix A
if (max(size(IND))>0)
    [betaIND, betaIND_INT] = regress(Y, XX(:,IND));
    hatError = (Y - XX(:,IND)*betaIND)*sqrt(n/(n-max(size(IND))));    
else
    hatError = (Y - mean(Y))*sqrt(n/(n-1));
end

Xsq = (XX).^2;
% Algorirhm 1 
VecLAMBDA = psi*lambda*sqrt(Xsq'*(hatError.^2)/n);%initial value 
if ( max(size(IND))>0)
        VecLAMBDA(IND) = 0*IND;
end
for K = 1 : MaxIt
    betahat =  LassoShootingVecLAMBDA(XX,Y,VecLAMBDA);%(1)
    shat = sum( ( abs(betahat) > 0 ) );

    [ beta2STEP, s2STEP, STDerror2STEP ] = MC_TE_PostEstimator ( Y, XX, betahat, 0, 0 );
    hatError = (Y - XX*beta2STEP)*sqrt(n/(n-s2STEP));
    
    VecLAMBDA = lambda*sqrt(Xsq'*(hatError.^2)/n);%(2)
    if ( max(size(IND))>0)
        VecLAMBDA(IND) = 0*IND;
    end
    
end
beta_L1 =  LassoShootingVecLAMBDA(XX,Y,VecLAMBDA);
betahat = beta_L1 ./ vv;
shat = sum( ( abs(betahat) > 0 ) );
end



function [w,wp,m] = LassoShootingVecLAMBDA(X, y, lambdaVec,varargin)
% This function computes the Least Squares parameters
% with a penalty on the L1-norm of the parameters
%
% Method used:
%   The Shooting method of [Fu, 1998]
%
% Modifications:
%   We precompute the Hessian diagonals, since they do not 
%   change between iterations
[maxIter,verbose,optTol,zeroThreshold] = process_options(varargin,'maxIter',10000,'verbose',0,'optTol',1e-5,'zeroThreshold',1e-4);
[n p] = size(X);

% Start from the Least Squares solution
%MM = eye(p);
%for j = 1 : p 
%    MM(j,j) = lambdaVec(j);
%end
%beta = pinv(X'*X + MM)*(X'*y);
beta = zeros(p,1);

% Start the log
w_old = beta;
k=1;
wp = beta;

if verbose==2
    fprintf('%10s %10s %15s %15s %15s\n','iter','shoots','n(w)','n(step)','f(w)');
end

m = 0;

XX2 = X'*X*2;
Xy2 = X'*y*2;
while m < maxIter
    
    
    
    beta_old = beta;
    for j = 1:p
        lambda = lambdaVec(j);
        % Compute the Shoot and Update the variable
        S0 = sum(XX2(j,:)*beta) - XX2(j,j)*beta(j) - Xy2(j);
        if S0 > lambda
            beta(j,1) = (lambda - S0)/XX2(j,j);
        elseif S0 < -lambda
            beta(j,1) = (-lambda - S0)/XX2(j,j);
        elseif abs(S0) <= lambda
            beta(j,1) = 0;
        end
        
    end
    
    m = m + 1;
    
    % Update the log
    if verbose==2
        fprintf('%10d %10d %15.2e %15.2e %15.2e\n',m,m*p,sum(abs(beta)),sum(abs(beta-w_old)),...
            sum((X*beta-y).^2)+lambdaVec'*abs(beta));
        w_old = beta;
        k=k+1;
        wp(:,k) = beta;
    end
    % Check termination
    if sum(abs(beta-beta_old)) < optTol
        break;
    end
    
    
end
if verbose
fprintf('Number of iterations: %d\nTotal Shoots: %d\n',m,m*p);
end
w = beta;
end

%  The lambda( 1- ALPHA | X ) is the 1-ALPHA quantile of
%  
%    || S ||_{\infty} = 2 max_{j <= p} | n^{1/2} E_n[ e_t x_{tj} / s_j ] |
%    s_j = sqrt{ E_n[ x_{tj}^2 ] }
%
%

function [ lambda_final ] = MC_TE_SimulateLambda ( X, sigma, seed, NumSim, n, ALPHA )

NumSim = max(NumSim, n);

[ Numrows, NumColumns ] = size( X );

NormXX = zeros(NumColumns,1);
lambda = zeros(NumSim,1);

for j = 1 : NumColumns
    NormXX(j) = norm(X(:,j)/sqrt(n));
end    


for k = 1 : NumSim  
    %randn('state',seed+k);
    error = sigma*randn(n,1);          
    lambda(k) = 2*max( abs(  ( X'*error )./NormXX ) );        
end

lambda_final = quantile(lambda, 1-ALPHA ); 

end 

function [b,bint,r,rint,stats] = regress(y,X,alpha)
%REGRESS Multiple linear regression using least squares.
%   B = REGRESS(Y,X) returns the vector B of regression coefficients in the
%   linear model Y = X*B.  X is an n-by-p design matrix, with rows
%   corresponding to observations and columns to predictor variables.  Y is
%   an n-by-1 vector of response observations.
%
%   [B,BINT] = REGRESS(Y,X) returns a matrix BINT of 95% confidence
%   intervals for B.
%
%   [B,BINT,R] = REGRESS(Y,X) returns a vector R of residuals.
%
%   [B,BINT,R,RINT] = REGRESS(Y,X) returns a matrix RINT of intervals that
%   can be used to diagnose outliers.  If RINT(i,:) does not contain zero,
%   then the i-th residual is larger than would be expected, at the 5%
%   significance level.  This is evidence that the I-th observation is an
%   outlier.
%
%   [B,BINT,R,RINT,STATS] = REGRESS(Y,X) returns a vector STATS containing, in
%   the following order, the R-square statistic, the F statistic and p value
%   for the full model, and an estimate of the error variance.
%
%   [...] = REGRESS(Y,X,ALPHA) uses a 100*(1-ALPHA)% confidence level to
%   compute BINT, and a (100*ALPHA)% significance level to compute RINT.
%
%   X should include a column of ones so that the model contains a constant
%   term.  The F statistic and p value are computed under the assumption
%   that the model contains a constant term, and they are not correct for
%   models without a constant.  The R-square value is one minus the ratio of
%   the error sum of squares to the total sum of squares.  This value can
%   be negative for models without a constant, which indicates that the
%   model is not appropriate for the data.
%
%   If columns of X are linearly dependent, REGRESS sets the maximum
%   possible number of elements of B to zero to obtain a "basic solution",
%   and returns zeros in elements of BINT corresponding to the zero
%   elements of B.
%
%   REGRESS treats NaNs in X or Y as missing values, and removes them.
%
%   See also LSCOV, POLYFIT, REGSTATS, ROBUSTFIT, STEPWISE.

%   References:
%      [1] Chatterjee, S. and A.S. Hadi (1986) "Influential Observations,
%          High Leverage Points, and Outliers in Linear Regression",
%          Statistical Science 1(3):379-416.
%      [2] Draper N. and H. Smith (1981) Applied Regression Analysis, 2nd
%          ed., Wiley.

%   Copyright 1993-2014 The MathWorks, Inc.


if  nargin < 2
    error(message('stats:regress:TooFewInputs'));
elseif nargin == 2
    alpha = 0.05;
end

% Check that matrix (X) and left hand side (y) have compatible dimensions
[n,ncolX] = size(X);
if ~isvector(y) || numel(y) ~= n
    error(message('stats:regress:InvalidData'));
end

% Remove missing values, if any
wasnan = (isnan(y) | any(isnan(X),2));
havenans = any(wasnan);
if havenans
   y(wasnan) = [];
   X(wasnan,:) = [];
   n = length(y);
end

% Use the rank-revealing QR to remove dependent columns of X.
[Q,R,perm] = qr(X,0);
if isempty(R)
    p = 0;
elseif isvector(R)
    p = double(abs(R(1))>0);
else
    p = sum(abs(diag(R)) > max(n,ncolX)*eps(R(1)));
end
if p < ncolX
    warning(message('stats:regress:RankDefDesignMat'));
    R = R(1:p,1:p);
    Q = Q(:,1:p);
    perm = perm(1:p);
end

% Compute the LS coefficients, filling in zeros in elements corresponding
% to rows of X that were thrown out.
b = zeros(ncolX,1);
b(perm) = R \ (Q'*y);

if nargout >= 2
    % Find a confidence interval for each component of x
    % Draper and Smith, equation 2.6.15, page 94
    RI = R\eye(p);
    nu = max(0,n-p);                % Residual degrees of freedom
    yhat = X*b;                     % Predicted responses at each data point.
    r = y-yhat;                     % Residuals.
    normr = norm(r);
    if nu ~= 0
        rmse = normr/sqrt(nu);      % Root mean square error.
        tval = tinv((1-alpha/2),nu);
    else
        rmse = NaN;
        tval = 0;
    end
    s2 = rmse^2;                    % Estimator of error variance.
    se = zeros(ncolX,1);
    se(perm,:) = rmse*sqrt(sum(abs(RI).^2,2));
    bint = [b-tval*se, b+tval*se];

    % Find the standard errors of the residuals.
    % Get the diagonal elements of the "Hat" matrix.
    % Calculate the variance estimate obtained by removing each case (i.e. sigmai)
    % see Chatterjee and Hadi p. 380 equation 14.
    if nargout >= 4
        hatdiag = sum(abs(Q).^2,2);
        ok = ((1-hatdiag) > sqrt(eps(class(hatdiag))));
        hatdiag(~ok) = 1;
        if nu > 1
            denom = (nu-1) .* (1-hatdiag);
            sigmai = zeros(length(denom),1);
            sigmai(ok) = sqrt(max(0,(nu*s2/(nu-1)) - (r(ok) .^2 ./ denom(ok))));
            ser = sqrt(1-hatdiag) .* sigmai;
            ser(~ok) = Inf;
            tval = tinv((1-alpha/2),nu-1); % see eq 2.26 Belsley et al. 1980
        elseif nu == 1
            ser = sqrt(1-hatdiag) .* rmse;
            ser(~ok) = Inf;
        else % if nu == 0
            ser = rmse*ones(length(y),1); % == Inf
        end

        % Create confidence intervals for residuals.
        rint = [(r-tval*ser) (r+tval*ser)];
    end

    % Calculate R-squared and the other statistics.
    if nargout == 5
        % There are several ways to compute R^2, all equivalent for a
        % linear model where X includes a constant term, but not equivalent
        % otherwise.  R^2 can be negative for models without an intercept.
        % This indicates that the model is inappropriate.
        SSE = normr.^2;              % Error sum of squares.
        RSS = norm(yhat-mean(y))^2;  % Regression sum of squares.
        TSS = norm(y-mean(y))^2;     % Total sum of squares.
        r2 = 1 - SSE/TSS;            % R-square statistic.
        if p > 1
            F = (RSS/(p-1))/s2;      % F statistic for regression
        else
            F = NaN;
        end
        prob = fpval(F,p-1,nu); % Significance probability for regression
        stats = [r2 F prob s2];

        % All that requires a constant.  Do we have one?
        if ~any(all(X==1,1))
            % Apparently not, but look for an implied constant.
            b0 = R\(Q'*ones(n,1));
            if (sum(abs(1-X(:,perm)*b0))>n*sqrt(eps(class(X))))
                warning(message('stats:regress:NoConst'));
            end
        end
    end

    % Restore NaN so inputs and outputs conform
    if havenans
        if nargout >= 3
            tmp = NaN(length(wasnan),1);
            tmp(~wasnan) = r;
            r = tmp;
            if nargout >= 4
                tmp = NaN(length(wasnan),2);
                tmp(~wasnan,:) = rint;
                rint = tmp;
            end
        end
    end

end % nargout >= 2


end 

%
%
%  Post estimator Trimms component j if \hat\sigma_j \hat\beta_j < eps
%
%  CompStdErr: is the component for which we want a std error
function [ betatilde, stilde, STDerror ] = MC_TE_PostEstimator ( Y, X, beta_FS, eps, CompStdErr )

[ n , p ] = size(X);

v = zeros(p,1);
for j = 1 : p 
    v(j) = norm( X(:,j) / sqrt(n) );
end

ind = ( abs(beta_FS.*v) > eps );
if( min(X(:,1))==max(X(:,1)) )
    ind(1) = 1;
end
XX = [];
ind_col = [];

for j = 1 : p 
    if ( ind(j) == 1 )
        XX = [ XX  X(:,j) ];
        ind_col = [ ind_col j ];
    end
end

if size(XX,2) > 1
    if sum(abs(XX(:,1)-XX(:,2))) == 0
        XX(:,2) = [];
        ind_col(2) = [];
    end
end

K = max(size(ind_col));
betatilde = zeros(p,1);

if     ( K >= n  )
    Minv = pinv(XX'*XX);
    
elseif ( K > 0  )
    
    if rank(XX) < size(XX,2),
        disp('hi there');
    end

    Minv = inv(XX'*XX);
        
end

if ( K > 0 )
    betatilde(ind_col) = Minv*XX'*Y;
end

stilde = K;

if (CompStdErr > 0 && CompStdErr <= p )
    STDerror = sqrt( Minv(CompStdErr,CompStdErr) );
else
    STDerror = 0;
end


end 


% PROCESS_OPTIONS - Processes options passed to a Matlab function.
%                   This function provides a simple means of
%                   parsing attribute-value options.  Each option is
%                   named by a unique string and is given a default
%                   value.
%
% Usage:  [var1, var2, ..., varn[, unused]] = ...
%           process_options(args, ...
%                           str1, def1, str2, def2, ..., strn, defn)
%
% Arguments:   
%            args            - a cell array of input arguments, such
%                              as that provided by VARARGIN.  Its contents
%                              should alternate between strings and
%                              values.
%            str1, ..., strn - Strings that are associated with a 
%                              particular variable
%            def1, ..., defn - Default values returned if no option
%                              is supplied
%
% Returns:
%            var1, ..., varn - values to be assigned to variables
%            unused          - an optional cell array of those 
%                              string-value pairs that were unused;
%                              if this is not supplied, then a
%                              warning will be issued for each
%                              option in args that lacked a match.
%
% Examples:
%
% Suppose we wish to define a Matlab function 'func' that has
% required parameters x and y, and optional arguments 'u' and 'v'.
% With the definition
%
%   function y = func(x, y, varargin)
%
%     [u, v] = process_options(varargin, 'u', 0, 'v', 1);
%
% calling func(0, 1, 'v', 2) will assign 0 to x, 1 to y, 0 to u, and 2
% to v.  The parameter names are insensitive to case; calling 
% func(0, 1, 'V', 2) has the same effect.  The function call
% 
%   func(0, 1, 'u', 5, 'z', 2);
%
% will result in u having the value 5 and v having value 1, but
% will issue a warning that the 'z' option has not been used.  On
% the other hand, if func is defined as
%
%   function y = func(x, y, varargin)
%
%     [u, v, unused_args] = process_options(varargin, 'u', 0, 'v', 1);
%
% then the call func(0, 1, 'u', 5, 'z', 2) will yield no warning,
% and unused_args will have the value {'z', 2}.  This behaviour is
% useful for functions with options that invoke other functions
% with options; all options can be passed to the outer function and
% its unprocessed arguments can be passed to the inner function.

% Copyright (C) 2002 Mark A. Paskin
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
% USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [varargout] = process_options(args, varargin)

% Check the number of input arguments
n = length(varargin);
if (mod(n, 2))
  error('Each option must be a string/value pair.');
end

% Check the number of supplied output arguments
if (nargout < (n / 2))
  error('Insufficient number of output arguments given');
elseif (nargout == (n / 2))
  warn = 1;
  nout = n / 2;
else
  warn = 0;
  nout = n / 2 + 1;
end

% Set outputs to be defaults
varargout = cell(1, nout);
for i=2:2:n
  varargout{i/2} = varargin{i};
end

% Now process all arguments
nunused = 0;
for i=1:2:length(args)
  found = 0;
  for j=1:2:n
    if strcmpi(args{i}, varargin{j})
      varargout{(j + 1)/2} = args{i + 1};
      found = 1;
      break;
    end
  end
  if (~found)
    if (warn)
      warning(sprintf('Option ''%s'' not used.', args{i}));
      args{i};
    else
      nunused = nunused + 1;
      unused{2 * nunused - 1} = args{i};
      unused{2 * nunused} = args{i + 1};
    end
  end
end

% Assign the unused arguments
if (~warn)
  if (nunused)
    varargout{nout} = unused;
  else
    varargout{nout} = cell(0);
  end
end




end










