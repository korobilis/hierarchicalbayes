function [atilda] = DK2002(y,bigG,W,H,Hinv)

% Durbin and Koopman (2002).  A simple and efficient simulation smoother for 
%                             state space time series analysis, Biometrika,
%                             89(3), 603-615.

[T,Tp] = size(bigG);

%% Step 1: Draw w+, a+ and y+
wplus = sqrt(W).*randn(Tp,1);
aplus = Hinv*wplus;
yplus = bigG*aplus + randn(T,1);

%% Step 2: Estimate a^ = E(w|y) and a^+ = E(w+|y+)
% C = speye(T);
% U = bigG';
% DU = bsxfun(@times,W,U);
% G = chol(C + bigG*DU)';
% invPX = DU/G'/G; % Shearle identity

D = H*sparse(1:Tp,1:Tp,1./W)*H';
P = D + bigG'*bigG;
C = chol(P);
ahat =   C\(C'\(bigG'*y));
ahatp =  C\(C'\(bigG'*yplus));

%% Step 3: Get sample of alpha
atilda = ahat - ahatp + aplus;
