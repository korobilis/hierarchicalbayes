% The first stage in HD coufounding
% Fit Lasso on the model of T on X 
% Use the sparsest model within one standard error of the minimum MSE
function activeX=HDCofoundingLasso(T,X)

%elseif  T_type=='binary'
% using 10-fold cross-validation 
%[B,FitInfo] = lassoglm(X,T,'binomial','CV',10);   
%elseif  T_type=='continuous'
[B,FitInfo] = lasso(X,T,'CV',10);
%end 
% the sparsest model within one standard error of the minimum MSE
idxLambda1SE = FitInfo.Index1SE;
activeX = (B(:,idxLambda1SE)~=0);
end 