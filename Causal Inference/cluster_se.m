function [se,vcluster,middle] = cluster_se(x,e,XpXinv,group,k)

n = size(e,1);
if nargin < 5 || isempty(k),
    k = size(XpXinv,1);
end
V = 0;
for i = 1:max(group),
    I = find(group == i);

    %generate the standard cluster matrix;
    V = V + (x(I,:)'*e(I,:))*(x(I,:)'*e(I,:))';
end
middle = V;

vcluster = ((n-1)/(n-k))*(max(group)/(max(group)-1))*XpXinv*V*XpXinv';

se = sqrt(diag(vcluster));
