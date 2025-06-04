function [se,vhetero,V] = hetero_se(x,e,XpXinv)

k = size(x,2);
n = size(x,1);

V = (x.*((e.^2)*ones(1,k)))'*x;

vhetero = (n/(n-k))*XpXinv*V*XpXinv';

se = sqrt(diag(vhetero));

