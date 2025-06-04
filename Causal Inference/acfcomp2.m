function a = acfcomp2(tseries,na)
% ACFCOMP2(TSERIES,NA) sample autocovariance function
% TSERIES must be a column vector, NA is number of autocorrelation to
% compute
% returns a column vector

n = length(tseries);
if nargin == 1 || isempty(na),
    na = ceil(10*log10(n));
end
a = zeros(na,1);
for j = 1:na
   a(j,1) = corr(tseries(1:end-(j-1),1),tseries(j:end,1));
end
