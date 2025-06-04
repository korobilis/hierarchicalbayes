function [y]=transx(x,tcode)
%    Transform x
%    Return Series with same dimension and corresponding dates
%    Missing values where not calculated
%    -- Tcodes:
%             1 Level
%             2 First Difference
%             3 Second Difference
%             4 Log-Level
%             5 Log-First-Difference
%             6 Log-Second-Difference
%             7 Annual log-Difference for Monthly data
%             8 Annual log-Difference for Quarterly data
%             9 Detrend Log Using 1-sided HP detrending for Monthly data
%            10 Detrend Log Using 1-sided HP detrending for Quarterly data
%            11 Detrend Log Using using Biweight Kernel Smoother for Monthly data
%            12 Detrend Log Using using Biweight Kernel Smoother for Quarterly data

%  Dimitris Korobilis, December 2018

small = 1.0e-10;
n=size(x,1);
y=zeros(n,1);        %storage space for y

if tcode == 1
    y=x;
elseif tcode == 2
    y(2:n)=diff(x);
elseif tcode == 3
    y(3:n)=diff(diff(x));
elseif tcode == 4
    if min(x) < small
        y = NaN; 
    end
    x=log(x);
    y=x;
elseif tcode == 5
    if min(x) < small
        y=NaN; 
    end
    y(2:n)=diff(log(x));
elseif tcode == 6
    if min(x) < small
        y=NaN; 
    end
    y(3:n)=diff(diff(log(x)));
elseif tcode == 7
    if min(x) < small
        y=NaN; 
    end
    y(13:n) = log(x(12+1:end,:)) - log(x(1:end-12,:));
elseif tcode == 8
    if min(x) < small
        y=NaN; 
    end
    y(13:n) = log(x(4+1:end,:)) - log(x(1:end-4,:));    
elseif tcode == 9
    if min(x) < small
        y=NaN; 
    end
    x = log(x);
    [Trend,~] = hpfilter(x,14400);
    y = x - Trend;
elseif tcode == 10
    if min(x) < small
        y=NaN; 
    end
    x=log(x);
    [Trend,~] = hpfilter(x,1600);
    y = x - Trend;
elseif tcode == 11
    if min(x) < small
        y=NaN; 
    end
    x=log(x);
    [Trend,~] = detrend_biweight(x,240);
    y = x - Trend;
elseif tcode == 12        
    if min(x) < small
        y=NaN; 
    end
    x=log(x);
    [Trend,~] = detrend_biweight(x,80);
    y = x - Trend;
else
    y=NaN;
end