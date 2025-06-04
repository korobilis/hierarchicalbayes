function [Trend,Cyclical] = hpfilter(Y,smoothing)
%HPFILTER Hodrick-Prescott filter for trend and cyclical components
%
% Syntax:
%
%	[Trend,Cyclical] = hpfilter(Y)
%	[Trend,Cyclical] = hpfilter(Y,smoothing)
%	hpfilter(...)
%
% Description:
%
%   Separate one or more time series into trend and cyclical components
%   with the Hodrick-Prescott filter. If no output arguments are specified,
%   HPFILTER displays a plot of the series and trend (with cycles removed).
%   The plot can be used to help select a smoothing parameter.
%
% Input Arguments:
%
%	Y - Time series data. Y may be a vector or a matrix. If Y is a vector,
%	  it represents a single series. If Y is a numObs-by-numSeries matrix,
%	  it represents numObs observations of numSeries series, with
%	  observations across any row assumed to occur at the same time. The
%	  last observation of any series is assumed to be the most recent.
%
% Optional Input Argument:
%
%	smoothing - Either a scalar to be applied to all series or a vector of
%	    length numSeries with values to be applied to corresponding series.
%	    The default is 1600, which is suggested	in [1] for quarterly data.
%	    If smoothing is 0, no smoothing occurs.	As the smoothing parameter
%	    increases, the smoothed series approaches a straight line. If
%	    smoothing is Inf, the series is detrended.
%
% Output Arguments:
%
%	Trend - Trend component of Y, the same size as Y.
%	Cyclical - Cyclical component of Y, the same size as Y.
%
% Notes:
%
%	o The Hodrick-Prescott filter separates a time series into a trend
%	  component and a cyclical component such that
%
%		Y = Trend + Cyclical
%
%	  The filter is equivalent to a cubic spline smoother, where the
%	  smoothed portion is in Trend.
%
%	o [1] suggests values for the smoothing parameter that depend upon
%	  the periodicity of the data:
%
%		Periodicity     smoothing
%       -----------     ---------
%		Yearly			100
%		Quarterly		1600
%		Monthly			14400
%
%   o The Hodrick-Prescott filter can produce anomalous endpoint effects in
%     very high-frequency data and should never be used for extrapolation.
%
% Reference:
%
%	[1] Hodrick, R. J., and E. C. Prescott. "Postwar U.S. Business Cycles:
%		An Empirical Investigation." Journal of Money, Credit, and Banking.
%		Vol. 29, 1997, pp. 1-16.

% Copyright 2006-2010 The MathWorks, Inc.

% Check input arguments:

if nargin < 1 || isempty(Y)
    
	error(message('econ:hpfilter:MissingInputData'))
      
end

if ~isscalar(Y) && isvector(Y) && isa(Y,'double')
    
	Y = Y(:);
	[numObs,numSeries] = size(Y);
    
elseif ndims(Y) == 2 && min(size(Y)) > 1 && isa(Y,'double')
    
	[numObs,numSeries] = size(Y);
    
else
    
	error(message('econ:hpfilter:InvalidInputArg1'))
    
end

if any(any(~isfinite(Y)))
    
	error(message('econ:hpfilter:InvalidInputData'))
    
end

if numObs < 3 % Treat samples with < 3 observations as trend data only
    
	warning(message('econ:hpfilter:InsufficientData'))
	Trend = Y;
	Cyclical = zeros(numObs,numSeries);
	return
    
end

if nargin < 2 || isempty(smoothing)
    
	warning(message('econ:hpfilter:DefaultQuarterlySmoothing'))
	smoothing = 1600;
    
end

if ~isvector(smoothing) || ~isa(smoothing,'double')
    
	error(message('econ:hpfilter:InvalidInputArg2'))
    
else
	if ~any(numel(smoothing) == [1,numSeries])
        
		error(message('econ:hpfilter:InconsistentSmoothingDimensions'))
        
	end
end

if any(isnan(smoothing))
    
	error(message('econ:hpfilter:InvalidSmoothing'))
    
end

if any(smoothing < 0)
    
	warning(message('econ:hpfilter:NegativeSmoothing'))
	smoothing = abs(smoothing);
    
end

% Run the filter with either scalar or vector smoothing:

if (numel(smoothing) == 1) || (max(smoothing) == min(smoothing)) % Scalar smoothing
    
    if numel(smoothing) > 1
		smoothing = smoothing(1);
    end
    
	if isinf(smoothing)	% Use OLS detrending
        
		Trend = Y-detrend(Y);
        
	else
        
		if numObs == 3 % Special case with 3 samples
            
			A = eye(numObs,numObs) + ...
				smoothing*[ 1 -2 1; -2 4 -2; 1 -2 1 ];
            
        else % General case with > 3 samples
            
			e = repmat([smoothing,-4*smoothing,(1+6*smoothing),...
				        -4*smoothing,smoothing],numObs,1);
			A = spdiags(e,-2:2,numObs,numObs);
			A(1,1) = 1+smoothing;
			A(1,2) = -2*smoothing;
			A(2,1) = -2*smoothing;
			A(2,2) = 1+5*smoothing;
			A(numObs-1,numObs-1) = 1+5*smoothing;
			A(numObs-1,numObs) = -2*smoothing;
			A(numObs,numObs-1) = -2*smoothing;
			A(numObs,numObs) = 1+smoothing;
            
		end
        
		Trend = A\Y;
        
	end
    
else % Vector smoothing
    
	Trend = zeros(numObs,numSeries);
    
	if numObs == 3 % Special case with 3 samples
        
		for i = 1:numSeries
            
            if isinf(smoothing(i)) % Use OLS detrending
                
				Trend(:,i) = Y(:,i)-detrend(Y(:,i));
                
            else
                
				A = eye(numObs,numObs) + ...
					smoothing(i)*[ 1 -2 1; -2 4 -2; 1 -2 1 ];
				Trend(:,i) = A\Y(:,i);
                
            end
            
		end
        
	else % General case with > 3 samples
        
        for i = 1:numSeries
            
            if isinf(smoothing(i)) % Use OLS detrending
                
				Trend(:,i) = Y(:,i)-detrend(Y(:,i));
                
            else
                
				e = repmat([smoothing(i),-4*smoothing(i),(1+6*smoothing(i)), ...
					        -4*smoothing(i),smoothing(i)],numObs,1);
				A = spdiags(e,-2:2,numObs,numObs);
				A(1,1) = 1+smoothing(i);
				A(1,2) = -2*smoothing(i);
				A(2,1) = -2*smoothing(i);
				A(2,2) = 1+5*smoothing(i);
				A(numObs-1,numObs-1) = 1+5*smoothing(i);
				A(numObs-1,numObs) = -2*smoothing(i);
				A(numObs,numObs-1) = -2*smoothing(i);
				A(numObs,numObs) = 1+smoothing(i);
				Trend(:,i) = A\Y(:,i);
                
            end
            
        end
        
	end
    
end

% If no output arguments, plot the results:

if nargout == 0
    
	figure(gcf);
	plot(Y,'b');
	hold on
	plot(Trend,'r');
	title('\bfHodrick-Prescott Filter');
    
	if numSeries == 1
		legend('Raw Data','Smoothed Trend');
	end
    
	hold off;
    
elseif nargout > 1
    
	Cyclical = Y-Trend;
    
end