function [Y_trend, Y_cycle] = detrend_biweight(Y, bw)

    % Detrending by biweight kernel smoother

    T = size(Y,1); % Sample size
    
    kernel = @(x) max((1-x.^2).^2, 0); % Biweight kernel
    diff_matrix = bsxfun(@minus, (1:T)', 1:T)/bw; % Matrix with (i,j) element (i-j)/bw
    weights = kernel(diff_matrix); % Matrix of observation weights
    weights_norm = bsxfun(@rdivide, weights, sum(weights, 2)); % Normalize weights so they sum to 1 in each row
    
    Y_trend = weights_norm*Y; % Trend: biweight averages
    Y_cycle = Y - Y_trend; % Cycle: residual

end