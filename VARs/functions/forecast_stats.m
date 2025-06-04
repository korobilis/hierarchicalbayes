function [fore,msfe,msfe_ALL,PL] = forecast_stats(Y,y_fore,h,series_to_eval,ndraws)

% Function that gives major forecast statistics
numSerToEv = length(series_to_eval);
fore     = zeros(ndraws,h,numSerToEv);
msfe     = zeros(h,numSerToEv);
msfe_ALL = zeros(h,size(Y,2));
PL       = zeros(h,numSerToEv);

for ii = 1:h  
    % Save forecasts
    fore(:,ii,:) = y_fore(:,ii,series_to_eval);
              
    yi = Y(ii,series_to_eval);
                   
    msfe(ii,:)     = (squeeze(mean(y_fore(:,ii,series_to_eval),1))' - yi).^2;
    msfe_ALL(ii,:) = (squeeze(mean(y_fore(:,ii,:),1))' - Y(ii,:)).^2;
               
    if ndraws > 1    
        for j = series_to_eval
            PL(ii,j) = ksdensity(squeeze(fore(:,ii,j)),yi(:,j));
        end
    else
        PL(ii,series_to_eval) = NaN;
    end
end