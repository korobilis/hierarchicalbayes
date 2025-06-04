function [Y,series,tcode,fcode] = load_data(VAR_size)

SMALL  = [1, 21, 33, 109, 110, 132, 157, 228];
MEDIUM = [SMALL, 10, 18, 32, 57, 68, 70, 72, 73, 75, 82, 208, 84, 88, 112, 122, 136, 138, 189, 203, 149, 165, 160];
LARGE  = 1:231; LARGE(SMALL) = []; LARGE = [SMALL, LARGE];
XLARGE = [LARGE, 423:446];
HUGE   = [XLARGE, 385:422];
UHD    = [HUGE, 232:384];

%% Read in the data
[A,B]  = xlsread('Combined.xlsx');
data   = A(7:end,:); % take data from 1960:Q1 to 2018:Q4 
tcode  = A(2,:); 
fcode  = A(1,:);
series = B(1,2:end);
dates  = B(4:end,1);

[~,M] = size(data);

% Transform variables using tcodes
Y = NaN*data;
for ivar = 1:M
    Y(:,ivar) = transxf(data(:,ivar),tcode(ivar));
    Y(:,ivar) = adjout(Y(:,ivar),4.5,4);
end

eval(['Y = Y(3:end,',VAR_size,');']);
eval(['series = series(',VAR_size,');']);
eval(['tcode = tcode(',VAR_size,');']);
eval(['fcode = fcode(',VAR_size,');']);
dates = dates(3:end);