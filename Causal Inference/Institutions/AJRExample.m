clear ;

%% Import data

data = dlmread('acemoglu_col_notext.txt','\t',1,0);
GDP = data(:,1);
Exprop = data(:,2);	
Mort = log(data(:,3));
Latitude = data(:,4);
Neo = data(:,5);
Africa = data(:,6);
Asia = data(:,7);
Namer = data(:,8);
Samer = data(:,9);

x = [Africa Asia Namer Samer Latitude Latitude.^2 Latitude.^3 ...
    (Latitude-.08).*((Latitude-.08) > 0) (Latitude-.16).*((Latitude-.16) > 0) ...
    (Latitude-.24).*((Latitude-.24) > 0) ((Latitude-.08).*((Latitude-.08) > 0)).^2 ...
    ((Latitude-.16).*((Latitude-.16) > 0)).^2 ((Latitude-.24).*((Latitude-.24) > 0)).^2 ...
    ((Latitude-.08).*((Latitude-.08) > 0)).^3, ((Latitude-.16).*((Latitude-.16) > 0)).^3 ...
    ((Latitude-.24).*((Latitude-.24) > 0)).^3  ];

%% Baseline with just latitude
z_B = [Mort Latitude ones(size(Mort))];
FS_B = z_B\Exprop;
SEF_B = hetero_se(z_B,Exprop-z_B*FS_B,inv(z_B'*z_B));

x_B = [Exprop Latitude ones(size(Mort))];
SS_B = (z_B'*x_B)\(z_B'*GDP);
SES_B = hetero_se(z_B,GDP-x_B*SS_B,inv(z_B'*x_B));

%% All Controls
z_All = [Mort x ones(size(Mort))];
FS_All = z_All\Exprop;
SEF_All = hetero_se(z_All,Exprop-z_All*FS_All,inv(z_All'*z_All));

x_All = [Exprop x ones(size(Mort))];
SS_All = (z_All'*x_All)\(z_All'*GDP);
SES_All = hetero_se(z_All,GDP-x_All*SS_All,inv(z_All'*x_All));

%% Variable Selection
[n,p] = size(x);
gamma = .1/log(n);
lambda = 1.1*2*sqrt(2*n*(log(2*(p/gamma))));

My = GDP - mean(GDP);
Mz = Mort - mean(Mort);
Md = Exprop - mean(Exprop);
Mx = x - ones(n,1)*mean(x);

PiMy = feasiblePostLasso(My,Mx,'lambda',lambda,'MaxIter',100);
PiMd = feasiblePostLasso(Md,Mx,'lambda',lambda,'MaxIter',100);
PiMz = feasiblePostLasso(Mz,Mx,'lambda',lambda,'MaxIter',100);

IndMy = abs(PiMy) > 0;
IndMd = abs(PiMd) > 0;
IndMz = abs(PiMz) > 0;

IndX = max([IndMy,IndMd,IndMz],[],2);

xSel = x(:,IndX);

z_Sel = [Mort xSel ones(size(Mort))];
FS_Sel = z_Sel\Exprop;
SEF_Sel = hetero_se(z_Sel,Exprop-z_Sel*FS_Sel,inv(z_Sel'*z_Sel));

x_Sel = [Exprop xSel ones(size(Mort))];
SS_Sel = (z_Sel'*x_Sel)\(z_Sel'*GDP);
SES_Sel = hetero_se(z_Sel,GDP-x_Sel*SS_Sel,inv(z_Sel'*x_Sel));


%% Cross-validation
rng(666333);
group = rand(n,1);
groupprc = [0,prctile(group,10:10:90),1];
nC = 10;
lambdaGrid = (20:70)';
nG = size(lambdaGrid,1);

y = GDP;
d = Exprop;
z = Mort;

CVy = zeros(nG,nC);
CVd = zeros(nG,nC);
CVz = zeros(nG,nC);
for ii = 1:nG
    disp(lambdaGrid(ii));
    for jj = 1:nC
        pred = (group > groupprc(jj) & group <= groupprc(jj+1));
        est = logical(1-pred);
        xjj = x(est,:);
        zjj = z(est,:);
        djj = d(est,:);
        yjj = y(est,:);
        Mxjj = xjj - ones(size(xjj,1),1)*mean(xjj);
        Mzjj = zjj - mean(zjj);
        Mdjj = djj - mean(djj);
        Myjj = yjj - mean(yjj);
        piZjj = feasiblePostLasso(Mzjj,Mxjj,'lambda',lambdaGrid(ii),'MaxIter',100);
        piDjj = feasiblePostLasso(Mdjj,Mxjj,'lambda',lambdaGrid(ii),'MaxIter',100);
        piYjj = feasiblePostLasso(Myjj,Mxjj,'lambda',lambdaGrid(ii),'MaxIter',100);
        indZjj = abs(piZjj) > 0;
        indDjj = abs(piDjj) > 0;
        indYjj = abs(piYjj) > 0;
        xp = x(pred,:);
        ip = ones(size(x(pred,:),1),1);
        ij = ones(size(x(est,:),1),1);
        zhat = [xp(:,indZjj) ip]*([xjj(:,indZjj) ij]\z(est,:));
        dhat = [xp(:,indDjj) ip]*([xjj(:,indDjj) ij]\d(est,:));
        yhat = [xp(:,indYjj) ip]*([xjj(:,indYjj) ij]\y(est,:));
        CVy(ii,jj) = sum((y(pred,:)-yhat).^2);
        CVd(ii,jj) = sum((d(pred,:)-dhat).^2);
        CVz(ii,jj) = sum((z(pred,:)-zhat).^2);        
    end
end

meanCVy = mean(CVy,2);
meanCVd = mean(CVd,2);
meanCVz = mean(CVz,2);

stdCVy = std(CVy,[],2);
stdCVd = std(CVd,[],2);
stdCVz = std(CVz,[],2);

c68 = tinv(1-.16,9);

mCVy = meanCVy == min(meanCVy);
mCVd = meanCVd == min(meanCVd);
mCVz = meanCVz == min(meanCVz);

by = max(meanCVy(mCVy) + c68*stdCVy(mCVy));
bd = max(meanCVd(mCVd) + c68*stdCVd(mCVd));
bz = max(meanCVz(mCVz) + c68*stdCVz(mCVz));

lambdaCVy = sqrt(10/9)*max(lambdaGrid(meanCVy < by));
lambdaCVd = sqrt(10/9)*max(lambdaGrid(meanCVd < bd));
lambdaCVz = sqrt(10/9)*max(lambdaGrid(meanCVz < bz));

PiMyCV = feasiblePostLasso(My,Mx,'lambda',lambdaCVy,'MaxIter',100);
PiMdCV = feasiblePostLasso(Md,Mx,'lambda',lambdaCVd,'MaxIter',100);
PiMzCV = feasiblePostLasso(Mz,Mx,'lambda',lambdaCVz,'MaxIter',100);

IndMyCV = abs(PiMyCV) > 0;
IndMdCV = abs(PiMdCV) > 0;
IndMzCV = abs(PiMzCV) > 0;

IndXCV = max([IndMyCV,IndMdCV,IndMzCV],[],2);

xCV = x(:,IndXCV);

z_CV = [Mort xCV ones(size(Mort))];
FS_CV = z_CV\Exprop;
SEF_CV = hetero_se(z_CV,Exprop-z_CV*FS_CV,inv(z_CV'*z_CV));

x_CV = [Exprop xCV ones(size(Mort))];
SS_CV = (z_CV'*x_CV)\(z_CV'*GDP);
SES_CV = hetero_se(z_CV,GDP-x_CV*SS_CV,inv(z_CV'*x_CV));



%% Display Results

disp('First Stage');
disp([FS_B(1) FS_All(1) FS_Sel(1) FS_CV(1)]);
disp([SEF_B(1) SEF_All(1) SEF_Sel(1) SEF_CV(1)]);

disp('Second Stage');
disp([SS_B(1) SS_All(1) SS_Sel(1) SS_CV(1)]);
disp([SES_B(1) SES_All(1) SES_Sel(1) SES_CV(1)]);

