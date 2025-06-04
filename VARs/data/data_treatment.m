clear

[aq bq]=xlsread('data_nolinks','Quarterly')

datesq=bq(2:end,1); %Strings with dates
namesq=bq(1,2:end);
xq=aq;
yq=xq;


[am bm]=xlsread('data_nolinks','Monthly')

datesm=bm(2:end,1); %Strings with dates
namesm=bm(1,2:end);
xm=am;

%Connecting loans and lending rates to pre-2003 series
in1=find(ismember(namesm,'Lending rate to NFC')==1);
in1b=find(ismember(namesm,'LNFC_hist')==1);

t1=sum(isfinite(xm(:,in1b)));
xx(:,1)=[xm(1:t1,in1b);xm(t1+1:end,in1)];

in2=find(ismember(namesm,'Lending rate to Households')==1);
in2b=find(ismember(namesm,'LHH_hist')==1);

t2=sum(isfinite(xm(:,in2b)));
xx(:,2)=[xm(1:t1,in2b);xm(t2+1:end,in2)];

in3=find(ismember(namesm,'Loans to NFC')==1);
in3b=find(ismember(namesm,'NFC_hist')==1);

t3=sum(isfinite(xm(:,in3b)));

k3=xm(t3,in3)/xm(t3,in3b);

xx(:,3)=[xm(1:t3-1,in3b)*k3;xm(t3:end,in3)];

in4=find(ismember(namesm,'Loans to HH')==1);
in4b=find(ismember(namesm,'HH_hist')==1);

t4=sum(isfinite(xm(:,in4b)));

k4=xm(t4,in4)/xm(t4,in4b);

xx(:,4)=[xm(1:t4-1,in4b)*k4;xm(t4:end,in4)];

ym=xm;

ym(:,[in1 in2 in3 in4])=xx;

exc=[in1b in2b in3b in4b];

namesm(exc)=[];

ym(:,exc)=[];

for i=1:size(ym,1)/3
    yyq(i,:)=mean(ym(3*(i-1)+1:3*i,:));
end

names=[namesq namesm];
y=[yq(1:end-1,:) yyq]; %Exclude the most recent release of capacity utilization

%Transformation and spefification 

[a2 b2]=xlsread('data_nolinks','Specification')

sp=a2(:,1);
tr=a2(:,2);

yt=y.*NaN;

for s=1:size(tr,1)
    if tr(s)==0
        yt(:,s)=y(:,s);
    elseif tr(s)==1
        yt(:,s)=y(:,s)/100;
    elseif tr(s)==2
        yt(:,s)=4*log(y(:,s));
    end
end

yfin=yt(:,sp==1);
namesfin=names(:,sp==1);

        
    
