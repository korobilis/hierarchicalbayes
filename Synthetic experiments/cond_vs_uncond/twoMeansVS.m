% 2-means post MCMC variable selection 
% Li and Pati (2017)
% INPUT 
% beta_save is a ( p by nsave) matrix obtained from MCMC 
% p is the number of 'groups' 


% OUTPUT
% beta_res is a (p by 1) sparse matrix of slopes after variable selection 


% With groupings, perform 2-mean algorithm on (d by 1) vectors 
function beta_res=twoMeansVS(beta_save)
[p,nsave]=size(beta_save);

% Perform a 2-means algorithm on abs(beta)
h=zeros(1,nsave);
for it=1:nsave
    [idx,Centroids]=kmeans(abs(beta_save(:,it)),2);
    if sum(idx==1)<sum(idx==2)
    h(it)=sum(idx==1);
    else
    h(it)=sum(idx==2);
    end 
end 
% Determine the number of non-zero signals  
H=mode(h);
% Take median 
beta_median=median(beta_save,2);
% Find the H largest values 
[sorted, sortedInds] = sort(mean(abs(beta_median),2),'descend');
topH = sortedInds(1:H);

beta_res=zeros(p,1);
beta_res(topH,:)=beta_median(topH,:);






end 


