function [OUT1,OUT2,OUT3]=dummy(x,y)

if nargin==1,
    table=tabulate(x);

    I=find(table(:,2)<=0);
    table(I,:)=[];

    K1=size(table,1);

    for i=1:K1-1,
        OUT1(:,i)=(x==table(i+1,1));
    end
    
    OUT2=[];
    OUT3=[];
    
elseif nargin==2,
    table1=tabulate(x);
    
    I=find(table1(:,2)<=0);
    table1(I,:)=[];
    
    K1=size(table1,1);
    
    for i=1:K1-1,
        OUT1(:,i)=(x==table1(i+1,1));
    end
    
    table2=tabulate(y);
    
    J=find(table2(:,2)<=0);
    table2(J,:)=[];
    
    K2=size(table2,1);
    
    for j=1:K2-1,
        OUT2(:,j)=(y==table2(j+1,1));
    end
    
    for i=1:K1-1,
        for j=1:K2-1,
            out3(:,j,i)=(x==table1(i+1,1) & y==table2(j+1,1));
        end
    end
    n=size(out3,1);
    OUT3=reshape(out3,n,(K1-1)*(K2-1));
    
else,
    error('2 or fewer input arguments are required');
end