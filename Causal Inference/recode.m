function r = recode(x)

x = x - min(x) + 1;
table=tabulate(x);

I=find(table(:,2)<=0);
table(I,:)=[];
K1=size(table,1);
table = [[1:K1]',table];

for i = 1:max(table(:,2)),
    I = find(x == i);
    if isempty(I) ~= 1,
        J = find(table(:,2) == i);
        x(I) = table(J,1);
    end
end

r = x;