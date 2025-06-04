function keep = findNonCollinear(Z,tol)

if nargin < 2
    tol = 1e-16;
end
if isempty(tol)
    tol = 1e-16;
end

p = size(Z,2);
keep = (1:p)';

for ii = 1:p
    use = keep ~= p-ii+1;
    e = Z(:,p-ii+1)-Z(:,keep(use))*pinv(Z(:,keep(use))'*Z(:,keep(use)))*(Z(:,keep(use))'*Z(:,p-ii+1));
    if sum(e.^2)/sum(Z(:,p-ii+1).^2) < tol
        keep = setdiff(keep,p-ii+1);
    end
end
