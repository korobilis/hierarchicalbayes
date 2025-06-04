
function [Q] = student_T_prior(beta,b0)

%%iGamma shrinkage prior
S1 = 1 + 1/2;
S2 = b0 + (beta.^2)./2;
invQ = gamrnd(S1,1./S2);
Q  = 1./invQ;
end 