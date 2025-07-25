function [l,p,eta] = logL(m,beta)
% LOGL returns the log-likelihood of the logistic model.
%
% Usage:
%  [l,p,eta] = logL(m,beta)
%
% Inputs:
%   m     : a glm_multinomial model
%   beta  : a vector of coefficients, or a string
%
% Outputs:
%   l   : the log-likelihood
%   p   : the fitted probabilities
%   eta : the linear estimates
%
% Notes:
%   When beta is a vector of coefficients, we have
%   eta(c) = X*beta(c) for category c
%   p(c)   = exp(eta(c))/sum(exp(eta(c)), clipped into the range 0.0001...0.9999
%   l      = sum(m.y.*log(p))
%
%   When beta=='saturated' or 'null', the saturated or null likelihoods are 
%   returned 

if isnumeric(beta)
    eta    = predictor(m,beta);
    expeta = exp(eta);
    p = 0;
    for i=1:m.q % select the right category
        p = p+(m.catid==i).*expeta(:,i);
    end
    p = p./sum(expeta,2);
elseif strcmp(beta,'null')
    % this will depend on whether there is an intercept or not
    if ~isempty(property(m,'intercept'))
        p = sum(m.original.y); p = p/sum(p);
        p = repmat(p,[size(m.original.y,1),1]);
        p = p(:);
        p = p(m.original.y(:)~=0);
    else
        p = 1/m.q;
    end
elseif strcmp(beta,'saturated')
    p = bsxfun(@times,m.original.y, 1./sum(m.original.y,2));
    p = p(:);
    p = p(p~=0);
end

p(p<0.0001)=0.0001;
p(p>0.9999)=0.9999;

l = l_from_p(m,p);



