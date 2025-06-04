
function pStar = getPstar(beta, theta, lambda1, lambda0) 
  part1 = theta*lambda1*exp(-lambda1*abs(beta));
  part0 = (1-theta)*lambda0*exp(-lambda0*abs(beta));
  pStar=(part1 / (part1 + part0));
end 