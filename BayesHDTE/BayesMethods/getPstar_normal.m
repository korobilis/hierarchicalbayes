

function pStar = getPstar_normal(beta, theta, tau0, tau1) 
  part1 = theta*normpdf(beta,0,sqrt(tau1));
  part0 = (1-theta)*normpdf(beta,0,sqrt(tau0));
  pStar=(part1 ./ (part1 + part0));
end 
