function LogTheta=getLogTheta(theta, a, b, w, gamma) 
  part1 = (a-1)*log(theta);
  part2 = (b-1)*log(1 - theta);
  part3 = sum(w.*gamma)*log(theta);
  part4 = sum(log((1 - theta.^w).^(1-gamma)));
  LogTheta=(part1 + part2 + part3 + part4);
end 