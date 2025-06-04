library("R.matlab")
library("glmnet")

a <- readMat("temp.mat")
if (a$standardize) {
    message("Standardized")
}
if (a$intercept) {
    message("Intercept")
}
x <- proc.time()
fit <- glmnet(a$X, a$y, family = a$type, standardize = a$standardize, intercept = a$intercept,
              alpha = a$alpha, nlambda = a$nlambda)
t <- proc.time() - x

if (is.list(fit$beta)) {
    message("is a list")
    fit$beta <- do.call("rbind", lapply(fit$beta, as.matrix))
} else {
    message("not a list")
}
  
writeMat("tempout.mat", a0 = fit$a0, beta = fit$beta, dev = fit$dev.ratio, 
         df = fit$df, lambda = fit$lambda, nulldev = fit$nulldev, elapsed = t)

