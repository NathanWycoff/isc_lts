#!/usr/bin/Rscript
#  R/verify_ar_var.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.27.2018

## Verify my analytic formula for an AR process's variance
Y <- 3
phi <- 0.8
sigma_z <- 2

# Emperically calculate a covariance matrix
iters <- 1e5
Z <- matrix(NA, nrow = iters, ncol = Y)
for (it in 1:iters) {
    Z[it,1] <- rnorm(1, 0, sigma_z)
    for (y in 2:Y) {
        Z[it,y] <- phi * Z[it, y-1] + rnorm(1, 0, sigma_z)
    }
}

SIGMA_emp <- var(Z)

# Analytical based on formula
SIGMA_anl <- matrix(NA, nrow = Y, ncol = Y)
pd <- 0
for (y1 in 1:Y) {
    pd <- pd + (phi^2)^(y1-1)
    for (y2 in y1:Y) {
        SIGMA_anl[y1,y2] <- SIGMA_anl[y2, y1] <- phi^(y2-y1) * pd
    }
}
SIGMA_anl <- sigma_z^2 * SIGMA_anl

print(SIGMA_emp)
print(SIGMA_anl)
