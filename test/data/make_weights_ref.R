# Reference values for the weighted-estimation tests in test/test_vs_r.jl (fixest).
# Weights are analytic weights scaled to sum to about 1, as with population shares,
# so a degrees-of-freedom formula that uses sum(weights) in place of n shows up.
# Run from this directory: Rscript make_weights_ref.R
suppressPackageStartupMessages(library(fixest))
set.seed(11)
n <- 800
d <- data.frame(g = sample(1:20, n, TRUE), h = sample(1:5, n, TRUE),
                x1 = rnorm(n), x2 = rnorm(n), z1 = rnorm(n), z2 = rnorm(n),
                c = rnorm(n))
d$wt    <- runif(n, 0.2, 3) / n
d$endo  <- 0.7 * d$z1 + 0.4 * d$z2 + 0.6 * d$c + 0.2 * d$x1 + d$g / 10 + rnorm(n)
d$y     <- 0.5 * d$endo + 0.3 * d$x1 - 0.2 * d$x2 + 0.8 * d$c + d$h / 5 +
           rnorm(n) * (1 + abs(d$x1))
d$count <- rpois(n, exp(0.3 + 0.4 * d$x1 - 0.2 * d$x2 + d$h / 10))
d$bin   <- rbinom(n, 1, plogis(-0.2 + 0.8 * d$x1 - 0.5 * d$x2 + d$h / 10))
write.csv(d, "weights_sim.csv", row.names = FALSE)

vc <- list(iid = "iid", hetero = "hetero", cluster = ~g)
fits <- list()
for (v in names(vc)) {
  fits[[paste0("ols_nofe_", v)]] <- feols(y ~ x1 + x2, d, weights = ~wt, vcov = vc[[v]])
  fits[[paste0("ols_fe_", v)]]   <- feols(y ~ x1 + x2 | h, d, weights = ~wt, vcov = vc[[v]])
  fits[[paste0("iv_nofe_", v)]]  <- feols(y ~ x1 + x2 | endo ~ z1 + z2, d, weights = ~wt,
                                          vcov = vc[[v]])
  fits[[paste0("iv_fe_", v)]]    <- feols(y ~ x1 + x2 | h | endo ~ z1 + z2, d, weights = ~wt,
                                          vcov = vc[[v]])
}
for (v in c("hetero", "cluster")) {
  fits[[paste0("pois_fe_", v)]]  <- fepois(count ~ x1 + x2 | h, d, weights = ~wt, vcov = vc[[v]])
  fits[[paste0("logit_fe_", v)]] <- feglm(bin ~ x1 + x2 | h, d, family = "logit",
                                          weights = ~wt, vcov = vc[[v]])
}

ref <- do.call(rbind, lapply(names(fits), function(k) {
  m <- fits[[k]]
  data.frame(case = k, term = sub("^fit_", "", names(coef(m))),
             coef = unname(coef(m)), se = unname(se(m)))
}))
ivf <- data.frame(case = c("iv_nofe_iid", "iv_fe_iid"), term = "ivf",
                  coef = c(fitstat(fits$iv_nofe_iid, "ivf")[[1]]$stat,
                           fitstat(fits$iv_fe_iid, "ivf")[[1]]$stat),
                  se = NA)
ref <- rbind(ref, ivf)
write.csv(ref, "weights_ref.csv", row.names = FALSE)
print(ref, digits = 8)
