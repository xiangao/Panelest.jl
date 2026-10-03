# Reference values for the feiv diagnostics test in test/runtests.jl (fixest fitstat).
# Run from this directory: Rscript make_feiv_ref.R
suppressPackageStartupMessages(library(fixest))
set.seed(7)
n <- 600
d <- data.frame(g = sample(1:12, n, TRUE), h = sample(1:4, n, TRUE),
                z1 = rnorm(n), z2 = rnorm(n), x = rnorm(n), c = rnorm(n))
d$w1 <- 0.6 * d$z1 + 0.3 * d$z2 + 0.5 * d$c + 0.2 * d$x + d$g / 10 + rnorm(n)
d$w2 <- 0.4 * d$z2 - 0.2 * d$z1 + 0.3 * d$c + rnorm(n)
d$y  <- d$w1 + 0.5 * d$w2 + 0.3 * d$x + 0.8 * d$c + d$h / 5 + rnorm(n)
write.csv(d, "feiv_sim.csv", row.names = FALSE)
m <- feols(y ~ x | g + h | w1 + w2 ~ z1 + z2, data = d, vcov = "hetero")
f <- fitstat(m, ~ ivf + wh)
ref <- data.frame(stat = c("ivf1", "ivf2", "wh", "beta_w1", "se_w1", "beta_w2", "se_w2"),
                  value = c(f[["ivf1::w1"]]$stat, f[["ivf1::w2"]]$stat, f$wh$stat,
                            coef(m)["fit_w1"], se(m)["fit_w1"], coef(m)["fit_w2"], se(m)["fit_w2"]))
write.csv(ref, "feiv_ref.csv", row.names = FALSE)
print(ref, digits = 10); print(f)
