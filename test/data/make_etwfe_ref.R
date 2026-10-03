# Reference values for test/test_etwfe.jl, from R etwfe 0.6.2 / fixest / marginaleffects.
# Run from this directory: Rscript make_etwfe_ref.R
suppressPackageStartupMessages({library(etwfe); library(dplyr); library(haven)})

data("mpdta", package = "did")
mp <- mpdta %>% transmute(countyreal, year, first_treat = first.treat, lemp, lpop)
write.csv(mp, "etwfe_mpdta.csv", row.names = FALSE)

st <- read_dta("~/projects/books/causal_econometrics_guide/data/did_staggered_6.dta") %>%
  mutate(first_treat = case_when(d4 == 1 ~ 2004, d5 == 1 ~ 2005, d6 == 1 ~ 2006, .default = 0)) %>%
  transmute(id, year, first_treat, y, x)
write.csv(st, "etwfe_staggered.csv", row.names = FALSE)

set.seed(42)
N <- 400; Tmax <- 6
ufe <- rnorm(N) * 0.3
pc <- expand.grid(id = 1:N, year = 1:Tmax) %>% arrange(id, year) %>%
  mutate(first_treat = c(3, 4, 5, 0)[((id - 1) %/% 100) + 1],
         treated = (first_treat > 0) & (year >= first_treat),
         z = round(cos(id * 1.3) + 0.1 * year, 6),
         count = rpois(n(), exp(1 + 0.2 * year + ufe[id] + 0.4 * treated + 0.2 * z))) %>%
  select(id, year, first_treat, z, count)
write.csv(pc, "etwfe_poisson.csv", row.names = FALSE)

out <- list()
add <- function(case, type, e) {
  key <- if (type == "simple") rep(NA, nrow(e)) else if (type == "calendar") e[["year"]] else e[["event"]]
  out[[length(out) + 1]] <<- data.frame(case = case, type = type, key = key,
                                        estimate = e$estimate, std_error = e$std.error)
}
run <- function(case, m, types = c("simple", "event", "calendar"), ...) {
  for (ty in types) add(case, ty, emfx(m, type = ty, ...))
}

m1 <- etwfe(lemp ~ lpop, tvar = year, gvar = first_treat, data = mp, vcov = ~countyreal)
run("mpdta_lpop", m1)
add("mpdta_lpop", "event_pre", emfx(m1, type = "event", post_only = FALSE))
m1n <- etwfe(lemp ~ lpop, tvar = year, gvar = first_treat, data = mp, cgroup = "never", vcov = ~countyreal)
run("mpdta_lpop_never", m1n)
m1c <- etwfe(lemp ~ 1, tvar = year, gvar = first_treat, data = mp, vcov = ~countyreal)
run("mpdta_noctrl", m1c)
m2 <- etwfe(y ~ x, tvar = year, gvar = first_treat, data = st, vcov = ~id)
run("staggered_x", m2)
m3 <- etwfe(count ~ 1, tvar = year, gvar = first_treat, data = pc, family = "poisson", cgroup = "never", vcov = ~id)
run("poisson_never", m3)
m3z <- etwfe(count ~ z, tvar = year, gvar = first_treat, data = pc, family = "poisson", vcov = ~id)
run("poisson_z", m3z)

ref <- do.call(rbind, out)
write.csv(ref, "etwfe_ref.csv", row.names = FALSE)
print(ref, digits = 8)
