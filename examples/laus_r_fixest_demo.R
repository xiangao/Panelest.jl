suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
})

laus_dir <- Sys.getenv("LAUS_DIR", path.expand("~/projects/data/laus"))
start_year <- as.integer(Sys.getenv("LAUS_START_YEAR", "2015"))
end_year <- as.integer(Sys.getenv("LAUS_END_YEAR", "2025"))

elapsed <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  list(value = value, seconds = proc.time()[["elapsed"]] - start)
}

build_laus_panel <- function(laus_dir, start_year, end_year) {
  series <- fread(
    file.path(laus_dir, "la.series"),
    sep = "\t",
    colClasses = "character",
    showProgress = FALSE
  )
  setnames(series, trimws(names(series)))
  series[, series_id := trimws(series_id)]
  series <- series[
    measure_code %chin% c("03", "04", "05", "06") & seasonal == "U",
    .(series_id, area_code, measure_code)
  ]

  area <- fread(
    file.path(laus_dir, "la.area"),
    sep = "\t",
    colClasses = "character",
    showProgress = FALSE
  )
  setnames(area, trimws(names(area)))
  area <- area[
    area_type_code == "F",
    .(area_code, county = area_text)
  ]

  county_raw <- fread(
    file.path(laus_dir, "la.data.64.County"),
    sep = "\t",
    colClasses = list(character = c("series_id", "period", "value")),
    select = c("series_id", "year", "period", "value"),
    showProgress = FALSE
  )
  setnames(county_raw, trimws(names(county_raw)))
  county_raw[, `:=`(
    series_id = trimws(series_id),
    period = trimws(period),
    value = trimws(value)
  )]
  county_raw <- county_raw[
    period != "M13" &
      year >= start_year &
      year <= end_year &
      value != "-"
  ]
  county_raw[, value := as.numeric(value)]

  joined <- merge(county_raw, series, by = "series_id")
  joined <- merge(joined, area, by = "area_code")

  panel <- dcast(
    joined,
    area_code + county + year + period ~ measure_code,
    value.var = "value"
  )
  setnames(
    panel,
    c("area_code", "county", "year", "period", "03", "04", "05", "06"),
    c("county_id", "county", "year", "period", "unemp_rate", "unemployment", "employment", "labor_force")
  )

  panel <- panel[
    !is.na(unemp_rate) &
      !is.na(unemployment) &
      !is.na(employment) &
      !is.na(labor_force) &
      labor_force > 0
  ]
  panel[, `:=`(
    state_fips = substr(county_id, 3, 4),
    month = as.integer(substr(period, 2, 3)),
    year_month = year * 100L + as.integer(substr(period, 2, 3)),
    log_labor_force = log(labor_force),
    log_unemployment = log(unemployment + 1),
    unemployment_share = unemployment / labor_force
  )]

  setorder(panel, county_id, year, month)
  panel[]
}

total_start <- proc.time()[["elapsed"]]

panel_timing <- elapsed(build_laus_panel(laus_dir, start_year, end_year))
laus_panel <- panel_timing$value

cat("LAUS county-month panel\n")
print(laus_panel[, .(
  rows = .N,
  counties = uniqueN(county_id),
  first_year = min(year),
  last_year = max(year),
  avg_unemp_rate = round(mean(unemp_rate), 2)
)])

cat("\nModel 1: county unemployment rate on labor-force size, county FE, and year-month FE\n")
model_rate_timing <- elapsed(feols(
  unemp_rate ~ log_labor_force | county_id + year_month,
  data = laus_panel,
  notes = FALSE
))
print(summary(model_rate_timing$value))

cat("\nModel 2: log unemployed workers on labor-force size, county FE, and year-month FE\n")
model_count_timing <- elapsed(feols(
  log_unemployment ~ log_labor_force | county_id + year_month,
  data = laus_panel,
  notes = FALSE
))
print(summary(model_count_timing$value))

coef_table <- data.table(
  model = c("unemployment rate", "log unemployment"),
  coefficient = c(coef(model_rate_timing$value)[["log_labor_force"]],
                  coef(model_count_timing$value)[["log_labor_force"]]),
  std_error = c(se(model_rate_timing$value)[["log_labor_force"]],
                se(model_count_timing$value)[["log_labor_force"]])
)
cat("\nCoefficients\n")
print(coef_table)

total_seconds <- proc.time()[["elapsed"]] - total_start
cat("\nTiming\n")
print(data.table(
  step = c("data.table panel build", "fixest unemployment rate", "fixest log unemployment", "total in-process"),
  seconds = round(c(
    panel_timing$seconds,
    model_rate_timing$seconds,
    model_count_timing$seconds,
    total_seconds
  ), 2)
))
