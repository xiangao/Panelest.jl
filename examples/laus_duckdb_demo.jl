using DataFrames
using DBInterface
using DuckDB
using Panelest
using StatsModels

const LAUS_DIR = get(ENV, "LAUS_DIR", expanduser("~/projects/data/laus"))
const START_YEAR = parse(Int, get(ENV, "LAUS_START_YEAR", "2015"))
const END_YEAR = parse(Int, get(ENV, "LAUS_END_YEAR", "2025"))

sqlpath(path) = replace(path, "'" => "''")

function build_laus_panel!(con; laus_dir = LAUS_DIR, start_year = START_YEAR, end_year = END_YEAR)
    series_path = sqlpath(joinpath(laus_dir, "la.series"))
    area_path = sqlpath(joinpath(laus_dir, "la.area"))
    county_path = sqlpath(joinpath(laus_dir, "la.data.64.County"))

    DBInterface.execute(con, """
    CREATE OR REPLACE VIEW laus_series AS
    SELECT
        trim(series_id) AS series_id,
        area_code,
        measure_code,
        seasonal
    FROM read_csv_auto('$series_path', delim='\\t', header=true, all_varchar=true)
    """)

    DBInterface.execute(con, """
    CREATE OR REPLACE VIEW laus_area AS
    SELECT area_type_code, area_code, area_text
    FROM read_csv_auto('$area_path', delim='\\t', header=true, all_varchar=true)
    """)

    DBInterface.execute(con, """
    CREATE OR REPLACE VIEW laus_county_raw AS
    SELECT
        trim(series_id) AS series_id,
        CAST(year AS INTEGER) AS year,
        CAST(substr(period, 2, 2) AS INTEGER) AS month,
        CAST(trim(value) AS DOUBLE) AS value
    FROM read_csv_auto('$county_path', delim='\\t', header=true, all_varchar=true)
    WHERE period != 'M13'
      AND trim(value) != '-'
      AND CAST(year AS INTEGER) BETWEEN $start_year AND $end_year
    """)

    DBInterface.execute(con, """
    CREATE OR REPLACE TABLE laus_panel AS
    WITH joined AS (
        SELECT
            s.area_code AS county_id,
            a.area_text AS county,
            substr(s.area_code, 3, 2) AS state_fips,
            r.year,
            r.month,
            r.year * 100 + r.month AS year_month,
            s.measure_code,
            r.value
        FROM laus_county_raw r
        JOIN laus_series s USING (series_id)
        JOIN laus_area a ON a.area_code = s.area_code
        WHERE s.seasonal = 'U'
          AND s.measure_code IN ('03', '04', '05', '06')
          AND a.area_type_code = 'F'
    ),
    wide AS (
        SELECT
            county_id,
            county,
            state_fips,
            year,
            month,
            year_month,
            max(CASE WHEN measure_code = '03' THEN value END) AS unemp_rate,
            max(CASE WHEN measure_code = '04' THEN value END) AS unemployment,
            max(CASE WHEN measure_code = '05' THEN value END) AS employment,
            max(CASE WHEN measure_code = '06' THEN value END) AS labor_force
        FROM joined
        GROUP BY county_id, county, state_fips, year, month, year_month
    )
    SELECT
        *,
        ln(labor_force) AS log_labor_force,
        ln(unemployment + 1) AS log_unemployment,
        unemployment / labor_force AS unemployment_share
    FROM wide
    WHERE unemp_rate IS NOT NULL
      AND unemployment IS NOT NULL
      AND employment IS NOT NULL
      AND labor_force IS NOT NULL
      AND labor_force > 0
    """)
end

function main()
    total_start = time()
    db = DuckDB.DB()
    con = DBInterface.connect(db)

    build_seconds = @elapsed build_laus_panel!(con)

    summary = DataFrame(DBInterface.execute(con, """
    SELECT
        count(*) AS rows,
        count(DISTINCT county_id) AS counties,
        min(year) AS first_year,
        max(year) AS last_year,
        round(avg(unemp_rate), 2) AS avg_unemp_rate
    FROM laus_panel
    """))
    println("LAUS county-month panel")
    println(summary)

    println("\nModel 1: county unemployment rate on labor-force size, county FE, and year-month FE")
    model_rate_seconds = @elapsed begin
        model_rate = feols(
            con,
            "laus_panel",
            @formula(unemp_rate ~ log_labor_force + fe(county_id) + fe(year_month)),
        )
    end
    println(model_rate)

    println("\nModel 2: log unemployed workers on labor-force size, county FE, and year-month FE")
    model_count_seconds = @elapsed begin
        model_count = feols(
            con,
            "laus_panel",
            @formula(log_unemployment ~ log_labor_force + fe(county_id) + fe(year_month)),
        )
    end
    println(model_count)

    println("\nCoefficients")
    println(DataFrame(
        model = ["unemployment rate", "log unemployment"],
        coefficient = [only(model_rate.beta), only(model_count.beta)],
        std_error = [sqrt(only(model_rate.vcov)), sqrt(only(model_count.vcov))],
    ))

    total_seconds = time() - total_start
    println("\nTiming")
    println(DataFrame(
        step = ["DuckDB panel build", "feols unemployment rate", "feols log unemployment", "total in-process"],
        seconds = round.([build_seconds, model_rate_seconds, model_count_seconds, total_seconds]; digits = 2),
    ))
end

main()
