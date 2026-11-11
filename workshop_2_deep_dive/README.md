# Workshop 2 · Impulse deep dive

**~3.5 hours · prerequisite: workshop 1 (or equivalent Impulse basics)**

Same setup as workshop 1: clone the repo, set `USER_SCHEMA` in `CONFIG`, use Serverless.

| # | Notebook | Time | What you learn |
|---|---|---|---|
| 1 | [`01_bronze_to_silver_data_engineering`](01_bronze_to_silver_data_engineering.py) | 60 min | build the Impulse **silver model** yourself from bronze: time base, data-quality findings (flag, don't drop), **RLE**, `channel_metrics`, EAV tags, POI events, **aliases + unit conversion** — then validate it with Impulse |
| — | [`declarative_pipeline/`](declarative_pipeline/) | 20 min | the same silver build as a **Lakeflow Declarative Pipeline** with expectations, orchestrated by a **job** that smoke-tests the result |
| 2a | [`02a_tsal_deep_dive`](02a_tsal_deep_dive.py) | 75 min | `SampleSeries` / `Intervals` / `PointsInTime` / `PointsInTimeSeries` on real trips, combining multi-rate signals, container & channel filters, alias fallback, K → °C, radians → route map, POI streams, resampling/integration, **custom UDFs**, interactive exploration with widgets |
| 2b | [`02b_report_sql_dashboard`](02b_report_sql_dashboard.py) | 60 min | a deep report: every event type, distance-weighted & 2-D histograms, **custom statistics**, point values, a calculated channel, incremental re-runs; SQL on gold; a 5-page dashboard |

Everything you create is in your schema with the prefixes `impulse_w2_silver_*` (notebook 1),
`impulse_w2_sdp_*` (pipeline) and `impulse_w2_*` (report), plus the dashboard
**"Impulse W2 - Fleet deep dive (<your schema>)"**.

The solver throughout is Impulse's `DefaultSolver`.
