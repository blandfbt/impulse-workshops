"""Workshop 2 -- bronze -> Impulse silver as a Lakeflow Spark Declarative Pipeline.

This is notebook 01_bronze_to_silver_data_engineering, restated declaratively: every silver table
is a materialized view, the pipeline works out the dependency graph (samples -> channels ->
metrics -> ...), and data-quality rules are pipeline *expectations* whose pass/fail
counts show up in the pipeline UI.

The transformation logic itself is shared with the instructor setup in `common/impulse_silver.py`.

Pipeline configuration (set by the bundle in resources/w2_silver.pipeline.yml, or by
declarative_pipeline/00_create_pipeline_and_job when you create it from a notebook):

    impulse_workshop.repo_root      workspace path of this repo (so `common` is importable)
    impulse_workshop.bronze_schema  catalog.schema holding the impulse_bronze_* tables
    impulse_workshop.table_prefix   prefix of the silver tables (default impulse_w2_sdp)
    impulse_workshop.trips          comma-separated session ids to process ("" = all trips)
"""

import sys

import pyspark.sql.functions as F
from pyspark import pipelines as dp

sys.path.insert(0, spark.conf.get("impulse_workshop.repo_root"))  # noqa: F821
from common import impulse_silver as silver  # noqa: E402

BRONZE = spark.conf.get("impulse_workshop.bronze_schema", "dbx_workshop_dev.shared")  # noqa: F821
PREFIX = spark.conf.get("impulse_workshop.table_prefix", "impulse_w2_sdp")  # noqa: F821
TRIPS = [int(t) for t in spark.conf.get("impulse_workshop.trips", "").split(",") if t.strip()]  # noqa: F821
spark.conf.set("spark.sql.session.timeZone", "UTC")  # noqa: F821  -- see silver.use_utc


def t(name: str) -> str:
    return f"{PREFIX}_{name}"


def bronze(name: str):
    df = spark.read.table(f"{BRONZE}.impulse_bronze_{name}")  # noqa: F821
    if name == "signal_data" and TRIPS:
        df = df.where(F.col("session_id").isin(TRIPS))
    return df


# ---------------------------------------------------------------------------------------------
# Samples (not persisted) -- with row-level expectations
# ---------------------------------------------------------------------------------------------
@dp.temporary_view(name="samples", comment="Bronze observations mapped to (container_id, channel_id, timestamp us, value)")
@dp.expect_or_drop("value_present", "value IS NOT NULL")
@dp.expect_or_fail("timestamp_present", "timestamp IS NOT NULL")
def samples():
    return silver.samples(bronze("signal_data"))


# ---------------------------------------------------------------------------------------------
# Data-quality findings (flagged, never dropped -- see DATA_NOTES.md)
# ---------------------------------------------------------------------------------------------
@dp.materialized_view(name=t("dq_findings"),
                      comment="Data-quality findings per (trip, signal); silver keeps the measured values")
def dq_findings():
    return silver.dq_findings(spark.read.table("samples"), bronze("signal_info"))  # noqa: F821


# ---------------------------------------------------------------------------------------------
# Required Impulse tables
# ---------------------------------------------------------------------------------------------
@dp.materialized_view(name=t("channels"), cluster_by=["container_id", "channel_id"],
                      comment="Impulse silver (required): RLE samples [tstart, tend) in epoch us")
@dp.expect_or_fail("interval_is_ordered", "tstart <= tend")
def channels():
    return silver.rle_encode(spark.read.table("samples"))  # noqa: F821


@dp.materialized_view(name=t("poi_channels"), cluster_by=["container_id", "channel_id"],
                      comment="Impulse silver (optional): Points-in-Time events (gear shifts, driver alerts)")
@dp.expect_or_fail("exactly_one_value", "(value_double IS NULL) <> (value_string IS NULL)")
def poi_channels():
    return silver.poi_channels(spark.read.table("samples"), bronze("signal_info"))  # noqa: F821


@dp.materialized_view(name=t("channel_metrics"), cluster_by=["container_id", "channel_id"],
                      comment="Impulse silver (required): one row per (trip, channel)")
@dp.expect("has_name", "channel_name IS NOT NULL")
def channel_metrics():
    sample_channels = silver.channel_metrics(spark.read.table(t("channels")),  # noqa: F821
                                             spark.read.table("samples"), bronze("signal_info"))  # noqa: F821
    return sample_channels.unionByName(silver.poi_channel_metrics(spark.read.table(t("poi_channels"))))  # noqa: F821


@dp.materialized_view(name=t("container_metrics"), cluster_by=["container_id"],
                      comment="Impulse silver (required): one row per trip")
@dp.expect_or_fail("valid_bounds", "start_ts < stop_ts")
def container_metrics():
    return silver.container_metrics(
        spark.read.table(t("channel_metrics")), silver.trip_keys(bronze("signal_data")),  # noqa: F821
        bronze("trip_info"), bronze("vehicle_info"), bronze("vehicle_state"))


# ---------------------------------------------------------------------------------------------
# Optional Impulse tables
# ---------------------------------------------------------------------------------------------
@dp.materialized_view(name=t("channel_tags"), cluster_by=["container_id", "channel_id"],
                      comment="Impulse silver (optional): EAV channel tags")
def channel_tags():
    return silver.channel_tags(spark.read.table(t("channel_metrics")), bronze("signal_info"))  # noqa: F821


@dp.materialized_view(name=t("container_tags"), cluster_by=["container_id"],
                      comment="Impulse silver (optional): EAV container tags")
def container_tags():
    return silver.container_tags(spark.read.table(t("container_metrics")))  # noqa: F821


@dp.materialized_view(name=t("channel_mapping"), comment="Impulse silver (optional): logical channel aliases")
def channel_mapping():
    return silver.channel_mapping(spark)  # noqa: F821


@dp.materialized_view(name=t("unit_conversion"), comment="Impulse silver (optional): unit families and factors")
@dp.expect_or_fail("positive_factor", "conversion_factor > 0")
def unit_conversion():
    return silver.unit_conversion(spark)  # noqa: F821
