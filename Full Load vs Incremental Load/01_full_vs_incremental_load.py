# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1: Full Load vs Incremental Load — The Problem
# MAGIC **Module:** Data Load Strategies | Medallion Architecture Pipeline Optimization
# MAGIC
# MAGIC ---
# MAGIC ## Learning Objectives
# MAGIC - Understand why full table overwrites fail at scale
# MAGIC - Measure the compute and time cost difference between full and incremental loads
# MAGIC - Identify the trigger conditions that make incremental loading necessary
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup
# MAGIC We simulate a Bronze-layer raw events table with growing row counts to replicate real pipeline pressure.

# COMMAND ----------

# DBTITLE 1,Cell 3
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, lit, rand, expr, to_timestamp, date_sub
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    TimestampType, DoubleType
)
from delta.tables import DeltaTable
import time

spark = SparkSession.builder.getOrCreate()

# Create a Unity Catalog volume for data storage
spark.sql("CREATE VOLUME IF NOT EXISTS datapipeline2026.default.pipeline_module")

# Use Unity Catalog Volume path
BASE_PATH = "/Volumes/datapipeline2026/default/pipeline_module"

print(f"Base path: {BASE_PATH}")
print("Spark session ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Simulated Source Data (Bronze Layer)
# MAGIC We generate a synthetic e-commerce events dataset.
# MAGIC - **Small dataset** → 50,000 rows  (simulates Day 1)
# MAGIC - **Large dataset** → 5,000,000 rows (simulates Day 90 after growth)

# COMMAND ----------

schema = StructType([
    StructField("event_id",      StringType(),    False),
    StructField("user_id",       IntegerType(),   False),
    StructField("event_type",    StringType(),    False),
    StructField("product_id",    StringType(),    False),
    StructField("amount",        DoubleType(),    True),
    StructField("event_ts",      TimestampType(), False),
])

def generate_events(n_rows: int, day_offset: int = 0):
    """
    Generate n_rows of synthetic event records offset by day_offset days from today.
    Returns a Spark DataFrame.
    """
    return (
        spark.range(n_rows)
        .select(
            expr("uuid()").alias("event_id"),
            (col("id") % 10000).cast("int").alias("user_id"),
            expr("CASE WHEN rand() < 0.4 THEN 'page_view' "
                 "     WHEN rand() < 0.7 THEN 'add_to_cart' "
                 "     ELSE 'purchase' END").alias("event_type"),
            expr("concat('PROD_', cast(floor(rand()*500) as string))").alias("product_id"),
            (rand() * 500).alias("amount"),
            expr(f"current_timestamp() - INTERVAL {day_offset} DAYS "
                 f"- (rand() * INTERVAL 86400 SECONDS)").alias("event_ts"),
        )
    )

# Generate small (Day 1) and large (Day 90) datasets
df_small = generate_events(50_000,   day_offset=0)
df_large = generate_events(5_000_000, day_offset=90)

print(f"Small dataset row count : {df_small.count():,}")
print(f"Large dataset row count : {df_large.count():,}")

# COMMAND ----------

df_small.count()

# COMMAND ----------

df_large.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write Source Tables to Delta
# MAGIC Both tables are persisted as Delta tables in the Bronze zone.

# COMMAND ----------

# DBTITLE 1,Cell 9
BRONZE_SMALL = f"{BASE_PATH}/bronze/events_small"
BRONZE_LARGE = f"{BASE_PATH}/bronze/events_large"

# Write small table
df_small.write.format("delta").mode("overwrite").save(BRONZE_SMALL)

# Write large table (partitioned by date for realism)
(df_large
    .withColumn("event_date", col("event_ts").cast("date"))
    .write
    .format("delta")
    .partitionBy("event_date")
    .mode("overwrite")
    .save(BRONZE_LARGE)
)

print("Bronze tables written.")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM delta.`/Volumes/datapipeline2026/default/pipeline_module/bronze/events_small/` limit 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. FULL LOAD PATTERN — Overwrite Everything Every Run
# MAGIC This is the legacy approach: read ALL source rows, transform, overwrite target.
# MAGIC
# MAGIC ### Why teams still use this:
# MAGIC - Simple to implement — no state management
# MAGIC - Guarantees a clean slate (no duplicates from merge logic)
# MAGIC
# MAGIC ### Why it breaks at scale:
# MAGIC - Reads and rewrites **every row**, even unchanged data
# MAGIC - Cluster must hold the entire dataset in memory/shuffle
# MAGIC - Costs scale linearly with table size
# MAGIC - Long runtimes block downstream consumers

# COMMAND ----------

SILVER_FULL = f"{BASE_PATH}/silver/events_full_load"

def run_full_load(source_path: str, target_path: str, label: str):
    """
    Simulate a full table overwrite (legacy pattern).
    Reads ALL rows from source and overwrites the target.
    """
    t_start = time.time()

    df_source = spark.read.format("delta").load(source_path)

    # --- Transformation logic (same in both strategies) ---
    df_transformed = (
        df_source
        .filter(col("event_type").isNotNull())
        .withColumn("amount_usd",       col("amount").cast("double"))
        .withColumn("load_ts",          current_timestamp())
        .withColumn("load_strategy",    lit("FULL_OVERWRITE"))
        .drop("amount")
    )

    # Overwrite the entire target
    (df_transformed
        .write
        .format("delta")
        .mode("overwrite")
        .option("mergeSchema", "true")
        .save(target_path))

    elapsed = time.time() - t_start
    row_count = df_transformed.count()

    print(f"\n{'='*55}")
    print(f"  FULL LOAD — {label}")
    print(f"{'='*55}")
    print(f"  Rows processed : {row_count:>15,}")
    print(f"  Elapsed time   : {elapsed:>15.2f}s")
    print(f"  Strategy       : {'FULL OVERWRITE':>15}")
    print(f"{'='*55}\n")
    return elapsed, row_count

# Run full load on SMALL dataset (baseline)
t_small, r_small = run_full_load(BRONZE_SMALL, SILVER_FULL, "Small Dataset (50K rows)")

# Run full load on LARGE dataset (scale problem)
t_large, r_large = run_full_load(BRONZE_LARGE, SILVER_FULL, "Large Dataset (5M rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. INCREMENTAL LOAD PATTERN — Process Only New Rows
# MAGIC This is the modern approach: track the last processed timestamp, read only new records.
# MAGIC
# MAGIC > **Key insight:** On Day 90, only ~55,000 rows arrived in the last 24 hours.
# MAGIC > There is no reason to reprocess the 4.9M rows that have not changed.

# COMMAND ----------

SILVER_INCREMENTAL = f"{BASE_PATH}/silver/events_incremental"
WATERMARK_PATH     = f"{BASE_PATH}/watermark/events_watermark"

from pyspark.sql.functions import max as spark_max

def get_watermark(watermark_path: str):
    """
    Read the last successfully processed timestamp from the watermark store.
    Returns None if no watermark exists (first run).
    """
    try:
        wm_df = spark.read.format("delta").load(watermark_path)
        wm = wm_df.select(spark_max("last_processed_ts")).collect()[0][0]
        print(f"Watermark found: {wm}")
        return wm
    except Exception:
        print("No watermark found — this is a first run.")
        return None

def save_watermark(watermark_path: str, new_ts):
    """
    Persist the new high-water mark so the next run knows where to start.
    """
    wm_df = spark.createDataFrame(
        [(new_ts,)],
        schema="last_processed_ts TIMESTAMP"
    )
    wm_df.write.format("delta").mode("overwrite").save(watermark_path)
    print(f"Watermark saved: {new_ts}")

def run_incremental_load(source_path: str, target_path: str,
                         watermark_path: str, label: str):
    """
    Incremental load: only processes rows newer than the stored watermark.
    """
    t_start = time.time()

    last_processed_ts = get_watermark(watermark_path)

    df_source = spark.read.format("delta").load(source_path)

    # Filter to only NEW records since the watermark
    if last_processed_ts:
        df_new = df_source.filter(col("event_ts") > lit(last_processed_ts))
    else:
        df_new = df_source  # First run: process everything

    # --- Same transformation logic as full load ---
    df_transformed = (
        df_new
        .filter(col("event_type").isNotNull())
        .withColumn("amount_usd",    col("amount").cast("double"))
        .withColumn("load_ts",       current_timestamp())
        .withColumn("load_strategy", lit("INCREMENTAL"))
        .drop("amount")
    )

    row_count = df_transformed.count()

    if row_count == 0:
        print("No new records to process.")
        return 0, 0

    # Append only the new records
    (df_transformed
        .write
        .format("delta")
        .mode("append")
        .save(target_path))

    # Advance the watermark to the latest event_ts we just processed
    new_max_ts = df_new.select(spark_max("event_ts")).collect()[0][0]
    save_watermark(watermark_path, new_max_ts)

    elapsed = time.time() - t_start

    print(f"\n{'='*55}")
    print(f"  INCREMENTAL LOAD — {label}")
    print(f"{'='*55}")
    print(f"  Rows processed : {row_count:>15,}")
    print(f"  Elapsed time   : {elapsed:>15.2f}s")
    print(f"  Strategy       : {'INCREMENTAL':>15}")
    print(f"{'='*55}\n")
    return elapsed, row_count

# First run — no watermark yet, processes all 5M rows
t_inc_first, r_inc_first = run_incremental_load(
    BRONZE_LARGE, SILVER_INCREMENTAL, WATERMARK_PATH,
    "Large Dataset — First Run (no watermark)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Simulate a Second Run — Only New Data Arrives
# MAGIC We append 55,000 new rows (one day's worth) and re-run the incremental pipeline.
# MAGIC This is where the savings become dramatic.

# COMMAND ----------

# Simulate new daily data arriving
df_new_day = generate_events(55_000, day_offset=0)

(df_new_day
    .withColumn("event_date", col("event_ts").cast("date"))
    .write
    .format("delta")
    .mode("append")
    .save(BRONZE_LARGE)
)

print(f"Appended {df_new_day.count():,} new rows to Bronze.")

# Re-run incremental — should only process the 55K new rows
t_inc_second, r_inc_second = run_incremental_load(
    BRONZE_LARGE, SILVER_INCREMENTAL, WATERMARK_PATH,
    "Large Dataset — Second Run (55K new rows)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Performance Comparison Summary

# COMMAND ----------

print("""
╔══════════════════════════════════════════════════════════════╗
║          FULL LOAD vs INCREMENTAL LOAD — SUMMARY             ║
╠══════════════════════════════════════════════════════════════╣
║  Scenario                    │ Rows Read │ Strategy          ║
╠══════════════════════════════════════════════════════════════╣
║  Full Load  (50K table)      │    50,000 │ Overwrite ALL     ║
║  Full Load  (5M table)       │ 5,000,000 │ Overwrite ALL     ║
║  Incremental (5M, Run 1)     │ 5,000,000 │ Append (no WM)    ║
║  Incremental (5M, Run 2)     │    55,000 │ Append (WM delta) ║
╠══════════════════════════════════════════════════════════════╣
║  Run 2 reads 98.9% FEWER rows than a full load               ║
║  At 1 DBU/hr this translates directly to cost savings        ║
╚══════════════════════════════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Key Takeaways
# MAGIC
# MAGIC | Dimension | Full Load | Incremental Load |
# MAGIC |---|---|---|
# MAGIC | **Rows read per run** | Always 100% of table | Only delta (new/changed rows) |
# MAGIC | **Compute cost** | Grows with table size | Stays roughly constant per day |
# MAGIC | **Complexity** | Low — no state needed | Medium — requires watermark store |
# MAGIC | **Failure recovery** | Re-run is safe (overwrite) | Requires idempotent design |
# MAGIC | **Schema evolution** | Handled on overwrite | Must be managed explicitly |
# MAGIC | **Recommended for** | Small tables < 1M rows | Any table growing over time |
# MAGIC
# MAGIC ---
# MAGIC > **Next:** Notebook 2 — Architecting the Watermarking Framework
