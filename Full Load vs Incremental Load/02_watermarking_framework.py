# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 2: Architecting the Watermarking Framework
# MAGIC **Module:** Data Load Strategies | Medallion Architecture Pipeline Optimization
# MAGIC
# MAGIC ---
# MAGIC ## Learning Objectives
# MAGIC - Build a production-grade watermark store using Delta Lake
# MAGIC - Implement high-water mark logic using both timestamp and sequential ID strategies
# MAGIC - Handle pipeline restarts and cluster failures with a safe state-recovery mechanism
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, lit, rand, expr,
    max as spark_max, min as spark_min
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    TimestampType, LongType
)
from delta.tables import DeltaTable
import time
from datetime import datetime, timedelta

spark = SparkSession.builder.getOrCreate()

username  = spark.sql("SELECT current_user()").collect()[0][0]
BASE_PATH = f"/Volumes/datapipeline2026/default/pipeline_module"

BRONZE_PATH    = f"{BASE_PATH}/bronze/orders"
SILVER_PATH    = f"{BASE_PATH}/silver/orders"
WATERMARK_PATH = f"{BASE_PATH}/watermark/orders_wm"

print(f"Base path : {BASE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate the Bronze Source Table (Orders)
# MAGIC Represents a raw orders feed from an OLTP system.
# MAGIC Orders arrive continuously; each has an `order_id` (sequential) and `created_ts`.

# COMMAND ----------

def create_orders_batch(start_id: int, n_rows: int, ts_offset_hours: int = 0):
    """
    Create a batch of order records starting at start_id.
    ts_offset_hours shifts event timestamps to simulate historical data.
    """
    return (
        spark.range(start_id, start_id + n_rows)
        .select(
            col("id").cast("long").alias("order_id"),
            expr("concat('CUST_', cast(floor(rand()*5000) as string))").alias("customer_id"),
            expr("CASE WHEN rand() < 0.5 THEN 'PENDING' "
                 "     WHEN rand() < 0.8 THEN 'SHIPPED' "
                 "     ELSE 'DELIVERED' END").alias("status"),
            (rand() * 1000 + 10).alias("order_amount"),
            expr(f"current_timestamp() - INTERVAL {ts_offset_hours} HOURS "
                 f"- (rand() * INTERVAL 3600 SECONDS)").alias("created_ts"),
            current_timestamp().alias("updated_ts"),
        )
    )

# Simulate 3 days of historical data already in Bronze
batch_day1 = create_orders_batch(start_id=1,      n_rows=20_000, ts_offset_hours=72)
batch_day2 = create_orders_batch(start_id=20_001, n_rows=25_000, ts_offset_hours=48)
batch_day3 = create_orders_batch(start_id=45_001, n_rows=30_000, ts_offset_hours=24)

full_bronze = batch_day1.union(batch_day2).union(batch_day3)

(full_bronze
    .write
    .format("delta")
    .mode("overwrite")
    .save(BRONZE_PATH))

print(f"Bronze orders table created: {full_bronze.count():,} rows")
print(f"Order ID range: 1 — 75,000")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Strategy A — Timestamp-Based Watermark
# MAGIC Track the maximum `created_ts` processed so far.
# MAGIC
# MAGIC **Best for:** Event-driven sources where records have reliable timestamps.
# MAGIC
# MAGIC **Weakness:** Clock skew or late-arriving records can cause missed rows.
# MAGIC Use a small "lookback buffer" (e.g., subtract 5 minutes) to compensate.

# COMMAND ----------

class TimestampWatermarkStore:
    """
    Manages a timestamp-based high-water mark persisted as a Delta table.
    Supports read, update, and safe recovery with a configurable lookback buffer.
    """

    def __init__(self, store_path: str, table_name: str,
                 lookback_seconds: int = 300):
        self.store_path       = f"{store_path}_ts"
        self.table_name       = table_name
        self.lookback_seconds = lookback_seconds

    def get(self):
        """
        Return the current watermark timestamp (minus lookback buffer).
        Returns None on first run.
        """
        try:
            wm_df = spark.read.format("delta").load(self.store_path)
            raw_ts = wm_df.filter(
                col("table_name") == self.table_name
            ).select(spark_max("watermark_ts")).collect()[0][0]

            if raw_ts is None:
                return None

            # Apply lookback buffer to catch late-arriving records
            buffered = raw_ts - timedelta(seconds=self.lookback_seconds)
            print(f"[WM-TS] Raw watermark     : {raw_ts}")
            print(f"[WM-TS] Buffered watermark : {buffered} "
                  f"(-{self.lookback_seconds}s lookback)")
            return buffered

        except Exception as e:
            print(f"[WM-TS] First run or store missing: {e}")
            return None

    def save(self, new_ts):
        """Persist a new high-water mark for this table."""
        wm_schema = "table_name STRING, watermark_ts TIMESTAMP, updated_at TIMESTAMP"
        wm_df = spark.createDataFrame(
            [(self.table_name, new_ts, datetime.utcnow())],
            schema=wm_schema
        )
        wm_df.write.format("delta").mode("overwrite").save(self.store_path)
        print(f"[WM-TS] Watermark advanced to: {new_ts}")

    def reset(self):
        """Reset watermark — use only for full backfill."""
        try:
            DeltaTable.forPath(spark, self.store_path).delete(
                col("table_name") == self.table_name
            )
            print(f"[WM-TS] Watermark reset for {self.table_name}")
        except Exception:
            print("[WM-TS] Nothing to reset.")


# Instantiate the watermark store
ts_wm = TimestampWatermarkStore(
    store_path=WATERMARK_PATH,
    table_name="orders",
    lookback_seconds=300   # 5-minute buffer for late records
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Strategy B — Sequential ID-Based Watermark
# MAGIC Track the maximum `order_id` processed so far.
# MAGIC
# MAGIC **Best for:** Tables with monotonically increasing surrogate keys (auto-increment PKs).
# MAGIC
# MAGIC **Advantage:** Immune to clock skew, no lookback buffer needed.
# MAGIC
# MAGIC **Weakness:** Does NOT capture updates to existing rows — only insertions.

# COMMAND ----------

class SequentialIDWatermarkStore:
    """
    Manages a sequential-ID-based high-water mark.
    Tracks the maximum processed integer ID from the source.
    """

    def __init__(self, store_path: str, table_name: str):
        self.store_path = f"{store_path}_id"
        self.table_name = table_name

    def get(self):
        """Return the last processed max ID, or -1 on first run."""
        try:
            wm_df   = spark.read.format("delta").load(self.store_path)
            max_id  = wm_df.filter(
                col("table_name") == self.table_name
            ).select(spark_max("last_id")).collect()[0][0]

            if max_id is None:
                return -1

            print(f"[WM-ID] Last processed ID: {max_id:,}")
            return max_id
        except Exception:
            print("[WM-ID] First run — starting from ID 0.")
            return -1

    def save(self, new_max_id: int):
        wm_schema = "table_name STRING, last_id LONG, updated_at TIMESTAMP"
        wm_df = spark.createDataFrame(
            [(self.table_name, int(new_max_id), datetime.utcnow())],
            schema=wm_schema
        )
        wm_df.write.format("delta").mode("overwrite").save(self.store_path)
        print(f"[WM-ID] Watermark advanced to ID: {new_max_id:,}")


id_wm = SequentialIDWatermarkStore(
    store_path=WATERMARK_PATH,
    table_name="orders"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run the Incremental Pipeline — Timestamp Strategy
# MAGIC First run has no watermark → processes all 75,000 rows.

# COMMAND ----------

def incremental_load_with_ts_watermark(bronze_path, silver_path, wm_store):
    """
    Full incremental load cycle using a timestamp watermark.
    Steps:
      1. Read watermark
      2. Filter Bronze to only new rows
      3. Transform
      4. Append to Silver
      5. Advance watermark
    """
    print("\n" + "="*60)
    print("  INCREMENTAL LOAD — Timestamp Watermark Strategy")
    print("="*60)

    last_ts = wm_store.get()
    df_bronze = spark.read.format("delta").load(bronze_path)

    # Filter: only rows newer than the watermark
    if last_ts:
        df_new = df_bronze.filter(col("created_ts") > lit(last_ts))
        print(f"Filtering rows with created_ts > {last_ts}")
    else:
        df_new = df_bronze
        print("No watermark — loading all rows (initial backfill).")

    row_count = df_new.count()
    print(f"Rows to process: {row_count:,}")

    if row_count == 0:
        print("Nothing to process. Pipeline exits cleanly.")
        return

    # Transform
    df_silver = (
        df_new
        .withColumn("order_amount_usd", col("order_amount").cast("double"))
        .withColumn("load_ts",          current_timestamp())
        .withColumn("wm_strategy",      lit("TIMESTAMP"))
    )

    # Append to Silver
    (df_silver
        .write
        .format("delta")
        .mode("append")
        .save(silver_path))

    # Advance watermark to the latest record we just processed
    new_max_ts = df_new.select(spark_max("created_ts")).collect()[0][0]
    wm_store.save(new_max_ts)

    print(f"\n✔ Load complete. {row_count:,} rows written to Silver.")
    print("="*60 + "\n")


# Run 1: Full backfill (no watermark)
incremental_load_with_ts_watermark(BRONZE_PATH, SILVER_PATH, ts_wm)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Simulate New Data Arriving + Run 2

# COMMAND ----------

# Bronze path for orders table
print(f"Bronze path: {BRONZE_PATH}")

# New batch: 8,000 orders in the last hour
new_batch = create_orders_batch(start_id=75_001, n_rows=8_000, ts_offset_hours=0)

(new_batch
    .write
    .format("delta")
    .mode("append")
    .save(BRONZE_PATH))

print(f"New batch appended: {new_batch.count():,} rows (IDs 75,001 – 83,000)")

# Run 2: Only processes the 8,000 new rows
incremental_load_with_ts_watermark(BRONZE_PATH, SILVER_PATH, ts_wm)

# COMMAND ----------

# MAGIC %sql
# MAGIC select created_at,count(*) from datapipeline2026.engmarch2026.bronze_orders
# MAGIC group by all
# MAGIC order by created_at desc limit 100;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select `_transformed_at`,order_id,count(*) from datapipeline2026.engmarch2026.silver_orders
# MAGIC group by all
# MAGIC order by `_transformed_at` desc, order_id desc
# MAGIC limit 100;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Failure Recovery — What Happens if the Cluster Dies Mid-Run?
# MAGIC
# MAGIC The watermark is only advanced **after** a successful write to Silver.
# MAGIC
# MAGIC If the cluster fails before `wm_store.save()` executes:
# MAGIC - The watermark stays at the **previous value**
# MAGIC - The next run re-reads the same window of data
# MAGIC - Duplicate rows may be written to Silver (append mode)
# MAGIC - **Solution:** Use MERGE (Notebook 3) instead of append for idempotency

# COMMAND ----------

print("""
╔══════════════════════════════════════════════════════════╗
║        WATERMARK FAILURE RECOVERY FLOW                   ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  [Cluster Start]                                         ║
║        │                                                 ║
║        ▼                                                 ║
║  Read Watermark from Delta Store                         ║
║        │                                                 ║
║        ▼                                                 ║
║  Filter Bronze → Only rows > last watermark              ║
║        │                                                 ║
║        ▼                                                 ║
║  Transform + Write to Silver                             ║
║        │                                                 ║
║        ▼    ← FAILURE POINT: If cluster dies here...     ║
║  Advance Watermark (save new max_ts / max_id)            ║
║        │                                                 ║
║        ▼                                                 ║
║  [Pipeline Complete]                                     ║
║                                                          ║
║  Recovery: Re-run pipeline → watermark unchanged →       ║
║  Same data window re-read → MERGE prevents duplicates    ║
╚══════════════════════════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Watermark State Inspection
# MAGIC Always be able to audit the state of your watermark store.

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from datapipeline2026.engmarch2026.silver_orders
# MAGIC
# MAGIC limit 100;

# COMMAND ----------

print(SILVER_PATH)

# COMMAND ----------

print("\n--- Current Timestamp Watermark ---")
spark.read.format("delta").load(f"{WATERMARK_PATH}_ts").show(truncate=False)

print("\n--- Silver Table Row Count ---")
silver_count = spark.read.format("delta").load(SILVER_PATH).count()
print(f"Silver rows: {silver_count:,}")

print("\n--- Watermark Store History ---")
DeltaTable.forPath(spark, f"{WATERMARK_PATH}_ts").history(5).select(
    "version", "timestamp", "operation", "operationMetrics"
).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Key Takeaways
# MAGIC
# MAGIC | Feature | Timestamp WM | Sequential ID WM |
# MAGIC |---|---|---|
# MAGIC | **Captures updates** | ✅ Yes (via `updated_ts`) | ❌ No — inserts only |
# MAGIC | **Clock skew risk** | ⚠️ Yes — use lookback buffer | ✅ None |
# MAGIC | **Late arrival handling** | Buffer subtracts N seconds | Not applicable |
# MAGIC | **Best source type** | CDC / event streams | OLTP auto-increment PKs |
# MAGIC | **Failure safe?** | Only if MERGE is used downstream | Only if MERGE is used downstream |
# MAGIC
# MAGIC ---
# MAGIC > **Next:** Notebook 3 — Executing High-Performance MERGE Operations
