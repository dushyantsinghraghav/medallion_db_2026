# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 3: Executing High-Performance MERGE Operations
# MAGIC **Module:** Data Load Strategies | Medallion Architecture Pipeline Optimization
# MAGIC
# MAGIC ---
# MAGIC ## Learning Objectives
# MAGIC - Write MERGE statements that handle INSERT, UPDATE, and DELETE in one atomic pass
# MAGIC - Configure schema evolution to protect Silver and Gold layers from source changes
# MAGIC - Optimise join conditions to prevent expensive data shuffles across distributed nodes
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup

# COMMAND ----------

# DBTITLE 1,Cell 3
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, lit, rand, expr,
    max as spark_max, when
)
from delta.tables import DeltaTable
import time

spark = SparkSession.builder \
    .config("spark.databricks.delta.schema.autoMerge.enabled", "true") \
    .getOrCreate()

username  = spark.sql("SELECT current_user()").collect()[0][0]
BASE_PATH = f"/Volumes/datapipeline2026/default/pipeline_module"

BRONZE_PATH     = f"{BASE_PATH}/bronze/customers"
SILVER_PATH     = f"{BASE_PATH}/silver/customers"
WATERMARK_PATH  = f"{BASE_PATH}/watermark/customers_wm"

print("Spark session ready.")
#print(f"Schema auto-merge: {spark.conf.get('spark.databricks.delta.schema.autoMerge.enabled')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the Customer Master Table
# MAGIC This represents a slowly changing dimension (SCD Type 1) from a CRM system.
# MAGIC Records can be **inserted** (new customers), **updated** (address/email changes),
# MAGIC or **soft-deleted** (is_active = False).

# COMMAND ----------

print(BRONZE_PATH)

# COMMAND ----------

# DBTITLE 1,Cell 5
# --- Initial state: 100,000 customer records ---
df_initial = (
    spark.range(1, 100_001)
    .select(
        col("id").cast("long").alias("customer_id"),
        expr("concat('User_', cast(id as string))").alias("full_name"),
        expr("concat('user', cast(id as string), '@example.com')").alias("email"),
        expr("CASE WHEN rand() < 0.3 THEN 'US' "
             "     WHEN rand() < 0.6 THEN 'UK' "
             "     ELSE 'AU' END").alias("country"),
        lit(True).alias("is_active"),
        (rand() * 10000).alias("lifetime_value"),
        current_timestamp().alias("updated_ts"),
    )
)

(df_initial
    .write
    .format("delta")
    .mode("overwrite")
    .save(BRONZE_PATH))

# Bootstrap Silver with the initial load (no MERGE needed on first run)
(df_initial
    .withColumn("load_ts",       current_timestamp())
    .withColumn("record_source", lit("CRM_INITIAL"))
    .write
    .format("delta")
    .mode("overwrite")
    .save(SILVER_PATH))

print(f"Initial load: {df_initial.count():,} customers written to Bronze and Silver.")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from delta.`/Volumes/datapipeline2026/default/pipeline_module/bronze/customers/` limit 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- i want to see the data in the silver table 
# MAGIC select * from delta.`/Volumes/datapipeline2026/default/pipeline_module/silver/customers/` limit 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Simulate a CDC (Change Data Capture) Batch
# MAGIC The next pipeline run receives a CDC batch containing:
# MAGIC - **20,000 updated** customers (email or address changed)
# MAGIC - **5,000 new** customers (IDs 100,001 – 105,000)
# MAGIC - **1,000 deletes** (is_active set to False)

# COMMAND ----------

# 20,000 updates to existing customers
df_updates = (
    spark.range(1, 20_001)
    .select(
        col("id").cast("long").alias("customer_id"),
        expr("concat('User_', cast(id as string))").alias("full_name"),
        expr("concat('updated_user', cast(id as string), '@newdomain.com')")
            .alias("email"),             # Changed email
        expr("CASE WHEN rand() < 0.5 THEN 'CA' ELSE 'DE' END")
            .alias("country"),           # Changed country
        lit(True).alias("is_active"),
        (rand() * 15000).alias("lifetime_value"),
        current_timestamp().alias("updated_ts"),
    )
)

# 5,000 new inserts
df_inserts = (
    spark.range(100_001, 105_001)
    .select(
        col("id").cast("long").alias("customer_id"),
        expr("concat('NewUser_', cast(id as string))").alias("full_name"),
        expr("concat('new_user', cast(id as string), '@example.com')").alias("email"),
        lit("SG").alias("country"),
        lit(True).alias("is_active"),
        (rand() * 5000).alias("lifetime_value"),
        current_timestamp().alias("updated_ts"),
    )
)

# 1,000 soft deletes
df_deletes = (
    spark.range(50_001, 51_001)
    .select(
        col("id").cast("long").alias("customer_id"),
        expr("concat('User_', cast(id as string))").alias("full_name"),
        expr("concat('user', cast(id as string), '@example.com')").alias("email"),
        lit("US").alias("country"),
        lit(False).alias("is_active"),   # Soft delete flag
        lit(0.0).alias("lifetime_value"),
        current_timestamp().alias("updated_ts"),
    )
)

# Combine CDC batch
df_cdc = df_updates.union(df_inserts).union(df_deletes)

(df_cdc
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(BRONZE_PATH))

print(f"CDC batch size : {df_cdc.count():,} rows")
print(f"  Updates      : {df_updates.count():,}")
print(f"  Inserts      : {df_inserts.count():,}")
print(f"  Soft deletes : {df_deletes.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. MERGE Operation — The Core Pattern
# MAGIC
# MAGIC `MERGE` is a single atomic operation that:
# MAGIC 1. **Matches** source rows to target rows on a join key
# MAGIC 2. **Updates** matched rows when data has changed
# MAGIC 3. **Inserts** unmatched rows (new records)
# MAGIC 4. **Deletes** rows based on a flag (soft delete pattern)
# MAGIC
# MAGIC > **Atomicity guarantee:** Either all changes commit or none do.
# MAGIC > If the cluster fails mid-MERGE, the target table is unchanged.

# COMMAND ----------

t_start = time.time()

# Load the target Silver table as a DeltaTable object
silver_table = DeltaTable.forPath(spark, SILVER_PATH)

# Read the incoming CDC batch
df_source = spark.read.format("delta").load(BRONZE_PATH)

# ── MERGE OPERATION ──────────────────────────────────────────────────────────
(silver_table.alias("target")
    .merge(
        df_source.alias("source"),
        # Join condition: match on the primary key
        "target.customer_id = source.customer_id"
    )
    # WHEN MATCHED + is_active = False → soft delete (mark inactive)
    .whenMatchedUpdate(
        condition="source.is_active = False",
        set={
            "is_active":       "source.is_active",
            "lifetime_value":  "source.lifetime_value",
            "updated_ts":      "source.updated_ts",
            "load_ts":         "current_timestamp()",
        }
    )
    # WHEN MATCHED + data actually changed → update
    .whenMatchedUpdate(
        condition="""
            source.email          != target.email
            OR source.country     != target.country
            OR source.lifetime_value != target.lifetime_value
        """,
        set={
            "full_name":       "source.full_name",
            "email":           "source.email",
            "country":         "source.country",
            "is_active":       "source.is_active",
            "lifetime_value":  "source.lifetime_value",
            "updated_ts":      "source.updated_ts",
            "load_ts":         "current_timestamp()",
        }
    )
    # WHEN NOT MATCHED → insert new record
    .whenNotMatchedInsert(
        values={
            "customer_id":     "source.customer_id",
            "full_name":       "source.full_name",
            "email":           "source.email",
            "country":         "source.country",
            "is_active":       "source.is_active",
            "lifetime_value":  "source.lifetime_value",
            "updated_ts":      "source.updated_ts",
            "load_ts":         "current_timestamp()",
            "record_source":   "'CRM_CDC'",
        }
    )
    .execute()
)
# ─────────────────────────────────────────────────────────────────────────────

elapsed = time.time() - t_start
print(f"\nMERGE completed in {elapsed:.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validate the MERGE Results

# COMMAND ----------

# Check row counts
silver_count = spark.read.format("delta").load(SILVER_PATH).count()
print(f"Silver row count after MERGE: {silver_count:,}")
print(f"Expected: 105,000 (100K original + 5K new inserts)")

# Verify updates applied
print("\n--- Sample of updated customers (should show @newdomain.com) ---")
(spark.read.format("delta").load(SILVER_PATH)
    .filter(col("customer_id") <= 5)
    .select("customer_id", "email", "country", "updated_ts", "load_ts")
    .show(5, truncate=False))

# Verify soft deletes
print("--- Soft-deleted customers (is_active = False) ---")
soft_deleted = (spark.read.format("delta").load(SILVER_PATH)
    .filter(col("is_active") == False)
    .count())
print(f"Inactive customers: {soft_deleted:,} (expected 1,000)")

# Verify new inserts
print("--- New customers (IDs 100,001+) ---")
new_customers = (spark.read.format("delta").load(SILVER_PATH)
    .filter(col("customer_id") > 100_000)
    .count())
print(f"New customers: {new_customers:,} (expected 5,000)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Schema Evolution — Handling Source Schema Changes
# MAGIC
# MAGIC Real pipelines must survive upstream schema changes.
# MAGIC Delta Lake supports schema evolution via `mergeSchema` and `autoMerge`.
# MAGIC
# MAGIC ### Scenario: Source adds a new column `loyalty_tier`

# COMMAND ----------

# Source adds a new column — this would fail without schema evolution enabled
df_new_schema = (
    spark.range(105_001, 106_001)
    .select(
        col("id").cast("long").alias("customer_id"),
        expr("concat('NewUser_', cast(id as string))").alias("full_name"),
        expr("concat('new_user', cast(id as string), '@example.com')").alias("email"),
        lit("US").alias("country"),
        lit(True).alias("is_active"),
        (rand() * 8000).alias("lifetime_value"),
        current_timestamp().alias("updated_ts"),
        # NEW COLUMN — not present in current Silver schema
        expr("CASE WHEN rand() < 0.33 THEN 'BRONZE' "
             "     WHEN rand() < 0.66 THEN 'SILVER' "
             "     ELSE 'GOLD' END").alias("loyalty_tier"),
    )
)

print("New source schema:")
df_new_schema.printSchema()

# COMMAND ----------

# DBTITLE 1,Cell 17
-- i want to see the data in the silver table 
describe delta.`/Volumes/datapipeline2026/default/pipeline_module/silver/customers/`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Optimising MERGE — Preventing Expensive Shuffles
# MAGIC
# MAGIC The most common MERGE performance problem is a **full shuffle**:
# MAGIC Delta must scan every file in the target to find matching rows.
# MAGIC
# MAGIC ### Optimisation 1: Z-Order by join key
# MAGIC Collocates rows with similar `customer_id` values in the same files,
# MAGIC allowing Delta to skip irrelevant files via data-skipping statistics.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- i want to see the data in the silver table 
# MAGIC describe  history delta.`/Volumes/datapipeline2026/default/pipeline_module/silver/customers/`
# MAGIC

# COMMAND ----------

# Z-Order the Silver table by the MERGE join key
spark.sql(f"OPTIMIZE delta.`{SILVER_PATH}` ZORDER BY (customer_id)")
print("Z-ORDER applied to Silver table on customer_id.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimisation 2: Partition the Source Before MERGE
# MAGIC Repartition the source DataFrame on the join key to align shuffle partitions
# MAGIC with the target's file layout. This reduces cross-node data movement.

# COMMAND ----------

# Without partitioning: random data distribution across executors
df_unpartitioned = spark.read.format("delta").load(BRONZE_PATH)

# With partitioning: rows with the same customer_id land on the same executor
df_partitioned = (
    spark.read
    .format("delta")
    .load(BRONZE_PATH)
    .repartition(200, col("customer_id"))   # Align with join key
)

print(f"Unpartitioned partitions : {df_unpartitioned.rdd.getNumPartitions()}")
print(f"Repartitioned partitions : {df_partitioned.rdd.getNumPartitions()}")
print("""
When MERGE executes, Spark shuffles source and target data by the join key.
Pre-partitioning the source ensures that shuffle is minimal — data is already
co-located. This is critical at 100M+ row scale.
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimisation 3: Filter Source to Affected Partitions Only
# MAGIC Never MERGE an entire source table when you only need a date partition.
# MAGIC Narrow the source filter to the specific partition window.

# COMMAND ----------

# BAD: Merging all history every run
df_bad_source = spark.read.format("delta").load(BRONZE_PATH)
print(f"[BAD]  Source rows (no filter): {df_bad_source.count():,}")

# GOOD: Only merge records updated in the last 24 hours
from pyspark.sql.functions import date_sub, current_date
df_good_source = (
    spark.read.format("delta").load(BRONZE_PATH)
    .filter(col("updated_ts") >= date_sub(current_date(), 1))
)
print(f"[GOOD] Source rows (24h filter): {df_good_source.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. MERGE Operation Metrics
# MAGIC Delta records per-operation metrics in the transaction log.

# COMMAND ----------

history = DeltaTable.forPath(spark, SILVER_PATH).history(3)
(history
    .select("version", "timestamp", "operation", "operationMetrics")
    .show(truncate=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Key Takeaways
# MAGIC
# MAGIC | Concept | Detail |
# MAGIC |---|---|
# MAGIC | **MERGE atomicity** | All-or-nothing — safe on cluster failure |
# MAGIC | **Condition ordering** | Evaluated top-to-bottom; put most selective condition first |
# MAGIC | **Schema evolution** | Enable `autoMerge` + use `whenNotMatchedInsertAll()` |
# MAGIC | **Z-Order** | Apply on join key to enable file-skipping during MERGE |
# MAGIC | **Pre-partition source** | `repartition(n, joinKey)` reduces shuffle cost |
# MAGIC | **Filter source window** | Never merge 100M rows to update 50K — filter first |
# MAGIC
# MAGIC ---
# MAGIC > **Next:** Notebook 4 — Partition Offsets and Idempotency Guarantees
