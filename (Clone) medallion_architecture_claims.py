# Databricks notebook source
# MAGIC %md
# MAGIC # Medallion Architecture — HealthVerity Claims
# MAGIC **Source:** `samples.healthverity.claims_sample_synthetic`
# MAGIC
# MAGIC | Layer  | Table / View | Purpose |
# MAGIC |--------|-------------|---------|
# MAGIC | 🥉 Bronze | `bronze_claims` | Raw ingestion, no changes |
# MAGIC | 🥈 Silver | `silver_claims` + view | Cleaned, typed, deduplicated |
# MAGIC | 🥇 Gold  | 3 tables + 3 views | Business-level aggregations |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 · Imports & Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType

# COMMAND ----------

DB_NAME = "datapipeline2026.engmarch2026"

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
spark.sql(f"USE {DB_NAME}")
print(f"✅ Using database: {DB_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🥉 Bronze Layer — Raw Ingestion

# COMMAND ----------

# Read source table exactly as-is — zero transformations
bronze_df = spark.table("datapipeline2026.engmarch2026.claims_sample_synthetic")
bronze_df.printSchema()

# COMMAND ----------

# Write to Bronze Delta table
(
    bronze_df.write
    .format("delta")
    .mode("overwrite")               # swap for "append" in incremental pipelines
    .option("overwriteSchema", "true")
    .saveAsTable(f"{DB_NAME}.bronze_claims")
)

print("✅ Bronze table created: samples.healthverity.bronze_claims")

# COMMAND ----------

# Sanity check — row count
spark.sql(f"SELECT COUNT(*) AS row_count FROM {DB_NAME}.bronze_claims").show()

# COMMAND ----------

# Sanity check — sample rows
spark.sql(f"SELECT * FROM {DB_NAME}.bronze_claims LIMIT 5").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🥈 Silver Layer — Cleansed & Conformed

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 · Load Bronze & Normalise Column Names

# COMMAND ----------

bronze = spark.table(f"{DB_NAME}.bronze_claims")

# Lowercase + strip whitespace from every column name
bronze = bronze.toDF(*[c.strip().lower().replace(" ", "_") for c in bronze.columns])

actual_cols = set(bronze.columns)
print("Available columns:", sorted(actual_cols))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 · Cast Date Columns

# COMMAND ----------

silver_df = (
    bronze
    .withColumn(
        "service_date",
        F.to_date(
            F.col("service_date")         if "service_date"    in actual_cols
            else F.col("date_of_service") if "date_of_service" in actual_cols
            else F.lit(None),
            "yyyy-MM-dd"
        )
    )
    .withColumn(
        "claim_date",
        F.to_date(
            F.col("claim_date")               if "claim_date"         in actual_cols
            else F.col("adjudication_date")   if "adjudication_date"  in actual_cols
            else F.lit(None),
            "yyyy-MM-dd"
        )
    )
)

print("✅ Date columns cast")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 · Cast Numeric Columns

# COMMAND ----------

silver_df = (
    silver_df
    .withColumn(
        "paid_amount",
        F.col("paid_amount").cast(DoubleType())    if "paid_amount"    in actual_cols
        else F.lit(None).cast(DoubleType())
    )
    .withColumn(
        "allowed_amount",
        F.col("allowed_amount").cast(DoubleType()) if "allowed_amount" in actual_cols
        else F.lit(None).cast(DoubleType())
    )
    .withColumn(
        "quantity",
        F.col("quantity").cast(IntegerType())      if "quantity"       in actual_cols
        else F.lit(None).cast(IntegerType())
    )
)

print("✅ Numeric columns cast")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 · Standardise String Columns

# COMMAND ----------

silver_df = (
    silver_df
    .withColumn(
        "claim_type",
        F.upper(F.trim(F.col("claim_type"))) if "claim_type" in actual_cols
        else F.lit(None).cast(StringType())
    )
    .withColumn(
        "diagnosis_code",
        F.trim(F.col("diagnosis_code")) if "diagnosis_code" in actual_cols
        else F.trim(F.col("icd_code"))  if "icd_code"       in actual_cols
        else F.lit(None).cast(StringType())
    )
    .withColumn(
        "procedure_code",
        F.trim(F.col("procedure_code")) if "procedure_code" in actual_cols
        else F.trim(F.col("cpt_code"))  if "cpt_code"       in actual_cols
        else F.lit(None).cast(StringType())
    )
    .withColumn(
        "patient_id",
        F.trim(F.col("patient_id"))     if "patient_id"  in actual_cols
        else F.trim(F.col("member_id")) if "member_id"   in actual_cols
        else F.lit(None).cast(StringType())
    )
    .withColumn(
        "provider_id",
        F.trim(F.col("provider_id")) if "provider_id" in actual_cols
        else F.trim(F.col("npi"))    if "npi"          in actual_cols
        else F.lit(None).cast(StringType())
    )
)

print("✅ String columns standardised")

# COMMAND ----------

display(silver_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 · Filter Invalid Records

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 · Deduplicate & Add Audit Columns

# COMMAND ----------

dedup_keys = [c for c in ["claim_id", "patient_id", "service_date", "procedure_code"]
              if c in actual_cols]

silver_df = (
    silver_df
    .dropDuplicates(dedup_keys)
    .withColumn("silver_load_ts", F.current_timestamp())
    .withColumn("source_table",   F.lit("datapipeline2026.engmarch2026.claims_sample_synthetic"))
)

print(f"✅ Dedup keys used   : {dedup_keys}")
print(f"✅ Final silver rows : {silver_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 · Write Silver Delta Table

# COMMAND ----------

(
    silver_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{DB_NAME}.silver_claims")
)

print("✅ Silver table created: medallion_health.silver_claims")
silver_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8 · Create Silver View

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE VIEW {DB_NAME}.vw_silver_claims AS
    SELECT *
    FROM   {DB_NAME}.silver_claims
    WHERE  silver_load_ts IS NOT NULL
""")

print("✅ Silver view created: medallion_health.vw_silver_claims")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 9 · Silver Quality Check

# COMMAND ----------

spark.sql(f"""
    SELECT
        COUNT(*)                        AS total_records,
        COUNT(DISTINCT patient_id)      AS unique_patients,
        COUNT(DISTINCT procedure_code)  AS unique_procedures,
        ROUND(AVG(paid_amount), 2)      AS avg_paid_amount,
        MIN(service_date)               AS earliest_service,
        MAX(service_date)               AS latest_service
    FROM {DB_NAME}.silver_claims
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🥇 Gold Layer — Business-Level Aggregations

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gold Table 1 · Patient-Level Summary

# COMMAND ----------

gold_patient_df = spark.sql(f"""
    SELECT
        patient_id,
        COUNT(*)                                       AS total_claims,
        COUNT(DISTINCT procedure_code)                 AS distinct_procedures,
        COUNT(DISTINCT diagnosis_code)                 AS distinct_diagnoses,
        COUNT(DISTINCT provider_id)                    AS distinct_providers,
        ROUND(SUM(paid_amount),    2)                  AS total_paid,
        ROUND(AVG(paid_amount),    2)                  AS avg_paid_per_claim,
        ROUND(SUM(allowed_amount), 2)                  AS total_allowed,
        MIN(service_date)                              AS first_service_date,
        MAX(service_date)                              AS last_service_date,
        DATEDIFF(MAX(service_date), MIN(service_date)) AS patient_tenure_days
    FROM {DB_NAME}.silver_claims
    GROUP BY patient_id
""")

gold_patient_df.show(5, truncate=False)

# COMMAND ----------

(
    gold_patient_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{DB_NAME}.gold_patient_summary")
)

print("✅ Gold table created: medallion_health.gold_patient_summary")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gold Table 2 · Monthly Spend Trend

# COMMAND ----------

gold_monthly_df = spark.sql(f"""
    SELECT
        DATE_FORMAT(service_date, 'yyyy-MM')  AS year_month,
        claim_type,
        COUNT(*)                              AS claim_count,
        COUNT(DISTINCT patient_id)            AS unique_patients,
        ROUND(SUM(paid_amount),    2)         AS total_paid,
        ROUND(AVG(paid_amount),    2)         AS avg_paid,
        ROUND(SUM(allowed_amount), 2)         AS total_allowed
    FROM {DB_NAME}.silver_claims
    WHERE service_date IS NOT NULL
    GROUP BY DATE_FORMAT(service_date, 'yyyy-MM'), claim_type
    ORDER BY year_month, claim_type
""")

gold_monthly_df.show(5, truncate=False)

# COMMAND ----------

(
    gold_monthly_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{DB_NAME}.gold_monthly_spend")
)

print("✅ Gold table created: medallion_health.gold_monthly_spend")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gold Table 3 · Procedure Cost Roll-Up

# COMMAND ----------

gold_procedure_df = spark.sql(f"""
    SELECT
        procedure_code,
        COUNT(*)                       AS usage_count,
        COUNT(DISTINCT patient_id)     AS patients_affected,
        ROUND(SUM(paid_amount), 2)     AS total_paid,
        ROUND(AVG(paid_amount), 2)     AS avg_paid,
        ROUND(MIN(paid_amount), 2)     AS min_paid,
        ROUND(MAX(paid_amount), 2)     AS max_paid
    FROM {DB_NAME}.silver_claims
    WHERE procedure_code IS NOT NULL
    GROUP BY procedure_code
    ORDER BY total_paid DESC
""")

gold_procedure_df.show(5, truncate=False)

# COMMAND ----------

(
    gold_procedure_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{DB_NAME}.gold_procedure_cost")
)

print("✅ Gold table created: medallion_health.gold_procedure_cost")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gold Views

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE VIEW {DB_NAME}.vw_gold_patient_summary AS
    SELECT *
    FROM   {DB_NAME}.gold_patient_summary
    ORDER  BY total_paid DESC
""")

print("✅ View created: medallion_health.vw_gold_patient_summary")

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE VIEW {DB_NAME}.vw_gold_monthly_spend AS
    SELECT *
    FROM   {DB_NAME}.gold_monthly_spend
    ORDER  BY year_month, claim_type
""")

print("✅ View created: medallion_health.vw_gold_monthly_spend")

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE VIEW {DB_NAME}.vw_gold_procedure_cost AS
    SELECT *
    FROM   {DB_NAME}.gold_procedure_cost
    ORDER  BY total_paid DESC
    LIMIT  50
""")

print("✅ View created: medallion_health.vw_gold_procedure_cost")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Summary — All Objects Created

# COMMAND ----------

spark.sql(f"SHOW TABLES IN {DB_NAME}").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Object Inventory
# MAGIC
# MAGIC | Layer  | Type  | Name                        | Description                      |
# MAGIC |--------|-------|-----------------------------|----------------------------------|
# MAGIC | Bronze | Table | `bronze_claims`             | Raw source copy (Delta)          |
# MAGIC | Silver | Table | `silver_claims`             | Cleaned, typed, deduplicated     |
# MAGIC | Silver | View  | `vw_silver_claims`          | Live view of silver table        |
# MAGIC | Gold   | Table | `gold_patient_summary`      | Per-patient aggregated metrics   |
# MAGIC | Gold   | Table | `gold_monthly_spend`        | Monthly spend by claim type      |
# MAGIC | Gold   | Table | `gold_procedure_cost`       | Procedure-level cost roll-up     |
# MAGIC | Gold   | View  | `vw_gold_patient_summary`   | Patient summary, ranked by spend |
# MAGIC | Gold   | View  | `vw_gold_monthly_spend`     | Monthly trend view               |
# MAGIC | Gold   | View  | `vw_gold_procedure_cost`    | Top 50 procedures by total paid  |
