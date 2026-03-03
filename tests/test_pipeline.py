#!/usr/bin/env python3
# =============================================================================
# test_pipeline.py
# US Accidents Big Data ML Assignment - Unit & Integration Tests
# Tests core pipeline components: ingestion, feature engineering, model training
# Usage: python -m pytest tests/test_pipeline.py -v
# =============================================================================

import os
import sys
import json
import tempfile
import pytest
import numpy as np
import pandas as pd

# ---- Add project root to path ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---- PySpark imports ----
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, BooleanType, TimestampType
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark import StorageLevel


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def spark():
    """Create a shared SparkSession for all tests."""
    spark = (
        SparkSession.builder
        .appName("USAccidents_Tests")
        .master("local[2]")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def sample_accidents_df(spark):
    """Create a small synthetic accidents DataFrame for testing."""
    schema = StructType([
        StructField("ID", StringType(), True),
        StructField("Source", StringType(), True),
        StructField("Severity", IntegerType(), True),
        StructField("Start_Time", TimestampType(), True),
        StructField("End_Time", TimestampType(), True),
        StructField("Start_Lat", DoubleType(), True),
        StructField("Start_Lng", DoubleType(), True),
        StructField("Temperature(F)", DoubleType(), True),
        StructField("Humidity(%)", DoubleType(), True),
        StructField("Visibility(mi)", DoubleType(), True),
        StructField("Wind_Speed(mph)", DoubleType(), True),
        StructField("Precipitation(in)", DoubleType(), True),
        StructField("Distance(mi)", DoubleType(), True),
        StructField("Weather_Condition", StringType(), True),
        StructField("State", StringType(), True),
        StructField("Sunrise_Sunset", StringType(), True),
    ])

    # Create 100 synthetic rows
    np.random.seed(42)
    n = 100
    data = [
        (
            f"A-{i}", "Source1",
            int(np.random.choice([1, 2, 3, 4])),
            pd.Timestamp("2021-06-01 08:30:00"),
            pd.Timestamp("2021-06-01 09:00:00"),
            float(np.random.uniform(25, 48)),
            float(np.random.uniform(-122, -70)),
            float(np.random.uniform(20, 100)) if np.random.random() > 0.1 else None,
            float(np.random.uniform(20, 100)) if np.random.random() > 0.05 else None,
            float(np.random.uniform(0, 10)),
            float(np.random.uniform(0, 50)) if np.random.random() > 0.1 else None,
            float(np.random.uniform(0, 1)) if np.random.random() > 0.3 else None,
            float(np.random.uniform(0.01, 5)),
            np.random.choice(["Clear", "Cloudy", "Rain", "Snow", None]),
            np.random.choice(["CA", "TX", "FL", "NY", "OH"]),
            np.random.choice(["Day", "Night"]),
        )
        for i in range(n)
    ]

    return spark.createDataFrame(data, schema=schema)


# =============================================================================
# TEST GROUP 1: DATA INGESTION VALIDATION
# =============================================================================

class TestDataIngestion:

    def test_schema_field_count(self, sample_accidents_df):
        """Test that the DataFrame has expected columns."""
        assert len(sample_accidents_df.columns) == 16

    def test_severity_range(self, sample_accidents_df):
        """Severity values must be in {1, 2, 3, 4}."""
        invalid = sample_accidents_df.filter(
            ~F.col("Severity").isin([1, 2, 3, 4])
        ).count()
        assert invalid == 0, f"Found {invalid} rows with invalid Severity values."

    def test_coordinate_range(self, sample_accidents_df):
        """Latitude must be in [24, 72], Longitude in [-170, -66] for US."""
        invalid = sample_accidents_df.filter(
            (F.col("Start_Lat") < 24) | (F.col("Start_Lat") > 72) |
            (F.col("Start_Lng") < -170) | (F.col("Start_Lng") > -66)
        ).count()
        assert invalid == 0, f"Found {invalid} out-of-range coordinates."

    def test_row_count_nonzero(self, sample_accidents_df):
        """Dataset must have at least 1 row."""
        assert sample_accidents_df.count() > 0

    def test_no_all_null_columns(self, sample_accidents_df):
        """No column should be 100% null."""
        total = sample_accidents_df.count()
        for col in ["Severity", "State", "Start_Lat"]:
            null_count = sample_accidents_df.filter(F.col(col).isNull()).count()
            assert null_count < total, f"Column '{col}' is entirely null."


# =============================================================================
# TEST GROUP 2: FEATURE ENGINEERING
# =============================================================================

class TestFeatureEngineering:

    def test_temporal_features_extraction(self, spark, sample_accidents_df):
        """Test that temporal features are correctly extracted."""
        df = (
            sample_accidents_df
            .withColumn("Hour",      F.hour("Start_Time"))
            .withColumn("Month",     F.month("Start_Time"))
            .withColumn("DayOfWeek", F.dayofweek("Start_Time"))
        )

        # Check Hour range [0, 23]
        hour_min = df.agg(F.min("Hour")).collect()[0][0]
        hour_max = df.agg(F.max("Hour")).collect()[0][0]
        assert 0 <= hour_min <= 23
        assert 0 <= hour_max <= 23

        # Check Month range [1, 12]
        month_val = df.agg(F.max("Month")).collect()[0][0]
        assert 1 <= month_val <= 12

    def test_duration_calculation(self, spark, sample_accidents_df):
        """Test accident duration is non-negative and bounded."""
        df = sample_accidents_df.withColumn(
            "Duration_Min",
            F.round(
                (F.unix_timestamp("End_Time") - F.unix_timestamp("Start_Time")) / 60.0,
                2
            )
        )
        min_dur = df.agg(F.min("Duration_Min")).collect()[0][0]
        assert min_dur >= 0, "Duration should be non-negative."

    def test_vector_assembler(self, spark, sample_accidents_df):
        """Test VectorAssembler creates correct feature vector size."""
        numeric_cols = ["Temperature(F)", "Humidity(%)", "Visibility(mi)"]
        df_filled = sample_accidents_df.fillna(
            {"Temperature(F)": 70.0, "Humidity(%)": 60.0, "Visibility(mi)": 10.0}
        )

        assembler = VectorAssembler(
            inputCols=numeric_cols,
            outputCol="features",
            handleInvalid="keep"
        )
        result = assembler.transform(df_filled)
        feat_size = result.select("features").first()[0].size
        assert feat_size == len(numeric_cols)

    def test_label_encoding(self, spark, sample_accidents_df):
        """Label should be 0-indexed: Severity-1 mapping to label in [0,3]."""
        df_label = sample_accidents_df.withColumn(
            "label", (F.col("Severity") - 1).cast("double")
        )
        min_label = df_label.agg(F.min("label")).collect()[0][0]
        max_label = df_label.agg(F.max("label")).collect()[0][0]
        assert min_label >= 0
        assert max_label <= 3


# =============================================================================
# TEST GROUP 3: DATA SPLITTING
# =============================================================================

class TestDataSplitting:

    def test_split_coverage(self, spark, sample_accidents_df):
        """Train + val + test split should cover all rows within ±5%."""
        df_label = sample_accidents_df.withColumn(
            "label", (F.col("Severity") - 1).cast("double")
        )
        total = df_label.count()

        train, val, test = df_label.randomSplit([0.70, 0.15, 0.15], seed=42)
        split_total = train.count() + val.count() + test.count()

        # Allow for minor statistical variation in randomSplit
        assert abs(split_total - total) <= 2, \
            f"Row count mismatch: {split_total} != {total}"

    def test_no_overlap_in_splits(self, spark, sample_accidents_df):
        """Verify train and test splits are disjoint by ID sampling."""
        df = sample_accidents_df
        train, _, test = df.randomSplit([0.70, 0.15, 0.15], seed=42)

        train_ids = set(row["ID"] for row in train.select("ID").collect())
        test_ids  = set(row["ID"] for row in test.select("ID").collect())
        overlap   = train_ids & test_ids
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping IDs in train/test splits."


# =============================================================================
# TEST GROUP 4: MODEL UTILITIES
# =============================================================================

class TestModelUtilities:

    def test_broadcast_join(self, spark, sample_accidents_df):
        """Test that broadcast join with state lookup adds expected columns."""
        from pyspark.sql.functions import broadcast

        state_data = [("CA", "West"), ("TX", "South"), ("FL", "South"),
                      ("NY", "Northeast"), ("OH", "Midwest")]
        state_df = spark.createDataFrame(state_data, ["State", "Region"])

        enriched = sample_accidents_df.join(broadcast(state_df), on="State", how="left")
        assert "Region" in enriched.columns

    def test_persist_unpersist(self, spark, sample_accidents_df):
        """Test that persist and unpersist cycle works without error."""
        df = sample_accidents_df
        df.persist(StorageLevel.MEMORY_AND_DISK)
        count_cached = df.count()
        df.unpersist()
        count_uncached = df.count()
        assert count_cached == count_uncached


# =============================================================================
# TEST GROUP 5: CONFIGURATION VALIDATION
# =============================================================================

class TestConfiguration:

    def test_spark_config_yaml_exists(self):
        """Check that spark_config.yaml exists in config directory."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "spark_config.yaml"
        )
        assert os.path.exists(config_path), f"spark_config.yaml not found at {config_path}"

    def test_spark_config_yaml_valid(self):
        """Check that spark_config.yaml can be parsed and has required keys."""
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "spark_config.yaml"
        )
        if not os.path.exists(config_path):
            pytest.skip("spark_config.yaml not found - run setup first")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        required_keys = ["app_name", "master", "driver_memory", "executor_memory",
                         "shuffle_partitions"]
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

    def test_tableau_config_json_valid(self):
        """Check that tableau_config.json is valid JSON with required fields."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "tableau_config.json"
        )
        if not os.path.exists(config_path):
            pytest.skip("tableau_config.json not found")

        with open(config_path, "r") as f:
            config = json.load(f)

        assert "data_sources" in config
        assert "dashboard_layout" in config
        assert len(config["dashboard_layout"]) == 4, "Expected 4 dashboards"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
