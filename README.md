# US Accidents Severity Classification

A comprehensive big data machine learning project for predicting traffic accident severity using Apache Spark MLlib on 4.2 million accident records.

## Overview

This project implements distributed machine learning algorithms to classify US traffic accidents into four severity levels using the US Accidents (March 2023) dataset. The system leverages Apache Spark for scalable data processing and MLlib for model training, achieving 62.21% multiclass test accuracy with Decision Tree and an AUC-ROC of 0.7815 with GBT binary classification.

## Key Features

- **Large-Scale Processing**: Handles 4.2M records with distributed PySpark processing
- **Multiple ML Models**: Logistic Regression, Random Forest, Decision Tree, and Gradient Boosted Trees
- **Advanced Feature Engineering**: 48 engineered features from temporal, meteorological, and geographical data
- **Hyperparameter Tuning**: Cross-validated model optimization with ParamGridBuilder
- **Statistical Validation**: Bootstrap confidence intervals and model comparison
- **Scalability Analysis**: Strong and weak scaling experiments across partition configurations
- **Business Intelligence**: Four Tableau dashboards for stakeholder insights



## Requirements

- Python 3.8+
- Apache Spark 3.5.0
- PySpark
- scikit-learn
- pandas, numpy, matplotlib, seaborn
- findspark

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install pyspark scikit-learn pandas numpy matplotlib seaborn findspark
   ```

2. **Run the pipeline**:
   Execute notebooks sequentially (1 → 2 → 3 → 4) or use:
   ```bash
   python code/scripts/run_pipeline.py
   ```

3. **View results**:
   - Model metrics: `code/data/samples/models/training_results.json`
   - Visualizations: `code/data/samples/eda_plots/`
   - Dashboards: `code/tableau/`

## Technical Highlights

- **Data Optimization**: 67% storage reduction via Parquet with Snappy compression (1,600 MB → 533 MB)
- **Efficient Caching**: Strategic MEMORY_AND_DISK persistence at 6 pipeline checkpoints
- **Optimal Configuration**: 4-partition setup identified as best cost-efficiency (55.03%); 8 partitions achieved fastest wall-clock time (3.22 s)
- **Statistical Rigor**: Bootstrap CIs confirm model stability (DT width: 0.0025, RF width: 0.0024)

