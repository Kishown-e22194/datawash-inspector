"""
datawash — A Python Data Cleaning & EDA Library
=====================================================
Provides a unified ``DataPipeline`` class for end-to-end data preparation,
and a standalone ``PlottingMethods`` factory for generating embeddable HTML charts.

Inheritance chain:
    autoprep.DataPipeline          → Step 1: Ingestion, sanitization, type correction
      └─ cleaner.DataPipeline      → Step 2: Summary, health report, imputation, outliers
           └─ eda_engine.DataPipeline → Step 3+4: Feature engineering & visualisation
                └─ stats_engine.InsightsEngine → Step 5: Unified statistical associations

Usage::

    from datawash import DataPipeline

    pipe = DataPipeline()

    # Step 1 — Ingest & Prep
    pipe.load_data('data.csv').sanitize_garbage().auto_type_correct()

    # Step 2 — Explore & Clean
    pipe.data_health_report()
    pipe.handle_missing_values(strategy='auto').remove_duplicates()

    # Step 3 — Feature Engineering (leak-safe)
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(pipe.df, test_size=0.2)
    pipe.fit_transform_ml(train, num_strategy='standard', cat_strategy='onehot')
    test_processed = pipe.transform_ml(test)

    # Or quick (full-dataset) convenience wrapper:
    # pipe.prepare_for_ml(num_strategy='standard', cat_strategy='onehot')

    # Step 4 — Visualise
    pipe.plot_correlation_matrix()
    pipe.plot_missing_values()
    pipe.univariate_all()

    # Step 5 — Statistical Associations
    pipe.plot_all_associations_heatmap()

    # Export
    pipe.save_data('cleaned.csv')
"""

from .stats_engine import InsightsEngine as DataPipeline
from .plot_factory import PlottingMethods, compose_report

__all__ = ['DataPipeline', 'PlottingMethods', 'compose_report']
__version__ = '0.3.0'
