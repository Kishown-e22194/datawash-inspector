# DataWash 🔬

**DataWash Inspector** is an automated, end-to-end Python data pipeline that cleans, prunes, prepares, and visualizes your datasets with zero manual effort. It includes a built-in serverless HTML dashboard for zero-code data exploration in Google Colab!

## Installation

You can install `datawash` via pip:

```bash
pip install datawash-inspector
```

## Quick Start

It only takes a few lines of code to completely sanitize, optimize, and encode your dataset for machine learning.

```python
from datawash import DataPipeline

# 1. Initialize Pipeline
pipe = DataPipeline()

# 2. Load and Auto-Clean
pipe.load_data('your_data.csv')
pipe.sanitize_garbage().auto_type_correct()
pipe.handle_missing_values(strategy='auto')

# 3. Export Clean Data
pipe.save_data('cleaned_data.csv')
```

## The Interactive Dashboard

`datawash` comes with a fully automated EDA (Exploratory Data Analysis) dashboard. 

**For Google Colab / Jupyter Users:**
Instantly inject a serverless, interactive HTML dashboard directly inside your notebook!
```python
# Magically embeds a stunning UI into your Colab cell
pipe.show_dashboard()
```

**For Local IDE Users (VS Code, PyCharm, etc):**
Automatically generate the HTML dashboard and open it in your default web browser!
```python
# Generates the report and pops it open in Chrome/Safari/Edge
pipe.show_dashboard()
```

### Dashboard Features
- **📥 Download Cleaned File:** A built-in button allows you to download your fully cleaned and optimized dataset instantly.
- **Univariate Subplots:** 3-panel layout (Box, Scatter, Histogram) for numeric columns.
- **Categorical Frequencies:** Bar charts with percentage labels.
- **Smart Relationships:** Automatically detects the top 5 most highly correlated features and plots them using dynamic chart routers (Num vs Num -> Scatter + OLS, Cat vs Num -> Box, Cat vs Cat -> Grouped Bar).
- **Correlations Heatmap:** Advanced Cramér's V, Eta, and Spearman correlation matrix.

## 📊 Standalone Smart Plotting

Don't want to load the full dashboard? You can generate individual charts directly in your notebook instantly!

```python
# 1. 3-Panel Univariate Subplots
pipe.univariate_subplots('price')

# 2. Smart Relationship Router (Auto-detects data types!)
pipe.plot_relationship('developer', 'price') # Generates Box Plot
pipe.plot_relationship('price', 'user_rating') # Generates Scatter + OLS

# 3. Categorical Frequencies with Percentages
pipe.plot_categorical_frequency('primary_genre')

# 4. Deep Statistical Heatmap
pipe.plot_all_associations_heatmap()
```

## Features

- **Memory Optimization:** Automatically downcasts large numbers to save memory (up to 50%+ reduction).
- **Auto-Cleaning:** Automatically converts pure whitespace strings or garbage text (`"N/A"`, `"?"`, `"-"`) to `NaN` values.
- **Smart Imputation:** Dynamically imputes missing numeric values with the median and categorical values with the mode.
- **Statistical EDA:** Calculates an advanced unified correlation matrix supporting numerical, categorical, and mixed variable types (Cramér's V, Eta, Spearman).
- **Feature Pruning:** Automatically identifies and drops highly redundant features (e.g., `> 90%` correlation) and high-cardinality IDs.

## License

This project is licensed under the MIT License.
