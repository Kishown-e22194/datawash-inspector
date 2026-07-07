import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import warnings

from .cleaner import DataPipeline as _BasePipeline

logger = logging.getLogger('datawash')

__all__ = ['DataPipeline']


class DataPipeline(_BasePipeline):
    """
    Step 3 — Feature Engineering  &  Step 4 — Visualization
    =========================================================
    Inherits Steps 1 + 2 and adds: numeric scaling, categorical encoding,
    a unified ``prepare_for_ml()`` pipeline, correlation heatmap,
    missing-value visualisation, univariate analysis, smart relationship
    plots, and categorical frequency charts.

    Scalers and encoders are stored so they can be reused on new data.
    """

    def __init__(self):
        super().__init__()
        self.df_processed = None   # Final ML-ready dataset
        self.scaler = None         # Fitted scaler (for inverse_transform / test data)
        self.encoder = None        # Fitted encoder (for inverse_transform / test data)

    # =========================================================
    # STEP 3: FEATURE ENGINEERING (NORMALIZATION)
    # =========================================================

    def extract_normalized_numeric_data(self, columns=None, strategy='standard'):
        """
        Extracts and scales numeric columns.  Safely drops NaN rows
        from the extracted subset before fitting.

        Strategies: 'minmax', 'standard', 'robust', 'log' (log1p).
        Returns the scaled DataFrame (does NOT mutate self.df).

        .. deprecated::
            This method fits on whatever data you pass in, risking data
            leakage.  Use :meth:`fit_transform_ml` and :meth:`transform_ml`
            for proper train/test workflows.
        """
        warnings.warn(
            "extract_normalized_numeric_data() fits on the provided data and "
            "risks data leakage.  Use fit_transform_ml(train) + "
            "transform_ml(test) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self._check_data():
            return None

        # Isolate targeted numeric data
        if columns:
            num_df = self.df[columns].select_dtypes(include=['number']).copy()
        else:
            num_df = self.df.select_dtypes(include=['number']).copy()

        if num_df.empty:
            logger.info("No numeric columns available to scale.")
            return num_df

        # Safety: sklearn scalers crash on NaN
        initial_len = len(num_df)
        num_df = num_df.dropna()
        if len(num_df) < initial_len:
            logger.warning(
                f"Warning: Dropped {initial_len - len(num_df)} rows with NaN before scaling."
            )

        # --- Log transform (special case — no sklearn scaler) ---
        if strategy == 'log':
            num_df = num_df.copy()
            # Shift negative values so log1p works on the whole column
            for col in num_df.columns:
                col_min = num_df[col].min()
                if col_min <= 0:
                    num_df[col] = num_df[col] - col_min + 1
            scaled_df = np.log1p(num_df)
            self.scaler = None
            logger.info(f"Extracted and log-transformed {scaled_df.shape[1]} numeric columns.")
            return scaled_df

        # --- Standard sklearn scalers ---
        if strategy == 'standard':
            self.scaler = StandardScaler()
        elif strategy == 'minmax':
            self.scaler = MinMaxScaler()
        elif strategy == 'robust':
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Unknown strategy '{strategy}'. Defaulting to 'standard'.")
            self.scaler = StandardScaler()

        scaled_values = self.scaler.fit_transform(num_df)
        scaled_df = pd.DataFrame(scaled_values, columns=num_df.columns, index=num_df.index)

        logger.info(f"Extracted and scaled {scaled_df.shape[1]} numeric columns using '{strategy}'.")
        return scaled_df

    def extract_normalized_categorical_data(self, columns=None, strategy='onehot', drop_first=False):
        """
        Extracts and encodes categorical columns.  Safely drops NaN rows
        from the extracted subset before fitting.

        Strategies: 'onehot', 'ordinal', 'uniform' (ordinal scaled 0-1).
        Returns the encoded DataFrame (does NOT mutate self.df).

        .. deprecated::
            This method fits on whatever data you pass in, risking data
            leakage.  Use :meth:`fit_transform_ml` and :meth:`transform_ml`
            for proper train/test workflows.
        """
        warnings.warn(
            "extract_normalized_categorical_data() fits on the provided data "
            "and risks data leakage.  Use fit_transform_ml(train) + "
            "transform_ml(test) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self._check_data():
            return None

        if columns:
            cat_df = self.df[columns].select_dtypes(exclude=['number']).copy()
        else:
            cat_df = self.df.select_dtypes(exclude=['number']).copy()

        if cat_df.empty:
            logger.info("No categorical columns available to encode.")
            return cat_df

        # Safety: OrdinalEncoder crashes on NaN
        initial_len = len(cat_df)
        cat_df = cat_df.dropna()
        if len(cat_df) < initial_len:
            logger.warning(
                f"Warning: Dropped {initial_len - len(cat_df)} rows with NaN before encoding."
            )

        if strategy == 'onehot':
            encoded_df = pd.get_dummies(cat_df, drop_first=drop_first, dtype=int)
            self.encoder = None  # get_dummies doesn't produce a reusable encoder

        elif strategy == 'ordinal':
            self.encoder = OrdinalEncoder()
            encoded_values = self.encoder.fit_transform(cat_df)
            encoded_df = pd.DataFrame(encoded_values, columns=cat_df.columns, index=cat_df.index)

        elif strategy == 'uniform':
            self.encoder = OrdinalEncoder()
            encoded_values = self.encoder.fit_transform(cat_df)
            uniform_scaler = MinMaxScaler()
            uniform_values = uniform_scaler.fit_transform(encoded_values)
            encoded_df = pd.DataFrame(uniform_values, columns=cat_df.columns, index=cat_df.index)

        else:
            logger.warning(f"Unknown strategy '{strategy}'. Returning raw categoricals.")
            return cat_df

        logger.info(
            f"Encoded {cat_df.shape[1]} categorical columns into "
            f"{encoded_df.shape[1]} features using '{strategy}'."
        )
        return encoded_df

    def merge_features(self, num_df, cat_df):
        """Safely merges numeric and categorical DataFrames after verifying index alignment."""
        if not num_df.index.equals(cat_df.index):
            raise ValueError(
                "Index mismatch between numeric and categorical datasets. "
                "Use prepare_for_ml() for a safe, unified pipeline."
            )

        self.df_processed = pd.concat([num_df, cat_df], axis=1)
        logger.info(f"Merged. ML-ready dataset shape: {self.df_processed.shape}")
        return self.df_processed

    # ---------------------------------------------------------
    # Unified ML-preparation (solves the NaN-index-mismatch bug)
    # ---------------------------------------------------------

    # -- Internal helpers for numeric / categorical transforms --

    @staticmethod
    def _build_scaler(strategy):
        """Return a fresh sklearn scaler for the given strategy name."""
        if strategy == 'standard':
            return StandardScaler()
        elif strategy == 'minmax':
            return MinMaxScaler()
        elif strategy == 'robust':
            return RobustScaler()
        else:
            logger.warning(f"Unknown num_strategy '{strategy}'. Using 'standard'.")
            return StandardScaler()

    def _scale_numeric(self, num_df, strategy, *, fit):
        """Scale numeric columns.  When *fit* is True the scaler is fitted;
        when False the stored scaler is reused (for test data)."""
        if num_df.empty:
            return num_df

        if strategy == 'log':
            # Log transform has no reusable scaler
            num_df = num_df.copy()
            for col in num_df.columns:
                col_min = num_df[col].min()
                if col_min <= 0:
                    num_df[col] = num_df[col] - col_min + 1
            if fit:
                self.scaler = None
            return np.log1p(num_df)

        # Clip astronomically large values to avoid overflow during scaler variance calculation
        safe_num_df = num_df.clip(lower=-1e100, upper=1e100)

        if fit:
            self.scaler = self._build_scaler(strategy)
            values = self.scaler.fit_transform(safe_num_df)
        else:
            if self.scaler is None:
                raise RuntimeError(
                    "No fitted scaler found. Call fit_transform_ml() on "
                    "training data before calling transform_ml().")
            values = self.scaler.transform(safe_num_df)

        return pd.DataFrame(values, columns=num_df.columns, index=num_df.index)

    def _encode_categorical(self, cat_df, strategy, drop_first, *, fit):
        """Encode categorical columns.  When *fit* is True the encoder is
        fitted; when False the stored encoder is reused (for test data)."""
        if cat_df.empty:
            return cat_df

        if strategy == 'onehot':
            if fit:
                self.encoder = None
                self._onehot_columns = None
            encoded = pd.get_dummies(cat_df, drop_first=drop_first, dtype=int)
            if fit:
                self._onehot_columns = encoded.columns
            else:
                # Align test columns to training columns
                ref_cols = self._onehot_columns if self._onehot_columns is not None else []
                encoded = encoded.reindex(columns=ref_cols, fill_value=0)
            return encoded

        if strategy in ('ordinal', 'uniform'):
            if fit:
                self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                vals = self.encoder.fit_transform(cat_df)
            else:
                if self.encoder is None:
                    raise RuntimeError(
                        "No fitted encoder found. Call fit_transform_ml() on "
                        "training data before calling transform_ml().")
                vals = self.encoder.transform(cat_df)

            if strategy == 'uniform':
                if fit:
                    self._uniform_scaler = MinMaxScaler()
                    vals = self._uniform_scaler.fit_transform(vals)
                else:
                    vals = self._uniform_scaler.transform(vals)

            return pd.DataFrame(vals, columns=cat_df.columns, index=cat_df.index)

        logger.warning(f"Unknown cat_strategy '{strategy}'. Using 'onehot'.")
        return self._encode_categorical(cat_df, 'onehot', drop_first, fit=fit)

    # -- Public API ---------------------------------------------------

    def fit_transform_ml(self, train_df=None, y=None,
                         num_strategy='standard', cat_strategy='onehot',
                         drop_first=False):
        """
        Fit scalers / encoders on **training data only**, then transform it.

        Parameters
        ----------
        train_df : DataFrame, optional
            Training split.  If *None*, uses ``self.df`` (full dataset).
        y : Series, optional
            Target labels. If provided, returns an aligned (self, y_aligned) tuple.
        num_strategy : str
            'standard', 'minmax', 'robust', or 'log'.
        cat_strategy : str
            'onehot', 'ordinal', or 'uniform'.
        drop_first : bool
            Drop the first dummy column when using one-hot encoding.

        Returns
        -------
        self – with ``self.df_processed`` set to the transformed training data
        and ``self.scaler`` / ``self.encoder`` fitted for later reuse.
        """
        data = train_df if train_df is not None else self.df
        if data is None or data.empty:
            logger.warning("No data to process.")
            return self

        # Store strategies for transform_ml to reuse
        self._ml_num_strategy = num_strategy
        self._ml_cat_strategy = cat_strategy
        self._ml_drop_first = drop_first

        clean = data.dropna()
        dropped = len(data) - len(clean)
        if dropped > 0:
            logger.info(f"fit_transform_ml: Dropped {dropped} rows containing NaN.")

        num_df = self._scale_numeric(
            clean.select_dtypes(include=['number']).copy(), num_strategy, fit=True)
        cat_df = self._encode_categorical(
            clean.select_dtypes(exclude=['number']).copy(), cat_strategy,
            drop_first, fit=True)

        parts = [df for df in (num_df, cat_df) if not df.empty]
        self.df_processed = pd.concat(parts, axis=1) if parts else pd.DataFrame()

        logger.info(f"ML-ready (train) dataset shape: {self.df_processed.shape}")
        
        if y is not None:
            self.y_processed = y.loc[clean.index]
            return self, self.y_processed
            
        return self

    def transform_ml(self, test_df, y=None):
        """
        Transform **test / unseen data** using the scalers and encoders
        that were previously fitted by ``fit_transform_ml()``.

        Parameters
        ----------
        test_df : DataFrame
            The held-out test split (same column schema as training data).
        y : Series, optional
            Target labels. If provided, returns an aligned (X_test, y_test) tuple.

        Returns
        -------
        DataFrame – the transformed test data.
        """
        if not hasattr(self, '_ml_num_strategy'):
            raise RuntimeError(
                "No fitted transformers found. "
                "Call fit_transform_ml(train_data) first.")

        clean = test_df.dropna()
        dropped = len(test_df) - len(clean)
        if dropped > 0:
            logger.info(f"transform_ml: Dropped {dropped} rows containing NaN.")

        num_df = self._scale_numeric(
            clean.select_dtypes(include=['number']).copy(),
            self._ml_num_strategy, fit=False)
        cat_df = self._encode_categorical(
            clean.select_dtypes(exclude=['number']).copy(),
            self._ml_cat_strategy, self._ml_drop_first, fit=False)

        parts = [df for df in (num_df, cat_df) if not df.empty]
        result = pd.concat(parts, axis=1) if parts else pd.DataFrame()

        logger.info(f"ML-ready (test) dataset shape: {result.shape}")
        
        if y is not None:
            return result, y.loc[clean.index]
            
        return result

    def prepare_for_ml(self, y=None, num_strategy='standard', cat_strategy='onehot', drop_first=False):
        """
        Convenience wrapper: fits **and** transforms ``self.df`` in one call.

        .. warning::
           This fits on the full dataset.  For proper ML workflows, split
           your data first and use ``fit_transform_ml(train)`` followed by
           ``transform_ml(test)`` to avoid data leakage.

        Returns self for chaining (or (self, y_aligned) if y is provided).
        """
        warnings.warn(
            "prepare_for_ml() fits on the FULL dataset, which causes data "
            "leakage if used before train/test split. For proper ML workflows, "
            "use fit_transform_ml(train) + transform_ml(test) instead.",
            UserWarning,
            stacklevel=2,
        )
        return self.fit_transform_ml(
            train_df=None,
            y=y,
            num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            drop_first=drop_first
        )

    def inverse_transform_numeric(self, scaled_df):
        """
        Reverses the numeric scaling using the stored scaler.
        Only works if the last scaling was done with a sklearn scaler (not 'log').
        """
        if self.scaler is None:
            logger.warning("No scaler stored (was 'log' used, or no scaling performed?).")
            return scaled_df

        original_values = self.scaler.inverse_transform(scaled_df)
        return pd.DataFrame(original_values, columns=scaled_df.columns, index=scaled_df.index)

    # =========================================================
    # STEP 4: ADVANCED INTERACTIVE VISUALIZATION
    # =========================================================

    def plot_missing_values(self, return_fig=False):
        """Bar chart of missing values per column, coloured by severity."""
        if not self._check_data():
            return self

        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False).reset_index()

        if missing.empty:
            logger.info("No missing values found in the dataset.")
            return self

        missing.columns = ['Column', 'Missing Count']
        fig = px.bar(
            missing, x='Column', y='Missing Count', text='Missing Count',
            title="Missing Values Summary",
            color='Missing Count', color_continuous_scale='Reds',
        )
        fig.update_traces(textposition='outside')
        if return_fig:
            return fig
        fig.show()
        return self

    def plot_correlation_matrix(self, return_fig=False):
        """Interactive heatmap of the Pearson correlation matrix for numeric columns."""
        if not self._check_data():
            return self

        num_df = self.df.select_dtypes(include=['number'])
        if num_df.shape[1] < 2:
            logger.info("Need at least 2 numeric columns for a correlation matrix.")
            return self

        corr = num_df.corr()

        # Dynamic sizing: scale with feature count, enforce a reasonable minimum
        n_features = len(corr.columns)
        dynamic_size = max(600, n_features * 40)
        font_size = max(8, min(12, 300 // n_features))  # shrink text for large matrices

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={'size': font_size},
            hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>',
        ))
        fig.update_layout(
            title="Correlation Matrix",
            width=dynamic_size,
            height=dynamic_size,
        )
        if return_fig:
            return fig
        fig.show()
        return self

    def univariate_subplots(self, col, return_fig=False):
        """3-panel subplot: vertical box, scatter (index vs value), histogram."""
        if not self._check_data():
            return self
        if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
            logger.warning(f"'{col}' not found or not numeric.")
            return self

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Distribution (Box)", "Value Progression", "Histogram"),
        )

        fig.add_trace(
            go.Box(y=self.df[col], name=col, boxpoints='outliers', marker_color='indigo'),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.df.index, y=self.df[col], mode='markers',
                marker=dict(color='teal', opacity=0.6), name="Index",
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Histogram(x=self.df[col], nbinsx=30, marker_color='darkcyan', name="Counts"),
            row=1, col=3,
        )

        fig.update_layout(
            height=500,  # Increased height to prevent vertical squishing
            title_text=f"Univariate Analysis: {col}",
            showlegend=False,
            autosize=True,
            margin=dict(b=80) # Add bottom margin for labels
        )
        if return_fig:
            return fig
        fig.show()
        return self

    def univariate_all(self):
        """Generates univariate subplots for every numeric column."""
        if not self._check_data():
            return self
        for col in self.df.select_dtypes(include=['number']).columns:
            self.univariate_subplots(col)
        return self

    def plot_relationship(self, col1, col2, return_fig=False):
        """
        Smart router: drops NaN for the target pair, detects types, and picks
        the correct chart.
          Num-Num  → Scatter + OLS trendline (graceful fallback if statsmodels missing)
          Cat-Num  → Box plot with all points
          Cat-Cat  → Grouped bar chart
        """
        if not self._check_data():
            return self
        if col1 not in self.df.columns or col2 not in self.df.columns:
            logger.warning("One or both columns not found.")
            return self

        plot_df = self.df[[col1, col2]].dropna()

        is_num1 = pd.api.types.is_numeric_dtype(plot_df[col1])
        is_num2 = pd.api.types.is_numeric_dtype(plot_df[col2])

        if is_num1 and is_num2:
            try:
                fig = px.scatter(
                    plot_df, x=col1, y=col2, trendline='ols', opacity=0.6,
                    title=f"Numeric Correlation: {col1} vs {col2}",
                )
            except ImportError:
                logger.info("'statsmodels' not installed — plotting without OLS trendline.")
                fig = px.scatter(
                    plot_df, x=col1, y=col2, opacity=0.6,
                    title=f"Numeric Correlation: {col1} vs {col2}",
                )

        elif (not is_num1 and is_num2) or (is_num1 and not is_num2):
            cat_col = col1 if not is_num1 else col2
            num_col = col2 if not is_num1 else col1
            fig = px.box(
                plot_df, x=cat_col, y=num_col, points="all", color=cat_col,
                title=f"Distribution Breakdown: {num_col} by {cat_col}",
            )

        else:
            counts = plot_df.groupby([col1, col2]).size().reset_index(name='Count')
            fig = px.bar(
                counts, x=col1, y="Count", color=col2, barmode="group",
                title=f"Categorical Frequency: {col1} vs {col2}",
            )
            
        # Fix font overlap for long categorical names
        fig.update_xaxes(tickangle=45)
        fig.update_layout(margin=dict(b=100))

        if return_fig:
            return fig
        fig.show()
        return self

    def plot_categorical_frequency(self, col, sort_by='count', return_fig=False):
        """Bar chart with raw counts and percentage labels. sort_by: 'count' | 'name'."""
        if not self._check_data():
            return self
        if col not in self.df.columns:
            logger.warning(f"Column '{col}' not found.")
            return self

        counts = self.df[col].value_counts().reset_index()
        counts.columns = [col, 'Count']

        if sort_by == 'name':
            counts = counts.sort_values(by=col)

        total = counts['Count'].sum()
        counts['Percentage'] = (counts['Count'] / total * 100).round(1).astype(str) + '%'

        fig = px.bar(
            counts, x=col, y='Count', text='Percentage', color=col,
            title=f"Frequency Analysis: {col}",
        )
        fig.update_traces(textposition='outside')
        
        # Fix font overlap
        fig.update_xaxes(tickangle=45)
        fig.update_layout(margin=dict(b=100))
        if return_fig:
            return fig
        fig.show()
        return self