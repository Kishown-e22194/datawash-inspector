import pandas as pd
import numpy as np
import logging
from .autoprep import DataPipeline as _BasePipeline

logger = logging.getLogger('datawash')

__all__ = ['DataPipeline']


class DataPipeline(_BasePipeline):
    """
    Step 2 — Structural Analysis & Cleaning
    =========================================
    Inherits all Step 1 methods and adds: data summary, health report,
    missing-value imputation, duplicate removal, outlier handling,
    and interactive row/column deletion.
    """

    # =========================================================
    # STEP 2: STRUCTURAL ANALYSIS & CLEANING
    # =========================================================

    def data_summary(self):
        """Displays a comprehensive overview of the dataset and returns self."""
        if not self._check_data():
            return self

        logger.info("-" * 50)
        logger.info("DATASET SUMMARY")
        logger.info("-" * 50)
        logger.info(f"Total Rows:    {self.df.shape[0]}")
        logger.info(f"Total Columns: {self.df.shape[1]}")

        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        dt_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()

        logger.info(f"\nNumerical Columns  ({len(num_cols)}): {', '.join(num_cols) if num_cols else 'None'}")
        logger.info(f"Categorical Columns ({len(cat_cols)}): {', '.join(cat_cols) if cat_cols else 'None'}")
        logger.info(f"Datetime Columns    ({len(dt_cols)}): {', '.join(dt_cols) if dt_cols else 'None'}")

        logger.info(f"\nPreview (first 10 rows):")
        logger.info(self.df.head(10).to_string())
        return self

    def data_health_report(self):
        """
        Comprehensive data-quality audit:
          - Missing values per column (count + percentage)
          - Constant columns (single unique value — zero information)
          - High-cardinality categoricals (>90 % unique — likely IDs)
        """
        if not self._check_data():
            return self

        total_rows = len(self.df)
        logger.info("=" * 60)
        logger.info("DATA HEALTH REPORT")
        logger.info("=" * 60)

        # --- Missing values ---
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            logger.info("\n✅ No missing values found.")
        else:
            logger.info(f"\n⚠️  Missing Values ({len(missing)} columns affected):")
            for col, count in missing.items():
                pct = count / total_rows * 100
                flag = "  🔴 CONSIDER DROPPING" if pct > 70 else ""
                logger.info(f"   {col:30s} → {count:>6} ({pct:5.1f}%){flag}")

        # --- Constant columns ---
        constant_cols = [col for col in self.df.columns if self.df[col].nunique(dropna=False) <= 1]
        if constant_cols:
            logger.info("\n🟡 Constant Columns (zero information — safe to drop):")
            for col in constant_cols:
                logger.info(f"   {col}")
        else:
            logger.info("\n✅ No constant columns found.")

        # --- High-cardinality categoricals ---
        cat_cols = self.df.select_dtypes(include=['object', 'category', 'string']).columns
        high_card = []
        for col in cat_cols:
            non_null = self.df[col].notna().sum()
            if non_null > 0:
                unique_ratio = self.df[col].nunique() / non_null
                if unique_ratio > 0.9:
                    high_card.append((col, self.df[col].nunique()))
        if high_card:
            logger.info("\n🟡 High-Cardinality Categoricals (likely IDs — not useful for ML):")
            for col, nuniq in high_card:
                logger.info(f"   {col:30s} → {nuniq} unique values")
        else:
            logger.info("\n✅ No high-cardinality categorical columns detected.")

        logger.info("=" * 60)
        return self

    def _compute_health_data(self):
        """
        Returns a structured dict of data-quality metrics, suitable for
        rendering in Streamlit, HTML reports, or the console.

        Returns
        -------
        dict with keys: 'missing', 'constant_cols', 'high_cardinality'
        """
        if self.df is None or self.df.empty:
            return {'missing': {}, 'constant_cols': [], 'high_cardinality': []}

        total_rows = len(self.df)

        # Missing values: {col: {'count': int, 'pct': float}}
        missing_raw = self.df.isnull().sum()
        missing_raw = missing_raw[missing_raw > 0].sort_values(ascending=False)
        missing = {
            col: {'count': int(cnt), 'pct': round(cnt / total_rows * 100, 1)}
            for col, cnt in missing_raw.items()
        }

        # Constant columns
        constant_cols = [
            col for col in self.df.columns
            if self.df[col].nunique(dropna=False) <= 1
        ]

        # High-cardinality categoricals
        cat_cols = self.df.select_dtypes(include=['object', 'category', 'string']).columns
        high_cardinality = []
        for col in cat_cols:
            non_null = self.df[col].notna().sum()
            if non_null > 0 and self.df[col].nunique() / non_null > 0.9:
                high_cardinality.append({'col': col, 'nunique': int(self.df[col].nunique())})

        return {
            'missing': missing,
            'constant_cols': constant_cols,
            'high_cardinality': high_cardinality,
        }

    def handle_missing_values(self, columns=None, strategy='auto', fill_value=None):
        """
        Imputes missing values.

        Strategies:
          - 'auto'    : mean for numeric, mode for categorical (recommended)
          - 'mean'    : numeric columns only
          - 'median'  : numeric columns only
          - 'mode'    : works on any dtype
          - 'constant': fills with ``fill_value``
        """
        if not self._check_data():
            return self

        if columns is None:
            columns = self.df.columns.tolist()

        successful_cols = []
        skipped_cols = []

        for col in columns:
            if col not in self.df.columns:
                continue

            if self.df[col].isnull().sum() == 0:
                continue

            applied_strategy = strategy

            # --- AUTO strategy: pick per-column ---
            if strategy == 'auto':
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    applied_strategy = 'mean'
                else:
                    applied_strategy = 'mode'

            # --- Apply the chosen strategy ---
            if applied_strategy == 'mean':
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                    successful_cols.append(col)
                else:
                    skipped_cols.append(col)

            elif applied_strategy == 'median':
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                    successful_cols.append(col)
                else:
                    skipped_cols.append(col)

            elif applied_strategy == 'mode':
                mode_val = self.df[col].mode()
                if not mode_val.empty:
                    self.df[col] = self.df[col].fillna(mode_val[0])
                    successful_cols.append(col)
                else:
                    skipped_cols.append(col)

            elif applied_strategy == 'constant':
                if fill_value is not None:
                    self.df[col] = self.df[col].fillna(fill_value)
                    successful_cols.append(col)
                else:
                    logger.error("Error: Must provide a 'fill_value' when strategy is 'constant'.")
                    return self

            elif applied_strategy == 'drop':
                initial_len = len(self.df)
                self.df = self.df.dropna(subset=[col]).reset_index(drop=True)
                if len(self.df) < initial_len:
                    logger.info(f"Dropped {initial_len - len(self.df)} rows missing '{col}'.")
                successful_cols.append(col)

        logger.info(f"\nImputation Strategy: '{strategy}'")
        if successful_cols:
            logger.info(f"  Successfully imputed: {', '.join(successful_cols)}")
        if skipped_cols:
            logger.info(f"  Skipped (dtype mismatch): {', '.join(skipped_cols)}")

        return self

    def standardize_text(self, columns=None, lower=True, strip=True, remove_special=False):
        """
        Normalizes text in categorical (object/string) columns to prevent messy
        duplicates (e.g., "New York", "new york ", and "NEW_YORK" become "new york").

        Parameters
        ----------
        columns : list[str] or str, optional
            Specific columns to standardize. If None, applies to all object/string columns.
        lower : bool, default True
            Convert all text to lowercase.
        strip : bool, default True
            Remove leading and trailing whitespace.
        remove_special : bool, default False
            Replace non-alphanumeric characters (like _ or -) with spaces.
        """
        if not self._check_data():
            return self

        if columns is None:
            # Auto-detect text columns
            cols_to_clean = self.df.select_dtypes(include=['object', 'string']).columns.tolist()
        else:
            if isinstance(columns, str):
                columns = [columns]
            cols_to_clean = [c for c in columns if c in self.df.columns]

        if not cols_to_clean:
            logger.info("No text columns found to standardize.")
            return self

        logger.info(f"Standardizing text for {len(cols_to_clean)} columns...")
        for col in cols_to_clean:
            # .str accessor on object columns naturally preserves NaN —
            # do NOT use .astype(str) which converts NaN → literal "nan"
            if lower:
                self.df[col] = self.df[col].str.lower()
            if remove_special:
                self.df[col] = self.df[col].str.replace(r'[^\w\s]', ' ', regex=True)
                self.df[col] = self.df[col].str.replace('_', ' ', regex=False)
                self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
            if strip:
                self.df[col] = self.df[col].str.strip()

            # Collapse empty strings to NaN (e.g. after stripping whitespace-only values)
            self.df[col] = self.df[col].replace({'': np.nan})

        logger.info(f"Text standardization complete for: {', '.join(cols_to_clean)}")
        return self

    def remove_duplicates(self):
        """Removes exact duplicate rows from the dataset."""
        if not self._check_data():
            return self

        initial_rows = self.df.shape[0]
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - self.df.shape[0]

        logger.info(f"Removed {removed} duplicate rows.")
        return self

    def handle_outliers(self, columns=None, action='flag'):
        """
        Detects outliers using the IQR method.

        Actions
        -------
        'flag'  : Creates boolean ``<col>_is_outlier`` columns.
                  These flag columns are stored in ``self._outlier_flags``
                  and **excluded** from ``select_dtypes(include=['number'])``
                  so that downstream ML prep doesn't try to scale them.
        'drop'  : Removes outlier rows entirely.
        """
        if not self._check_data():
            return self

        if columns is None:
            columns = self.df.select_dtypes(include=['number']).columns.tolist()
        elif isinstance(columns, str):
            columns = [columns]

        # Unified mask for 'drop' action
        global_outlier_mask = pd.Series(False, index=self.df.index)

        # Track flag columns so downstream methods can exclude them
        if not hasattr(self, '_outlier_flags'):
            self._outlier_flags = []

        for col in columns:
            if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
                logger.warning(f"Skipping '{col}': Not found or not numeric.")
                continue

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            col_outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)

            if action == 'flag':
                flag_name = f"{col}_is_outlier"
                # Store as bool so it behaves well with pandas indexing but is excluded from numerics
                self.df[flag_name] = col_outliers.astype(bool)
                self._outlier_flags.append(flag_name)
                logger.info(f"Flagged {col_outliers.sum()} outliers in '{col}'. Created '{flag_name}' (bool).")

            elif action == 'drop':
                global_outlier_mask = global_outlier_mask | col_outliers

        if action == 'drop':
            total = global_outlier_mask.sum()
            self.df = self.df[~global_outlier_mask].reset_index(drop=True)
            logger.info(f"Dropped {total} outlier rows across specified columns.")

        return self

    def optimize_memory(self):
        """
        Reduces memory usage by downcasting numeric columns.
        - float64 -> float32
        - int64 -> int32, int16, or int8 depending on range
        """
        if not self._check_data():
            return self

        start_mem = self.df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage before optimization: {start_mem:.2f} MB")

        for col in self.df.columns:
            col_type = self.df[col].dtype

            if col_type != object:
                c_min = self.df[col].min()
                c_max = self.df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.df[col] = self.df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.df[col] = self.df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.df[col] = self.df[col].astype(np.int32)
                elif str(col_type)[:5] == 'float':
                    # Only attempt downcast if values fit within standard Int64 bounds
                    if c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                        non_na = self.df[col].dropna()
                        try:
                            if len(non_na) > 0 and (non_na == non_na.astype('int64')).all():
                                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                                    self.df[col] = self.df[col].astype('Int8')
                                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                                    self.df[col] = self.df[col].astype('Int16')
                                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                                    self.df[col] = self.df[col].astype('Int32')
                                else:
                                    self.df[col] = self.df[col].astype('Int64')
                                continue
                        except Exception:
                            pass
                    
                    # Prevent Overflow to Infinity by checking float32 bounds
                    if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                        self.df[col] = self.df[col].astype(np.float32)

        end_mem = self.df.memory_usage().sum() / 1024**2
        if start_mem > 0:
            reduction = 100 * (start_mem - end_mem) / start_mem
            logger.info(f"Memory usage after optimization: {end_mem:.2f} MB ({reduction:.1f}% reduction)")

        return self

    def delete_columns(self, columns=None):
        """
        Deletes columns from the dataset.

        Parameters
        ----------
        columns : list[str] or str, optional
            Column name(s) to drop.  If *None*, falls back to an
            interactive prompt (suitable for notebooks / terminals).
        """
        if not self._check_data():
            return self

        if columns is None:
            logger.info(f"Current columns: {self.df.columns.tolist()}")
            cols_input = input("Enter columns to delete (comma-separated), or press Enter to skip: ")
            if not cols_input.strip():
                return self
            columns = [col.strip() for col in cols_input.split(',')]
        elif isinstance(columns, str):
            columns = [columns]

        valid_cols = [col for col in columns if col in self.df.columns]
        invalid_cols = [col for col in columns if col not in self.df.columns]

        if valid_cols:
            self.df = self.df.drop(columns=valid_cols)
            logger.info(f"Successfully deleted columns: {valid_cols}")
        if invalid_cols:
            logger.warning(f"Warning: Columns not found and skipped: {invalid_cols}")

        return self

    def delete_rows(self, indices=None):
        """
        Deletes rows from the dataset by index.

        Parameters
        ----------
        indices : list[int] or int, optional
            Row index(es) to drop.  If *None*, falls back to an
            interactive prompt (suitable for notebooks / terminals).
        """
        if not self._check_data():
            return self

        if indices is None:
            logger.info(f"Current index range: {self.df.index.min()} to {self.df.index.max()}")
            rows_input = input("Enter row indices to delete (comma-separated), or press Enter to skip: ")
            if not rows_input.strip():
                return self
            try:
                indices = [int(idx.strip()) for idx in rows_input.split(',')]
            except ValueError:
                logger.error("Error: Please enter valid numeric indices separated by commas.")
                return self
        elif isinstance(indices, int):
            indices = [indices]

        valid_rows = [idx for idx in indices if idx in self.df.index]
        invalid_rows = [idx for idx in indices if idx not in self.df.index]

        if valid_rows:
            self.df = self.df.drop(index=valid_rows).reset_index(drop=True)
            logger.info(f"Successfully deleted {len(valid_rows)} rows. Index reset.")
        if invalid_rows:
            logger.warning(f"Warning: Row indices not found: {invalid_rows}")

        return self