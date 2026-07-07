import pandas as pd
import numpy as np
import json
import os
import io
import sys
import logging

# ---------------------------------------------------------------------------
# Library-wide logger: behaves like print() by default, but users can
# control verbosity with logging.getLogger('datawash').setLevel(...)
# ---------------------------------------------------------------------------
logger = logging.getLogger('datawash')
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

__all__ = ['DataPipeline']


class DataPipeline:
    """
    Step 1 — Data Ingestion & Preparation
    ======================================
    Handles smart file loading (Colab / local), garbage-string sanitization,
    auto-type correction (numeric + datetime), and data export.

    All mutation methods return ``self`` so calls can be chained:
        pipe.load_data('data.csv').sanitize_garbage().auto_type_correct()
    """

    def __init__(self):
        self.df = None
        self.df_original = None  # Immutable snapshot of the raw data after loading
        self.garbage_strings = [
            '?', 'n/a', 'N/A', 'NA', 'NULL', 'null',
            'None', 'none', 'NaN', 'nan', ' ', '', '-', '#DIV/0!',
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_data(self):
        """Returns True if a DataFrame is loaded, otherwise warns and returns False."""
        if self.df is None:
            logger.warning("No data loaded. Run load_data() first.")
            return False
        return True

    # ---------------------------------------------------------
    # A. Smart Environment-Aware Data Ingestion
    # ---------------------------------------------------------
    def load_data(self, file_path=None):
        """Detects environment (Colab vs IDE) and routes to the correct upload method."""

        # 1. Check if the script is running inside Google Colab
        in_colab = 'google.colab' in sys.modules

        # --- COLAB ROUTE ---
        if in_colab and not file_path:
            from google.colab import files
            logger.info("Google Colab detected. Please upload your dataset:")
            uploaded = files.upload()

            for filename, file_content in uploaded.items():
                logger.info(f'\nProcessing "{filename}"...')
                self._parse_file(filename, file_content=file_content, is_bytes=True)
                break  # Process only the first uploaded file

        # --- LOCAL IDE / TERMINAL ROUTE ---
        else:
            if not file_path:
                file_path = input("Please enter the full path to your dataset: ").strip()

            # Strip quotes (handles drag-and-drop into terminals)
            file_path = file_path.strip('"\'')

            if not os.path.exists(file_path):
                logger.error(f"Error: The file at '{file_path}' was not found.")
                return self

            logger.info(f'\nProcessing "{file_path}"...')
            self._parse_file(file_path, is_bytes=False)

        # Preserve an immutable snapshot of the raw data
        if self.df is not None:
            self.df_original = self.df.copy()

        return self

    def _parse_file(self, source, file_content=None, is_bytes=False):
        """Helper method to parse the file regardless of how it was uploaded."""
        ext = os.path.splitext(source)[1].lower()

        # Colab requires io.BytesIO, local IDEs just need the file path
        target = io.BytesIO(file_content) if is_bytes else source

        try:
            if ext == '.csv':
                self.df = pd.read_csv(target)

            elif ext in ['.xls', '.xlsx']:
                self.df = pd.read_excel(target)

            elif ext == '.json':
                if is_bytes:
                    parsed_json = json.loads(file_content.decode('utf-8'))
                else:
                    with open(source, 'r', encoding='utf-8') as f:
                        parsed_json = json.load(f)
                self.df = pd.json_normalize(parsed_json)
                logger.info("JSON successfully flattened using pd.json_normalize().")

            elif ext == '.parquet':
                self.df = pd.read_parquet(target)

            elif ext in ['.html', '.htm']:
                # read_html expects a string, not BytesIO — decode if needed
                if is_bytes:
                    html_target = file_content.decode('utf-8')
                else:
                    html_target = target
                html_tables = pd.read_html(html_target)
                if len(html_tables) > 0:
                    self.df = html_tables[0]
                    if len(html_tables) > 1:
                        logger.warning(
                            f"Warning: Loaded the first of {len(html_tables)} tables found."
                        )
                else:
                    logger.error("Error: No <table> elements found.")
                    self.df = None
                    return
            else:
                logger.error(f"Unsupported file type: {ext}")
                self.df = None
                return

            logger.info(
                f"Success! Loaded dataset with {self.df.shape[0]} rows "
                f"and {self.df.shape[1]} columns."
            )

        except Exception as e:
            logger.error(f"Failed to parse data. Error: {str(e)}")
            self.df = None

    # ---------------------------------------------------------
    # B. Garbage String Handling
    # ---------------------------------------------------------
    def sanitize_garbage(self):
        """Converts known garbage strings and hidden whitespace into actual NaNs."""
        if not self._check_data():
            return self

        # 1. Deduplicate column names (e.g., 'A', 'A' -> 'A', 'A_1')
        if not self.df.columns.is_unique:
            cols = pd.Series(self.df.columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
            self.df.columns = cols
            logger.warning("Duplicate column names detected and automatically renamed.")

        # 2. Universal Infinity to NaN
        num_cols_for_inf = self.df.select_dtypes(include=['number']).columns
        if len(num_cols_for_inf) > 0:
            self.df[num_cols_for_inf] = self.df[num_cols_for_inf].replace([np.inf, -np.inf], np.nan)

        # 3. Strip whitespace on string columns FIRST, then replace garbage
        object_columns = self.df.select_dtypes(include=['object', 'string']).columns
        if len(object_columns) > 0:
            for col in object_columns:
                self.df[col] = self.df[col].str.strip()

            self.df[object_columns] = self.df[object_columns].replace(self.garbage_strings, np.nan)

            # Catch any remaining empty strings after stripping
            for col in object_columns:
                self.df[col] = self.df[col].replace({'': np.nan})

        logger.info("Sanitization complete: Garbage strings and pure whitespace converted to NaNs.")
        return self

    def standardize_text(self):
        """
        Converts all string columns to lowercase and strips whitespace.
        Useful for standardizing categorical data before encoding.
        """
        if not self._check_data():
            return self

        object_columns = self.df.select_dtypes(include=['object', 'string']).columns
        if len(object_columns) > 0:
            for col in object_columns:
                self.df[col] = self.df[col].astype(str).str.lower().str.strip()
            logger.info("Standardized text columns (lowercase + stripped).")
        return self

    # ---------------------------------------------------------
    # C. Auto-Type Correction (Numeric + Datetime)
    # ---------------------------------------------------------
    def auto_type_correct(self):
        """
        Safely coerces string columns:
          1. To numeric — if >50 % of non-null values parse as numbers.
          2. To datetime — if >50 % of non-null values parse as dates
             (only attempted on columns that were NOT converted to numeric).
        """
        if not self._check_data():
            return self

        corrected_numeric = []
        corrected_datetime = []
        object_columns = self.df.select_dtypes(include=['object']).columns.tolist()

        for col in object_columns:
            non_null_count = self.df[col].notna().sum()
            if non_null_count == 0:
                continue

            # --- Attempt numeric conversion first ---
            clean_col = self.df[col]
            if hasattr(clean_col, 'str'):
                clean_col = clean_col.str.replace(',', '', regex=False)
            
            converted_num = pd.to_numeric(clean_col, errors='coerce')
            numeric_ratio = converted_num.notna().sum() / non_null_count
            if numeric_ratio > 0.9:
                if numeric_ratio < 1.0:
                    logger.warning(f"Coercing '{col}' to numeric: {100*(1-numeric_ratio):.1f}% of values forced to NaN.")
                self.df[col] = converted_num
                corrected_numeric.append(col)
                continue  # Don't try datetime if already numeric

            # --- Attempt datetime conversion ---
            try:
                # format='mixed' requires pandas >= 2.0; fall back gracefully
                try:
                    converted_dt = pd.to_datetime(self.df[col], errors='coerce', format='mixed')
                except (TypeError, ValueError):
                    converted_dt = pd.to_datetime(self.df[col], errors='coerce', infer_datetime_format=True)
                datetime_ratio = converted_dt.notna().sum() / non_null_count
                if datetime_ratio > 0.9:
                    if datetime_ratio < 1.0:
                        logger.warning(f"Coercing '{col}' to datetime: {100*(1-datetime_ratio):.1f}% of values forced to NaN.")
                    self.df[col] = converted_dt
                    corrected_datetime.append(col)
            except Exception:
                pass  # Some exotic values can cause issues; skip silently

        if corrected_numeric:
            logger.info(f"Auto-corrected to numeric: {', '.join(corrected_numeric)}")
        if corrected_datetime:
            logger.info(f"Auto-corrected to datetime: {', '.join(corrected_datetime)}")
        if not corrected_numeric and not corrected_datetime:
            logger.info("No columns required type conversion.")

        return self

    # ---------------------------------------------------------
    # D. Data Export
    # ---------------------------------------------------------
    def save_data(self, path, fmt=None):
        """
        Exports the current DataFrame.

        Parameters
        ----------
        path : str
            Destination file path.
        fmt : str, optional
            Export format: 'csv', 'excel'/'xlsx', 'json', 'parquet'.
            If *None* (default), the format is inferred from the file extension.
        """
        if not self._check_data():
            return self

        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(os.path.abspath(path))
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            logger.info(f"Created missing directory: {parent_dir}")

        # Infer format from file extension when not explicitly provided
        if fmt is None:
            ext = os.path.splitext(path)[1].lower()
            ext_map = {
                '.csv': 'csv',
                '.xlsx': 'excel', '.xls': 'excel',
                '.json': 'json',
                '.parquet': 'parquet',
            }
            fmt = ext_map.get(ext)
            if fmt is None:
                # Default to csv if extension is missing/unknown
                logger.warning(f"No valid extension found for '{path}', defaulting to CSV.")
                fmt = 'csv'
                if not ext:
                    path += '.csv'
        else:
            fmt = fmt.lower()

        try:
            if fmt == 'csv':
                self.df.to_csv(path, index=False)
            elif fmt in ('excel', 'xlsx'):
                self.df.to_excel(path, index=False)
            elif fmt == 'json':
                self.df.to_json(path, orient='records', indent=2)
            elif fmt == 'parquet':
                self.df.to_parquet(path, index=False)
            else:
                logger.error(f"Unsupported export format: '{fmt}'. Use csv/excel/json/parquet.")
                return self
            logger.info(f"Data saved to '{path}' ({fmt}).")
        except Exception as e:
            logger.error(f"Failed to save data. Error: {str(e)}")

        return self

    # ---------------------------------------------------------
    # E. Reset to Original
    # ---------------------------------------------------------
    def reset_data(self):
        """Restores self.df to the original state captured right after load_data()."""
        if self.df_original is not None:
            self.df = self.df_original.copy()
            logger.info("Dataset reset to original loaded state.")
        else:
            logger.warning("No original data snapshot found. Load data first.")
        return self