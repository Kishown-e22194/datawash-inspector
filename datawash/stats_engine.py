import pandas as pd
import numpy as np
import scipy.stats as ss
import plotly.graph_objects as go
import logging

# Inherit from the ML Pipeline
from .eda_engine import DataPipeline as _MLPipeline

logger = logging.getLogger('datawash')

__all__ = ['InsightsEngine']

class InsightsEngine(_MLPipeline):
    """
    Step 5 — Deep Statistical Insights
    =========================================================
    Inherits Steps 1, 2, 3, and 4. Adds advanced statistical 
    calculations (Cramér's V, Correlation Ratio) for mixed data types.
    """
    
    def __init__(self):
        super().__init__()

    # =========================================================
    # INTERNAL STATISTICAL HELPERS
    # =========================================================

    def _cramers_v(self, x, y):
        """Calculates bias-corrected Cramér's V for two categorical variables."""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        
        # Bias correction
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        
        # Prevent division by zero or negative denominator
        denominator = min((kcorr-1), (rcorr-1))
        if denominator <= 0:
            return 0.0
            
        return np.sqrt(phi2corr / denominator)

    def _correlation_ratio(self, categories, measurements):
        """Calculates the Correlation Ratio (Eta) for Cat vs Num variables."""
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        
        for i in range(0, cat_num):
            cat_indices = np.argwhere(fcat == i)
            n_array[i] = len(cat_indices)
            y_avg_array[i] = np.average(measurements.iloc[cat_indices.flatten()])
            
        y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
        numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
        denominator = np.sum(np.power(np.subtract(measurements.values, y_total_avg), 2))
        
        if denominator == 0:
            return 0.0
            
        return np.sqrt(numerator / denominator)

    # =========================================================
    # STEP 5: THE UNIFIED HEATMAP
    # =========================================================

    def _compute_associations_matrix(self):
        """Internal helper to compute the unified associations matrix."""
        columns = self.df.columns
        n_cols = len(columns)
        
        if n_cols < 2:
            logger.warning("Need at least 2 columns to calculate associations.")
            return None
            
        assoc_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

        for i in range(n_cols):
            col1 = columns[i]
            for j in range(i, n_cols):
                col2 = columns[j]
                
                if i == j:
                    assoc_matrix.loc[col1, col2] = 1.0
                    continue

                pair_df = self.df[[col1, col2]].dropna()
                
                if len(pair_df) < 2:
                    val = 0.0
                else:
                    try:
                        is_num1 = pd.api.types.is_numeric_dtype(pair_df[col1])
                        is_num2 = pd.api.types.is_numeric_dtype(pair_df[col2])

                        if is_num1 and is_num2:
                            corr = pair_df[col1].corr(pair_df[col2], method='spearman')
                            val = abs(corr) if not np.isnan(corr) else 0.0
                        elif not is_num1 and not is_num2:
                            val = self._cramers_v(pair_df[col1], pair_df[col2])
                        else:
                            if is_num1:
                                val = self._correlation_ratio(pair_df[col2], pair_df[col1])
                            else:
                                val = self._correlation_ratio(pair_df[col1], pair_df[col2])
                    except Exception as e:
                        logger.warning(f"Could not compute association for '{col1}' vs '{col2}': {e}")
                        val = 0.0
                
                assoc_matrix.loc[col1, col2] = val
                assoc_matrix.loc[col2, col1] = val
                
        return assoc_matrix

    def plot_all_associations_heatmap(self, return_fig=False):
        """
        Generates a unified heatmap across ALL data types:
        - Num vs Num: Absolute Spearman's rho (catches non-linear, maps to [0,1])
        - Cat vs Cat: Cramér's V
        - Mixed (Num vs Cat): Correlation Ratio (Eta)
        """
        if not self._check_data():
            return self

        logger.info("Calculating pairwise statistical associations... This may take a moment.")
        assoc_matrix = self._compute_associations_matrix()
        
        if assoc_matrix is None:
            return self

        n_cols = len(assoc_matrix.columns)
        
        # Dynamic font sizing for large matrices
        text_size = 12 if n_cols < 15 else 9 if n_cols < 30 else 6
        show_text = n_cols <= 45 # Hide text entirely if it's too massive

        # Render the Heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=assoc_matrix.values,
            x=assoc_matrix.columns.tolist(),
            y=assoc_matrix.columns.tolist(),
            colorscale='YlOrRd', # Sequential color scale mapping from 0 to 1
            zmin=0, zmax=1, 
            text=np.round(assoc_matrix.values, 2) if show_text else None,
            texttemplate='%{text}' if show_text else '',
            hovertemplate='X: %{x}<br>Y: %{y}<br>Association: %{z:.2f}<extra></extra>',
            textfont=dict(size=text_size)
        ))
        
        # Dynamically scale height/width based on feature count
        matrix_size = n_cols * 45
        plot_size = max(600, matrix_size)

        fig.update_layout(
            title="Unified Associations Matrix (Absolute Spearman, Cramér's V, Eta)",
            width=plot_size, 
            height=plot_size,
            yaxis_autorange='reversed',
            template='plotly_white',
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black")
        )
        
        if return_fig:
            return fig
            
        fig.show()
        return self

    def generate_html_report(self, filename="datainspector_report.html", auto_open=True, include_download_btn=True):
        """
        Automatically generates a comprehensive HTML report containing basic distributions 
        for all columns and the full associations heatmap. Saves it to disk and optionally opens it.
        """
        import os
        import webbrowser
        from .plot_factory import PlottingMethods, compose_report
        
        if not self._check_data():
            return self
            
        logger.info(f"Generating full HTML report to {filename}...")
        
        # --- Data Summary & Health Report HTML ---
        health = self._compute_health_data()
        mem_mb = self.df.memory_usage().sum() / 1024**2
        
        summary_html = f'''
        <div style="display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap;">
            <div class="chart-card" style="flex: 1; min-width: 200px; text-align: center; margin-bottom: 0;">
                <h3 style="color: #a0aec0; margin-bottom: 10px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Total Rows</h3>
                <div style="font-size: 42px; font-weight: bold; color: var(--accent);">{self.df.shape[0]:,}</div>
            </div>
            <div class="chart-card" style="flex: 1; min-width: 200px; text-align: center; margin-bottom: 0;">
                <h3 style="color: #a0aec0; margin-bottom: 10px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Total Columns</h3>
                <div style="font-size: 42px; font-weight: bold; color: #48bb78;">{self.df.shape[1]:,}</div>
            </div>
            <div class="chart-card" style="flex: 1; min-width: 200px; text-align: center; margin-bottom: 0;">
                <h3 style="color: #a0aec0; margin-bottom: 10px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Memory Usage</h3>
                <div style="font-size: 42px; font-weight: bold; color: #ed8936;">{mem_mb:.2f} <span style="font-size: 20px;">MB</span></div>
            </div>
        </div>
        '''
        
        health_html = "<div class='chart-card' style='margin-bottom: 30px;'><h2 style='margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px;'>Data Health Audit</h2>"
        
        # Missing values
        if health['missing']:
            health_html += f"<h3 style='color: #fc8181; margin-top: 25px; margin-bottom: 15px;'>⚠️ Missing Values <span style='font-size: 14px; font-weight: normal; color: #a0aec0;'>({len(health['missing'])} columns)</span></h3>"
            health_html += "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px;'>"
            for col, v in health['missing'].items():
                health_html += f"<div style='background: rgba(252,129,129,0.1); border-left: 4px solid #fc8181; padding: 12px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'><strong>{col}</strong><br><span style='font-size: 14px;'>{v['count']:,} missing ({v['pct']}%)</span></div>"
            health_html += "</div>"
        else:
            health_html += "<div style='background: rgba(72,187,120,0.1); border-left: 4px solid #48bb78; padding: 15px; border-radius: 6px; margin-top: 20px;'><span style='color: #48bb78; font-size: 18px;'>✅</span> <strong>Perfect:</strong> No missing values found.</div>"
            
        # Constant Columns
        if health['constant_cols']:
            health_html += f"<h3 style='color: #f6ad55; margin-top: 25px; margin-bottom: 15px;'>🟡 Constant Columns <span style='font-size: 14px; font-weight: normal; color: #a0aec0;'>({len(health['constant_cols'])})</span></h3>"
            health_html += "<div style='display: flex; gap: 10px; flex-wrap: wrap;'>"
            for col in health['constant_cols']:
                health_html += f"<span style='background: rgba(246,173,85,0.1); color: #f6ad55; padding: 6px 14px; border-radius: 20px; border: 1px solid rgba(246,173,85,0.3); font-size: 14px;'>{col}</span>"
            health_html += "</div>"
        else:
            health_html += "<div style='background: rgba(72,187,120,0.1); border-left: 4px solid #48bb78; padding: 15px; border-radius: 6px; margin-top: 20px;'><span style='color: #48bb78; font-size: 18px;'>✅</span> <strong>Perfect:</strong> No constant columns found.</div>"
            
        # High Cardinality
        if health['high_cardinality']:
            health_html += f"<h3 style='color: #63b3ed; margin-top: 25px; margin-bottom: 15px;'>🔵 High-Cardinality Categoricals <span style='font-size: 14px; font-weight: normal; color: #a0aec0;'>({len(health['high_cardinality'])})</span></h3>"
            health_html += "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px;'>"
            for h in health['high_cardinality']:
                health_html += f"<div style='background: rgba(99,179,237,0.1); border-left: 4px solid #63b3ed; padding: 12px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'><strong>{h['col']}</strong><br><span style='font-size: 14px;'>{h['nunique']:,} unique values</span></div>"
            health_html += "</div>"
        else:
            health_html += "<div style='background: rgba(72,187,120,0.1); border-left: 4px solid #48bb78; padding: 15px; border-radius: 6px; margin-top: 20px;'><span style='color: #48bb78; font-size: 18px;'>✅</span> <strong>Perfect:</strong> No high-cardinality categorical columns detected.</div>"
            
        health_html += "</div>"
        
        if include_download_btn:
            import base64
            csv_data = self.df.to_csv(index=False)
            b64_csv = base64.b64encode(csv_data.encode('utf-8')).decode('utf-8')
            download_html = f'''
            <div style="margin-bottom: 30px; text-align: center;">
                <a href="data:text/csv;base64,{b64_csv}" download="clean_datawash_file.csv" 
                   style="background-color: #4CAF50; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; border-radius: 8px; font-weight: bold;">
                   📥 Download Cleaned file
                </a>
            </div>
            '''
        else:
            download_html = ""
        
        tabs = {}
        tabs["Overview & Health"] = [summary_html, health_html, download_html]
        
        # To prevent massive 50MB+ HTML files that crash Colab, we sample data for charts
        original_df = self.df
        if len(self.df) > 1500:
            logger.info("Sampling data to 1500 rows for lightning-fast frontend visual generation...")
            self.df = self.df.sample(1500, random_state=42)
            
        first = True
        
        # Univariate Analysis (Numeric)
        num_cols = self.df.select_dtypes(include=['number']).columns
        num_charts = []
        for col in num_cols:
            fig = self.univariate_subplots(col, return_fig=True)
            if fig and fig != self:
                html = fig.to_html(full_html=False, include_plotlyjs='cdn' if first else False)
                num_charts.append(f"<div class='chart-card' style='margin-bottom: 30px;'>{html}</div>")
                first = False
        if num_charts:
            tabs["Univariate Analysis"] = num_charts
                
        # Categorical Analysis
        cat_cols = self.df.select_dtypes(exclude=['number']).columns
        cat_charts = []
        for col in cat_cols:
            fig = self.plot_categorical_frequency(col, return_fig=True)
            if fig and fig != self:
                html = fig.to_html(full_html=False, include_plotlyjs='cdn' if first else False)
                cat_charts.append(f"<div class='chart-card' style='margin-bottom: 30px;'>{html}</div>")
                first = False
        if cat_charts:
            tabs["Categorical Analysis"] = cat_charts
                
        # Top Relationships (Smart Routing)
        rel_charts = []
        assoc_matrix = self._compute_associations_matrix()
        if assoc_matrix is not None:
            upper_tri = assoc_matrix.where(np.triu(np.ones(assoc_matrix.shape), k=1).astype(bool))
            unstacked = upper_tri.unstack().dropna()
            top_pairs = unstacked.sort_values(ascending=False).head(5)
            
            for (col1, col2), score in top_pairs.items():
                fig = self.plot_relationship(col1, col2, return_fig=True)
                if fig and fig != self:
                    html = fig.to_html(full_html=False, include_plotlyjs='cdn' if first else False)
                    subtitle = f"<h4 style='text-align:center; color: #a0aec0; text-transform: uppercase; letter-spacing: 1px;'>Association Score: <span style='color: var(--accent);'>{score:.2f}</span></h4>"
                    rel_charts.append(f"<div class='chart-card' style='margin-bottom: 30px;'>{subtitle}{html}</div>")
                    first = False
        if rel_charts:
            tabs["Top Relationships"] = rel_charts
                
        # Heatmap
        assoc_charts = []
        if len(self.df.columns) >= 2:
            fig = self.plot_all_associations_heatmap(return_fig=True)
            if fig and fig != self:
                html = fig.to_html(full_html=False, include_plotlyjs='cdn' if first else False)
                assoc_charts.append(f"<div class='chart-card' style='margin-bottom: 30px;'>{html}</div>")
        if assoc_charts:
            tabs["Correlations Heatmap"] = assoc_charts
                
        # Restore the full dataset
        self.df = original_df
                
        full_html = compose_report(tabs, title="DataWash Automated Report")
        
        filepath = os.path.abspath(filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_html)
            
        logger.info(f"Report saved successfully to {filepath}")
        
        if auto_open:
            webbrowser.open('file://' + filepath.replace('\\', '/'))
            
        return full_html

    def show_dashboard(self):
        """
        Magically injects a breathtaking static HTML dashboard directly into 
        a Google Colab notebook cell or opens it in your local web browser.
        Zero background servers, zero 500 errors!
        """
        logger.info("Generating Serverless Dashboard... Please wait a moment.")
        
        # Check if we are in Colab/Jupyter
        in_colab = False
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                in_colab = True
        except ImportError:
            pass
            
        if in_colab:
            try:
                from google.colab import output
                import subprocess
                import socket
                
                # 1. Generate the FULL static HTML report (with the embedded CSV download button!)
                # We can safely embed the 10MB CSV now because we aren't using iframes
                self.generate_html_report(filename="datawash_dashboard.html", auto_open=False, include_download_btn=True)
                
                # 2. Find an open port dynamically to avoid "Address already in use" errors on multiple runs
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("",0))
                s.listen(1)
                port = s.getsockname()[1]
                s.close()
                
                # 3. Start a background web server
                subprocess.Popen(["python", "-m", "http.server", str(port)])
                
                # 4. Open securely in a Colab iframe using the proxy server
                print("\\n[SUCCESS] Opening Premium Dashboard securely via Colab Proxy!")
                output.serve_kernel_port_as_iframe(port, path='/datawash_dashboard.html', width='100%', height='800')
            except Exception as e:
                # Fallback to standard local behavior if anything fails
                print(f"\\n[WARNING] Colab proxy server failed ({e}). Falling back to local generation.")
                self.generate_html_report(filename="datawash_dashboard.html", auto_open=True, include_download_btn=True)
        else:
            # Local IDE usage
            print("\n[SUCCESS] Opening Serverless Dashboard in your web browser!")
            self.generate_html_report(filename="datawash_dashboard.html", auto_open=True)
            
        return self

    def drop_redundant_features(self, threshold=0.9):
        """
        Automatically drops features that are highly correlated with each other
        (association score > threshold) to prevent multicollinearity.
        
        Parameters
        ----------
        threshold : float
            The association score (0 to 1) above which two features are
            considered redundant. The feature that appears later in the
            dataframe will be dropped.
        """
        if not self._check_data():
            return self
            
        logger.info(f"Calculating associations to identify redundant features (threshold > {threshold})...")
        assoc_matrix = self._compute_associations_matrix()
        
        if assoc_matrix is None:
            return self
            
        # Get the upper triangle of the matrix to avoid dropping both features in a pair
        # and to avoid comparing a feature to itself
        upper_tri = assoc_matrix.where(np.triu(np.ones(assoc_matrix.shape), k=1).astype(bool))
        
        # Find columns where any correlation is greater than the threshold
        to_drop = set()
        pairs_log = []
        for col in upper_tri.columns:
            for row in upper_tri.index:
                val = upper_tri.loc[row, col]
                if pd.notna(val) and val > threshold:
                    pairs_log.append((row, col, round(val, 3)))
                    to_drop.add(col)  # drop the later column

        if pairs_log:
            logger.info(f"\n  Redundant pairs (score > {threshold}):")
            for feat_a, feat_b, score in pairs_log:
                action = "DROPPING" if feat_b in to_drop else "KEEPING"
                logger.info(f"    {feat_a:30s} <-> {feat_b:30s}  score={score}  [{action} {feat_b}]")

        to_drop = list(to_drop)
        if to_drop:
            self.df = self.df.drop(columns=to_drop)
            logger.info(f"\n  Dropped {len(to_drop)} redundant features: {to_drop}")
        else:
            logger.info("No redundant features found above the threshold.")
            
        return self