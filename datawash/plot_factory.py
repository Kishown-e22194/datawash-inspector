import pandas as pd
import plotly.express as px
import logging

logger = logging.getLogger('datawash')

__all__ = ['PlottingMethods', 'compose_report']

class PlottingMethods:
    """
    Step 6 — Custom Modular Plotting
    =========================================================
    A decoupled class designed to generate granular Plotly charts.
    Instead of rendering them directly to the screen, it returns 
    HTML-wrapped strings suitable for embedding in web applications,
    dashboards, or automated email reports.
    """
    
    def __init__(self, df: pd.DataFrame = None):
        """
        Initializes the plotting factory.
        Accepts a clean DataFrame (e.g., pipeline.df_processed).
        """
        self.df = df

    def _validate_data(self, *columns):
        """Internal helper to ensure data and columns exist before plotting."""
        if self.df is None or self.df.empty:
            logger.error("Plotting failed: DataFrame is empty or None.")
            return False
            
        valid_cols = [c for c in columns if c is not None]
        for col in valid_cols:
            if col not in self.df.columns:
                logger.error(f"Plotting failed: Column '{col}' not found in dataset.")
                return False
                
        return True

    # =========================================================
    # MODULAR CHART GENERATORS (Returning HTML Strings)
    # =========================================================

    def get_bar_html(self, x, y=None, color=None, title=None, barmode='group',
                     include_plotlyjs='cdn'):
        """
        Generates an HTML-wrapped Bar Chart.
        If 'y' is None, it plots the frequency count of 'x'.

        Set *include_plotlyjs* to False for every chart after the first
        on the same page to avoid loading the Plotly library multiple times.
        """
        if not self._validate_data(x, y, color):
            return ""

        if y is None:
            # Aggregate counts if no Y axis is provided
            plot_df = self.df[x].value_counts().reset_index()
            plot_df.columns = [x, 'Count']
            # Bring the color column into the subset so px.bar can find it
            if color and color in self.df.columns:
                plot_df = plot_df.merge(
                    self.df[[x, color]].drop_duplicates(), on=x, how='left'
                )
            fig = px.bar(plot_df, x=x, y='Count', color=color, title=title or f"{x} Frequency")
        else:
            fig = px.bar(self.df, x=x, y=y, color=color, barmode=barmode, title=title or f"{y} by {x}")

        fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"))
        # full_html=False returns just the <div>.
        return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)

    def get_pie_html(self, names, values=None, title=None, include_plotlyjs='cdn'):
        """
        Generates an HTML-wrapped Pie Chart.
        If 'values' is None, it calculates the distribution of 'names'.

        Set *include_plotlyjs* to False for every chart after the first
        on the same page to avoid loading the Plotly library multiple times.
        """
        if not self._validate_data(names, values):
            return ""

        if values is not None:
            fig = px.pie(self.df, names=names, values=values, title=title or f"Distribution of {names}")
        else:
            # Aggregate counts if no specific numeric values are provided
            plot_df = self.df[names].value_counts().reset_index()
            plot_df.columns = [names, 'Count']
            fig = px.pie(plot_df, names=names, values='Count', title=title or f"{names} Share")
        
        # Clean up the visual presentation
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"))
        
        return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)

    def get_histogram_html(self, x, color=None, nbins=30, title=None, include_plotlyjs='cdn'):
        """
        Generates an HTML-wrapped Histogram for distribution analysis.

        Set *include_plotlyjs* to False for every chart after the first
        on the same page to avoid loading the Plotly library multiple times.
        """
        if not self._validate_data(x, color):
            return ""

        fig = px.histogram(self.df, x=x, color=color, nbins=nbins, title=title or f"Distribution of {x}")
        
        # Add a subtle gap between bars for readability
        fig.update_layout(
            bargap=0.1, 
            template="plotly_white", 
            paper_bgcolor="white", 
            plot_bgcolor="white", 
            font=dict(color="black")
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)

    def get_scatter_html(self, x, y, color=None, size=None, trendline=None,
                         title=None, include_plotlyjs='cdn'):
        """
        Generates an HTML-wrapped Scatter Plot.

        Parameters
        ----------
        x, y : str
            Column names for the X and Y axes.
        color : str, optional
            Column name to color-code points by.
        size : str, optional
            Column name to size points by.
        trendline : str, optional
            'ols' for a linear trendline (requires statsmodels).
        include_plotlyjs : str or bool
            Set to False for every chart after the first on the same page.
        """
        if not self._validate_data(x, y, color, size):
            return ""

        try:
            fig = px.scatter(
                self.df, x=x, y=y, color=color, size=size,
                trendline=trendline, opacity=0.7,
                title=title or f"{y} vs {x}",
            )
        except ImportError:
            # statsmodels not installed — fall back without trendline
            logger.info("'statsmodels' not installed — plotting without trendline.")
            fig = px.scatter(
                self.df, x=x, y=y, color=color, size=size,
                opacity=0.7, title=title or f"{y} vs {x}",
            )

        fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"))

        return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)

# =========================================================
# BATCH EXPORT UTILITIES
# =========================================================

def compose_report(tabs, title="DataWash Inspector"):
    """
    Wraps a dictionary of tabs (name -> list of HTML chart strings) 
    into a complete, styled, interactive modern HTML App.
    """
    tab_buttons = ""
    tab_contents = ""
    
    first = True
    for i, (tab_name, chart_list) in enumerate(tabs.items()):
        if not chart_list:
            continue
            
        tab_id = f"tab-{i}"
        active_class = "active" if first else ""
        display_style = "block" if first else "none"
        
        # Sidebar button
        tab_buttons += f'<button class="tablinks {active_class}" onclick="openTab(event, \'{tab_id}\')">{tab_name}</button>\n'
        
        # Tab content wrapper
        tab_contents += f'<div id="{tab_id}" class="tabcontent" style="display: {display_style};">\n'
        
        # Wrap each chart/content block in a modern glassmorphism card
        for chart_html in chart_list:
            tab_contents += f'<div class="chart-card">{chart_html}</div>\n'
            
        tab_contents += '</div>\n'
        first = False

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #f0f2f5;
            --sidebar-bg: #1e1e2d;
            --sidebar-hover: #2b2b40;
            --text-main: #333;
            --text-light: #888;
            --card-bg: rgba(255, 255, 255, 0.95);
            --accent: #4361ee;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}
        /* Sidebar styling */
        .sidebar {{
            width: 250px;
            background-color: var(--sidebar-bg);
            color: white;
            display: flex;
            flex-direction: column;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            z-index: 10;
        }}
        .sidebar-header {{
            padding: 20px;
            font-size: 1.2rem;
            font-weight: 600;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            text-align: center;
            letter-spacing: 1px;
        }}
        .sidebar button {{
            background: none;
            color: #ccc;
            border: none;
            padding: 15px 20px;
            text-align: left;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            border-left: 4px solid transparent;
        }}
        .sidebar button:hover {{
            background-color: var(--sidebar-hover);
            color: white;
        }}
        .sidebar button.active {{
            background-color: var(--sidebar-hover);
            color: white;
            border-left: 4px solid var(--accent);
            font-weight: 600;
        }}
        /* Main Content Area */
        .main-content {{
            flex-grow: 1;
            padding: 30px;
            overflow-y: auto;
            background-color: var(--bg-color);
        }}
        /* Glassmorphism Cards */
        .chart-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.05);
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            animation: fadeIn 0.4s ease;
            overflow-x: auto;
        }}
        /* Tables inside cards */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
</head>
<body>

    <div class="sidebar">
        <div class="sidebar-header">
            🧼 DataWash
        </div>
        {tab_buttons}
    </div>

    <div class="main-content">
        {tab_contents}
    </div>

    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
            
            // Trigger Plotly to resize graphs if they were hidden
            window.dispatchEvent(new Event('resize'));
        }}
    </script>
</body>
</html>'''