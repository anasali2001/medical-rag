import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go


def generate_trend_chart(df, test_name):
    """
    Generates a Plotly chart with existing data + 30-day prediction.
    """
    # Filter for specific test
    subset = df[df['test_name'] == test_name].copy()
    if len(subset) < 2:
        return None, "Not enough data points for prediction."

    subset['report_date'] = pd.to_datetime(subset['report_date'])
    subset = subset.sort_values('report_date')

    # --- ML Prediction ---
    # Convert date to ordinal (integer) for regression
    subset['date_ordinal'] = subset['report_date'].map(pd.Timestamp.toordinal)

    X = subset[['date_ordinal']]
    y = subset['value']

    model = LinearRegression()
    model.fit(X, y)

    # Predict 30 days out
    last_date = subset['report_date'].max()
    future_date = last_date + pd.Timedelta(days=30)
    future_df = pd.DataFrame({'date_ordinal': [future_date.toordinal()]})
    predicted_value = model.predict(future_df)[0]

    # --- Visualization ---
    fig = px.line(subset, x='report_date', y='value', markers=True, title=f"{test_name} Trends")

    # Color code points based on status
    colors = {'Normal': 'green', 'Abnormal': 'red', 'Unknown': 'grey'}
    fig.update_traces(marker=dict(size=10))

    # Add Prediction Point
    fig.add_trace(go.Scatter(
        x=[future_date],
        y=[predicted_value],
        mode='markers+text',
        marker=dict(color='orange', size=12, symbol='star'),
        name='Prediction (30d)',
        text=[f"{predicted_value:.2f}"],
        textposition="top center"
    ))

    return fig, predicted_value
