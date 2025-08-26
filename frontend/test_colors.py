#!/usr/bin/env python3
"""
Test script to verify plotly colors work correctly
"""

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

# Configure plotly for better colors and visibility
pio.templates.default = "plotly_white"

# Define a nice color palette for visualizations
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Create sample data
data = {
    'Category': ['A', 'B', 'C', 'D', 'E'],
    'Value': [10, 20, 15, 25, 30]
}
df = pd.DataFrame(data)

# Create a bar chart with proper colors
fig = px.bar(
    df, 
    x='Category', 
    y='Value',
    title='Test Visualization with Colors',
    template='plotly_white',
    color_discrete_sequence=COLOR_PALETTE
)

# Apply additional styling
fig.update_layout(
    font=dict(color='black'),
    paper_bgcolor='white',
    plot_bgcolor='white'
)

print("✅ Test visualization created with proper colors!")
print("Colors used:", COLOR_PALETTE[:5])
print("Template:", pio.templates.default)
