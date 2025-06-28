import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

metadata_df = pd.read_csv('train_metadata - Copy.csv')

# Define approximate bounding box for India
india_df = metadata_df[
    (metadata_df['latitude'] >= 6) & (metadata_df['latitude'] <= 38) &
    (metadata_df['longitude'] >= 68) & (metadata_df['longitude'] <= 98)
]

# Create scatter plot for India only
fig = px.scatter_map(india_df, lat='latitude', lon='longitude', color='primary_label', 
                        hover_name='primary_label', hover_data=['latitude', 'longitude'], 
                        title='Geographical Distribution of Bird Species in India',
                        height=600)
fig.update_layout(mapbox_style="open-street-map")
fig.show()
