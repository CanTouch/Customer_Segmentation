# Core libraries for dashboarding and data manipulation
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime as dt
import os

# Set up the Streamlit page
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_and_prepare_data():
    """
    Loads, cleans, and prepares the data for RFM analysis.
    This function encapsulates all the data processing steps.
    """
    # Try loading from the workspace root if not found in 'data/raw/'
    data_path = 'data/raw/online-retail.xlsx'
    if not os.path.exists(data_path):
        data_path = 'Online Retail.xlsx'
    try:
        df = pd.read_excel(data_path)
    except FileNotFoundError:
        st.error(f"Error: '{data_path}' not found. Please ensure the file exists.")
        st.stop()
    
    # Data Cleaning and Preparation using 'pandas'
    df_cleaned = df.copy()
    df_cleaned = df_cleaned[df_cleaned['Quantity'] > 0]
    df_cleaned = df_cleaned[df_cleaned['UnitPrice'] > 0]
    df_cleaned.dropna(subset=['CustomerID'], inplace=True)
    df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype(str)
    
    # RFM Feature Engineering using 'pandas' and 'datetime'
    df_cleaned['Total'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']
    current_date = df_cleaned['InvoiceDate'].max() + dt.timedelta(days=1)
    
    rfm_table = df_cleaned.groupby('CustomerID').agg({
        'InvoiceDate': lambda date: (current_date - date.max()).days,
        'InvoiceNo': lambda num: num.nunique(),
        'Total': lambda price: price.sum()
    }).reset_index()
    
    rfm_table.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    return rfm_table

def scale_and_cluster_data(rfm_table, n_clusters):
    """
    Scales the RFM data and applies K-Means clustering.
    """
    # Apply log transformation to handle skewness using 'numpy'
    rfm_table['log_Frequency'] = np.log1p(rfm_table['Frequency'])
    rfm_table['log_Monetary'] = np.log1p(rfm_table['Monetary'])
    
    features_to_scale = ['Recency', 'log_Frequency', 'log_Monetary']
    X = rfm_table[features_to_scale].values
    
    # Scale the data using StandardScaler from 'scikit-learn'
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means clustering from 'scikit-learn'
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_table['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Add cluster centroids to the returned data
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=features_to_scale)
    
    return rfm_table, X_scaled, cluster_centers_df

def create_persona_summary(rfm_table, cluster_centers_df):
    """
    Analyzes cluster characteristics and generates a persona summary.
    """
    cluster_summary = rfm_table.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    # Map cluster numbers to persona names based on typical RFM characteristics
    # Note: These are a generalized example and may need to be adjusted based on the data
    persona_map = {}
    sorted_clusters = cluster_centers_df.sort_values(by='Monetary', ascending=False)
    
    if len(sorted_clusters) >= 2:
        persona_map[sorted_clusters.index[0]] = "Champions"
        persona_map[sorted_clusters.index[1]] = "Loyal Customers"
    if len(sorted_clusters) >= 3:
        persona_map[sorted_clusters.index[2]] = "Potential Promoters"
    if len(sorted_clusters) >= 4:
        persona_map[sorted_clusters.index[3]] = "At-Risk"
    if len(sorted_clusters) >= 5:
        persona_map[sorted_clusters.index[4]] = "Hibernating"
    if len(sorted_clusters) >= 6:
        persona_map[sorted_clusters.index[5]] = "Lost Customers"
    
    # Add a "Persona" column to the summary using 'pandas'
    cluster_summary['Persona'] = cluster_summary.index.map(persona_map)
    cluster_summary = cluster_summary.sort_values(by='Recency').reset_index()
    
    # Format the monetary values for display
    cluster_summary['Monetary'] = cluster_summary['Monetary'].apply(lambda x: f"${x:.2f}")
    
    # Get the raw RFM centroids and add persona names
    raw_centroids = cluster_centers_df.copy()
    raw_centroids['Cluster'] = raw_centroids.index
    raw_centroids['Persona'] = raw_centroids.index.map(persona_map)

    return cluster_summary, persona_map, raw_centroids

# Main application logic using 'streamlit'
st.title("Customer Segmentation Dashboard")
st.markdown("""
This interactive dashboard allows you to explore different customer segments based on their purchasing behavior.
The segmentation is performed using the Recency, Frequency, and Monetary (RFM) model and K-Means clustering.
""")

# Load the data once and cache it for performance with Streamlit's caching
@st.cache_data
def get_rfm_data():
    return load_and_prepare_data()

rfm_data = get_rfm_data()

# User input for number of clusters using Streamlit widgets
with st.sidebar:
    st.header("Dashboard Controls")
    num_clusters = st.slider(
        "Select the number of customer segments (k):",
        min_value=2, max_value=6, value=4, step=1
    )
    st.info(f"Choosing {num_clusters} segments allows for clear, actionable customer personas.")

# Run clustering based on user selection
rfm_clustered, X_scaled, cluster_centers = scale_and_cluster_data(rfm_data.copy(), num_clusters)
cluster_summary, persona_map, raw_centroids = create_persona_summary(rfm_clustered, cluster_centers)

# Display a high-level summary of the clusters using Streamlit's dataframe
st.subheader("Customer Persona Summary")
st.write("Each row in this table represents a distinct customer segment, defined by their average RFM values.")
st.dataframe(cluster_summary.set_index('Cluster'))

# Plot the 3D RFM clusters using 'plotly.express'
st.subheader("Visualizing the Customer Segments")
st.write("This 3D scatter plot shows how the customer segments are distributed in the RFM space.")
rfm_clustered['Persona'] = rfm_clustered['Cluster'].map(persona_map)

fig_3d = px.scatter_3d(
    rfm_clustered,
    x='Recency',
    y='Frequency',
    z='Monetary',
    color='Persona',
    title=f"3D RFM Clusters (k={num_clusters})",
    labels={'Recency': 'Recency (Days)', 'Frequency': 'Frequency (# of Orders)', 'Monetary': 'Monetary Value ($)'},
    height=600
)

# Use plotly's responsive layout
fig_3d.update_layout(
    margin=dict(l=0, r=0, b=0, t=40)
)
st.plotly_chart(fig_3d, use_container_width=True)

# Add a radar chart to visualize cluster centroids
st.subheader("Cluster Persona Radar Chart")
st.write("This radar chart visually compares the average RFM values of each customer segment.")
fig_radar = px.line_polar(
    raw_centroids,
    r=['Recency', 'Frequency', 'Monetary'],
    theta=['Recency', 'Frequency', 'Monetary'],
    color='Persona',
    line_close=True,
    title='Cluster Centroids',
    labels={'r': 'Metric Value'}
)
st.plotly_chart(fig_radar, use_container_width=True)


# Detailed analysis for each persona using Streamlit's selectbox and columns
st.subheader("Detailed Persona Profiles")
st.write("Select a persona below to see a more detailed breakdown of its characteristics and potential actions.")
selected_persona_name = st.selectbox(
    "Choose a persona:",
    options=list(persona_map.values())
)

selected_cluster_id = [k for k, v in persona_map.items() if v == selected_persona_name][0]
persona_data = rfm_clustered[rfm_clustered['Cluster'] == selected_cluster_id]
persona_metrics = cluster_summary[cluster_summary['Persona'] == selected_persona_name]

if not persona_metrics.empty:
    st.markdown(f"### The **{selected_persona_name}** Segment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Average Recency (Days)", value=f"{persona_metrics['Recency'].iloc[0]:.0f}")
        
    with col2:
        st.metric(label="Average Frequency (Orders)", value=f"{persona_metrics['Frequency'].iloc[0]:.0f}")
    
    with col3:
        st.metric(label="Average Monetary ($)", value=f"{persona_metrics['Monetary'].iloc[0]}")
    
    st.markdown(f"**Cluster Size**: This segment contains **{len(persona_data)}** customers.")
    
    # Provide actionable insights for the selected persona
    if selected_persona_name == "Champions":
        st.success("This segment is your most valuable. They buy recently, frequently, and spend the most. Focus on retention, loyalty programs, and personalized offers.")
    elif selected_persona_name == "Loyal Customers":
        st.success("These customers are consistent buyers. Keep them engaged with targeted cross-selling campaigns and special promotions.")
    elif selected_persona_name == "Potential Promoters":
        st.info("They have the potential to become loyal customers. Encourage more frequent purchases with 'buy one, get one' deals or special event invitations.")
    elif selected_persona_name == "At-Risk":
        st.warning("These customers haven't purchased in a while and are at risk of churning. Win them back with exclusive discounts or special 'we miss you' offers.")
    elif selected_persona_name == "Hibernating":
        st.warning("They were once active but have been inactive for a long time. Try to reactivate them with very compelling offers and surveys to understand their needs.")
    elif selected_persona_name == "Lost Customers":
        st.error("These are your least active customers. Consider a low-cost reactivation effort, but focus resources on more promising segments.")
    else:
        st.write("Select a persona from the dropdown to see more details.")
        
# Display the raw RFM data for reference
with st.expander("Show Raw RFM Data"):
    st.dataframe(rfm_clustered.head())