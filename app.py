import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import IsolationForest
import os

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Forecasting & Inventory Optimization Dashboard")

st.markdown("---")


# Load Dataset

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"],dayfirst=True)
    return df

df = load_data()


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

# Forecast values from Task 4 (Best Model: XGBoost)

forecast_data = {
    "Furniture": [31329.707031, 10251.143555, 7477.699219],
    "Technology": [20986.835938, 19069.994141, 26975.033203],
    "Office Supplies": [29322.787109, 25241.556641, 25241.556641],
    "West": [29513.925781, 11628.667969, 14750.610352],
    "East": [19910.339844, 16948.789062, 23313.078125]
}

# Best model metrics
best_model = "XGBoost"
mae = 13384.12
rmse = 18560.81

if page == "Sales Overview":

    st.header("📈 Sales Overview Dashboard")

    # Create two columns for filters
    col1, col2 = st.columns(2)
    
    with col1:
        selected_region = st.selectbox(
            "Select Region",
            ["All"] + sorted(df["Region"].unique().tolist())
        )
    
    with col2:
        selected_category = st.selectbox(
            "Select Category",
            ["All"] + sorted(df["Category"].unique().tolist())
    )

    filtered_df = df.copy()

    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]

    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["Category"] == selected_category]

    # Total Sales
    total_sales = filtered_df["Sales"].sum()

    st.metric("Total Sales", f"${total_sales:,.2f}")

    # Sales by Year
    yearly_sales = filtered_df.groupby(filtered_df["Order Date"].dt.year)["Sales"].sum()

    fig, ax = plt.subplots(figsize=(8,4))
    yearly_sales.plot(kind="bar", ax=ax)
    ax.set_title("Total Sales by Year")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # Monthly Sales Trend
    monthly_sales = (
        filtered_df
        .set_index("Order Date")
        .resample("ME")["Sales"]
        .sum()
    )

    fig2, ax2 = plt.subplots(figsize=(10,4))
    monthly_sales.plot(ax=ax2)

    ax2.set_title("Monthly Sales Trend")
    ax2.set_ylabel("Sales")

    st.pyplot(fig2)

# -----------------------------
# Forecast Explorer
# -----------------------------

elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    selection = st.selectbox(
        "Select Category or Region",
        list(forecast_data.keys())
    )

    horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    forecast_values = forecast_data[selection][:horizon]

    forecast_df = pd.DataFrame({
        "Month": [f"Month {i}" for i in range(1, horizon + 1)],
        "Forecast": forecast_values
    })

    st.subheader(f"{selection} Forecast")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(
        forecast_df["Month"],
        forecast_df["Forecast"],
        marker="o",
        linewidth=2
    )

    ax.set_ylabel("Forecasted Sales")
    ax.set_title(f"{selection} Sales Forecast")

    st.pyplot(fig)

    st.dataframe(forecast_df, use_container_width=True)

    st.markdown("### Model Performance")

    col1, col2 = st.columns(2)

    col1.metric("Model", best_model)
    col2.metric("Forecast Horizon", f"{horizon} Month(s)")

    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")

# -----------------------------
# Anomaly Report
# -----------------------------

elif page == "Anomaly Report":

    st.header("🚨 Anomaly Report")

    st.subheader("Detected Sales Anomalies")

    image_path = "charts/isolation_forest_anomalies.png"

    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image)
    else:
        st.warning("Anomaly chart not found.")

    st.markdown("---")
    st.subheader("Anomaly Dates and Sales Values")

    # Monthly sales
    monthly_sales = (
        df.groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
        .sum()
        .reset_index()
    )
    
    # Isolation Forest
    model = IsolationForest(
        contamination=0.10,
        random_state=42
    )
    
    monthly_sales["Anomaly"] = model.fit_predict(monthly_sales[["Sales"]])
    
    # Keep only anomalies
    anomalies = monthly_sales[monthly_sales["Anomaly"] == -1]
    
    # Display table
    st.dataframe(
        anomalies[["Order Date", "Sales"]],
        use_container_width=True
    )


# -----------------------------
# Product Demand Segments
# -----------------------------

elif page == "Product Demand Segments":

    st.header("📦 Product Demand Segments")

    st.subheader("K-Means Product Clusters")

    image_path = "charts/task6_product_clusters.png"

    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image)
    else:
        st.warning("Cluster chart not found.")

    st.markdown("---")
    st.subheader("Demand Cluster Assignment")

    # Aggregate features by Sub-Category
    cluster_df = df.groupby("Sub-Category").agg(
        Total_Sales=("Sales", "sum"),
        Average_Order_Value=("Sales", "mean")
    ).reset_index()
    
    # Assign simple cluster labels based on total sales
    cluster_df["Cluster"] = pd.qcut(
        cluster_df["Total_Sales"],
        q=4,
        labels=["Low", "Medium", "High", "Very High"]
    )
    
    # Display table
    st.dataframe(
        cluster_df[["Sub-Category", "Cluster"]],
        use_container_width=True
    )
    st.subheader("Stocking Strategy Recommendations")

    strategy = pd.DataFrame({
        "Demand Cluster": ["Low", "Medium", "High", "Very High"],
        "Recommended Stocking Strategy": [
            "Maintain minimum inventory and monitor demand.",
            "Keep moderate stock and review periodically.",
            "Maintain higher inventory to meet consistent demand.",
            "Prioritize inventory replenishment and safety stock."
        ]
    })
    
    st.table(strategy)

        
