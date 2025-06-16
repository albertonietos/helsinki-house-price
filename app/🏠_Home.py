import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Configure page
st.set_page_config(
    page_title="Helsinki House Price Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 Helsinki House Price Analysis")
st.write("A comprehensive analysis of Helsinki metropolitan area property prices.")
st.write("**By [Alberto Nieto](https://www.linkedin.com/in/albertonietosandino/)**")

st.markdown("---")

# Quick overview
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
    ### 📊 Data Explorer
    Explore the dataset with interactive maps and visualizations
    - Geographic distribution of properties
    - Price and size patterns
    - Interactive color schemes
    """
    )
    if st.button("🗺️ Explore Data", use_container_width=True):
        st.switch_page("pages/1_📊_Data_Explorer.py")

with col2:
    st.markdown(
        """
    ### 🔮 Price Predictor
    Get instant price estimates for Helsinki properties
    - Machine learning predictions
    - Property details input
    - Market comparison
    """
    )
    if st.button("💰 Predict Price", use_container_width=True):
        st.switch_page("pages/2_🔮_Price_Predictor.py")

with col3:
    st.markdown(
        """
    ### 🧠 Model Analysis
    Deep dive into model performance and feature importance
    - Model metrics and accuracy
    - Feature importance analysis
    - Partial dependence plots
    """
    )
    if st.button("📈 Analyze Model", use_container_width=True):
        st.switch_page("pages/3_🧠_Model_Analysis.py")

st.markdown("---")


# Quick stats
@st.cache_data
def load_data():
    try:
        return pd.read_excel(
            "./data/cleaned/helsinki_house_price_cleaned.xls", engine="xlrd"
        )
    except FileNotFoundError:
        st.error("❌ Data file not found. Please check that the data file exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()


data = load_data()
data = data[data.Total_rooms.notna() & data.Latitude.notna()]

st.subheader("📈 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Properties", f"{len(data):,}")
with col2:
    st.metric("Average Price", f"{data['Price'].mean():,.0f} €")
with col3:
    st.metric("Average Size", f"{data['Size'].mean():.0f} m²")
with col4:
    st.metric(
        "Price Range", f"{data['Price'].min():,.0f} - {data['Price'].max():,.0f} €"
    )

# Quick predictor
st.subheader("🚀 Quick Price Estimate")
st.write("Get a fast estimate - or use the full predictor for detailed analysis")

col1, col2, col3 = st.columns(3)
with col1:
    size = st.slider("Size (m²)", 20, 200, 80)
with col2:
    year = st.slider("Year Built", 1950, 2024, 1990)
with col3:
    rooms = st.selectbox("Rooms", [1, 2, 3, 4, 5, 6], index=2)

# Quick prediction
X = data[["Size", "Year", "Total_rooms", "Latitude", "Longitude"]]
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
forest = RandomForestRegressor(random_state=42)
forest.fit(X_train.values, y_train.values)

# Use Helsinki Center coordinates for quick estimate
helsinki_lat, helsinki_lon = 60.1699, 24.9384
quick_pred = forest.predict([[size, year, rooms, helsinki_lat, helsinki_lon]])

st.success(f"💰 **Quick Estimate**: {quick_pred[0]:,.0f} € (Helsinki Center location)")
st.caption(
    "For detailed predictions with location selection, use the full Price Predictor"
)

st.markdown("---")

st.markdown(
    """
### 📋 About This Analysis

This application analyzes property prices in the Helsinki metropolitan area using data scraped from [Etuovi.com](https://www.etuovi.com/) from 2021.
The analysis includes properties from Helsinki, Vantaa, and Espoo.

**Features:**
- Interactive data visualization with multiple color schemes
- Machine learning price prediction using a Random Forest model
- Comprehensive model analysis with partial dependence plots
- Feature importance analysis to understand the model's predictions

"""
)
