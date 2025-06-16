import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="Data Explorer - Helsinki House Prices",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Explorer")
st.write("Explore the Helsinki property dataset with interactive visualizations")


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


with st.spinner("Loading data..."):
    data = load_data()

st.success(f"✅ Successfully loaded {len(data)} property records")

# Data overview section
st.header("📋 Dataset Overview")

col1, col2 = st.columns([2, 1])

with col1:
    st.write(
        """
    This dataset contains property listings scraped from [**Etuovi**](https://www.etuovi.com/)
    covering Helsinki, Vantaa and Espoo areas.
    """
    )

    if st.checkbox("Display a sample of the data"):
        st.write(data.sample(10))

with col2:
    # Clean data and inform user
    initial_count = len(data)
    data = data[data.Total_rooms.notna() & data.Latitude.notna()]
    removed_count = initial_count - len(data)

    if removed_count > 0:
        st.info(f"ℹ️ Removed {removed_count} properties with missing data")

# Prepare data for visualization
X = data[["Size", "Year", "Total_rooms", "Latitude", "Longitude"]]
y = data["Price"]
data = data.rename(columns={"Latitude": "lat", "Longitude": "lon"})

# Map visualization section
st.header("🗺️ Interactive Property Map")

st.write(
    """
The map below shows the geographic distribution of the most relevantproperties in our dataset.
Each point represents a property listing, with colors indicating price levels
and dot sizes representing the property size in square meters.

There are different color schemes to help with visualizing the price of the properties as the distribution contains a wide range of prices.
"""
)

# Color mapping options
color_scheme = st.selectbox(
    "🎨 Choose color scheme:",
    options=["Percentile-based", "Logarithmic", "Price tiers", "Linear (original)"],
    index=0,
    help="Different ways to map prices to colors for better visualization",
)

min_price, max_price = data["Price"].min(), data["Price"].max()
median_price = data["Price"].median()
q25, q75 = data["Price"].quantile([0.25, 0.75])

data_viz = data.copy()

if color_scheme == "Percentile-based":
    data_viz["price_normalized"] = data["Price"].rank(pct=True) * 255
    scheme_info = "Colors based on price percentiles - more balanced distribution"

elif color_scheme == "Logarithmic":
    # Add small constant to avoid log(0) and ensure positive values
    safe_prices = np.maximum(data["Price"], 1)
    log_prices = np.log10(safe_prices)
    min_log, max_log = log_prices.min(), log_prices.max()
    data_viz["price_normalized"] = (
        (log_prices - min_log) / (max_log - min_log) * 255
    ).astype(int)
    scheme_info = "Logarithmic scaling - good for wide price ranges"

elif color_scheme == "Price tiers":

    def price_to_tier(price):
        if price < q25:
            return 50
        elif price < median_price:
            return 100
        elif price < q75:
            return 150
        else:
            return 200

    data_viz["price_normalized"] = data["Price"].apply(price_to_tier)
    scheme_info = f"4 price tiers: <{q25:,.0f} € | {q25:,.0f}-{median_price:,.0f} € | {median_price:,.0f}-{q75:,.0f} € | >{q75:,.0f} €"

else:  # Linear (original)
    data_viz["price_normalized"] = (
        (data["Price"] - min_price) / (max_price - min_price) * 255
    ).astype(int)
    scheme_info = "Linear scaling from min to max price"


def price_to_color(normalized_price):
    red = int(normalized_price)
    blue = int(255 - normalized_price)
    return [red, 0, blue, 160]


data_viz["color"] = data_viz["price_normalized"].apply(price_to_color)

min_size, max_size = data["Size"].min(), data["Size"].max()
data_viz["size"] = (data["Size"] - min_size) / (max_size - min_size) * 50 + 10

st.info(f"💡 **{scheme_info}**")
st.markdown("**Visual Legend:**")
st.markdown(
    """
- **🎨 Color:**
    - 🔵 Low Price
    - 🟡 Medium
    - 🔴 High Price
- **📏 Size:**
    - Small dots = Small properties
    - Large dots = Large properties
"""
)

# Price and size statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Lowest Price", f"{min_price:,.0f} €")
with col2:
    st.metric("Median Price", f"{median_price:,.0f} €")
with col3:
    st.metric("Mean Price", f"{data['Price'].mean():,.0f} €")
with col4:
    st.metric("Highest Price", f"{max_price:,.0f} €")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Smallest Property", f"{min_size:.0f} m²")
with col2:
    st.metric("Median Property Size", f"{data['Size'].median():,.0f} m²")
with col3:
    st.metric("Mean Property Size", f"{data['Size'].mean():,.0f} m²")
with col4:
    st.metric("Largest Property", f"{max_size:.0f} m²")

data_viz["price_formatted"] = data_viz["Price"].apply(lambda x: f"{x:,.0f} €")

# Interactive map
st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=data["lat"].mean(),
            longitude=data["lon"].mean(),
            zoom=9,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=data_viz,
                get_position="[lon, lat]",
                get_color="color",
                get_radius="size",
                radius_scale=20,
                radius_min_pixels=3,
                radius_max_pixels=60,
                pickable=True,
                auto_highlight=True,
            ),
        ],
        tooltip={
            "html": "<b>Price:</b> {price_formatted}<br/><b>Size:</b> {Size} m²<br/><b>Year:</b> {Year}<br/><b>Rooms:</b> {Total_rooms}",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        },
    )
)


# 3D density map
def map_3d(data, lat, lon, zoom):
    st.write(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state={
                "latitude": lat,
                "longitude": lon,
                "zoom": zoom,
                "pitch": 50,
            },
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=data,
                    get_position=["lon", "lat"],
                    radius=100,
                    elevation_scale=4,
                    elevation_range=[0, 1000],
                    pickable=True,
                    extruded=True,
                ),
            ],
        )
    )


st.header("🏗️ 3D Property Density")
st.write(
    """
In order to appreciate the density of the ads, we can aggregate the entries in histogram bins in the following 3D map. The red high bars likely represent newly built apartment housing in which all the apartments have been advertised at the same time.
"""
)

midpoint = (np.average(data["lat"]), np.average(data["lon"]))
map_3d(data, midpoint[0], midpoint[1], 9.5)
