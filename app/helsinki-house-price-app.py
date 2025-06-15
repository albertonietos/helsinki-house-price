import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Configure page
st.set_page_config(
    page_title="Helsinki House Price Analysis",
    page_icon="🏠",
    initial_sidebar_state="collapsed",
)

st.title("🏠 Helsinki House Price Analysis")
st.write(
    "A project by [**Alberto Nieto**](https://www.linkedin.com/in/albertonietosandino/)"
)
DATA_URL = "./data/cleaned/helsinki_house_price_cleaned.xls"

st.header("The Data")


@st.cache_data
def load_data():
    try:
        return pd.read_excel(DATA_URL, engine="xlrd")
    except FileNotFoundError:
        st.error("❌ Data file not found. Please check that the data file exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()


with st.spinner("Loading data..."):
    data = load_data()

st.success(f"✅ Successfully loaded {len(data)} property records")

st.write(
    """
    Below, you can take a look at the raw data that is used
    for all the analysis. The data was scraped from [**Etuovi**](https://www.etuovi.com/)
    by looking for the entries for the cities of Helsinki, Vantaa and Espoo.
    """
)
if st.checkbox("Display a sample of the data"):
    st.write(data.sample(10))

# Clean data and inform user
initial_count = len(data)
data = data[data.Total_rooms.notna() & data.Latitude.notna()]
removed_count = initial_count - len(data)

if removed_count > 0:
    st.info(
        f"ℹ️ Removed {removed_count} properties with missing data (rooms or location)"
    )
X = data[["Size", "Year", "Total_rooms", "Latitude", "Longitude"]]
y = data["Price"]
data = data.rename(columns={"Latitude": "lat", "Longitude": "lon"})


st.write(
    """
    The map below shows the geographic distribution of the properties in our dataset.
    Each point represents a property listing, with colors indicating price levels
    and dot sizes representing the property size in square meters.
    """
)

# Color mapping options
color_scheme = st.selectbox(
    "🎨 Choose color scheme:",
    options=["Percentile-based", "Logarithmic", "Price tiers", "Linear (original)"],
    index=0,
    help="Different ways to map prices to colors for better visualization",
)

# Get price statistics for better color mapping
min_price, max_price = data["Price"].min(), data["Price"].max()
median_price = data["Price"].median()
q25, q75 = data["Price"].quantile([0.25, 0.75])

data_viz = data.copy()

if color_scheme == "Percentile-based":
    # Use percentiles for more balanced color distribution
    data_viz["price_normalized"] = data["Price"].rank(pct=True) * 255
    scheme_info = "Colors based on price percentiles - more balanced distribution"

elif color_scheme == "Logarithmic":
    # Logarithmic scaling for wide ranges
    # Add small constant to avoid log(0) and ensure positive values
    safe_prices = np.maximum(data["Price"], 1)  # Minimum price of €1
    log_prices = np.log10(safe_prices)
    min_log, max_log = log_prices.min(), log_prices.max()
    data_viz["price_normalized"] = (
        (log_prices - min_log) / (max_log - min_log) * 255
    ).astype(int)
    scheme_info = "Logarithmic scaling - good for wide price ranges"

elif color_scheme == "Price tiers":
    # Tier-based coloring
    def price_to_tier(price):
        if price < q25:
            return 50  # Blue
        elif price < median_price:
            return 100  # Light blue
        elif price < q75:
            return 150  # Yellow
        else:
            return 200  # Red

    data_viz["price_normalized"] = data["Price"].apply(price_to_tier)
    scheme_info = f"4 price tiers: <{q25:,.0f} € | {q25:,.0f}-{median_price:,.0f} € | {median_price:,.0f}-{q75:,.0f} € | >{q75:,.0f} €"

else:  # Linear (original)
    data_viz["price_normalized"] = (
        (data["Price"] - min_price) / (max_price - min_price) * 255
    ).astype(int)
    scheme_info = "Linear scaling from min to max price"


# Create color based on price (blue to red gradient)
def price_to_color(normalized_price):
    # Blue (low) to Red (high) gradient
    red = int(normalized_price)
    blue = int(255 - normalized_price)
    return [red, 0, blue, 160]


data_viz["color"] = data_viz["price_normalized"].apply(price_to_color)

# Size dots based on property size (m²), not price
min_size, max_size = data["Size"].min(), data["Size"].max()
data_viz["size"] = (data["Size"] - min_size) / (
    max_size - min_size
) * 50 + 10  # Size 10-60 based on property size (m²)

# Price legend with scheme information
st.info(f"💡 **{scheme_info}**")
st.markdown("**Visual Legend:**")
st.markdown("- **🎨 Color:** 🔵 Low Price → 🟡 Medium → 🔴 High Price")
st.markdown(
    "- **📏 Size:** Small dots = Small properties, Large dots = Large properties"
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Lowest Price", f"{min_price:,.0f} €")
with col2:
    st.metric("Median Price", f"{median_price:,.0f} €")
with col3:
    st.metric("Mean Price", f"{data['Price'].mean():,.0f} €")
with col4:
    st.metric("Highest Price", f"{max_price:,.0f} €")

# Add size range info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Smallest Property", f"{min_size:.0f} m²")
with col2:
    st.metric("Median Property Size", f"{data['Size'].median():,.0f} m²")
with col3:
    st.metric("Mean Property Size", f"{data['Size'].mean():,.0f} m²")
with col4:
    st.metric("Largest Property", f"{max_size:.0f} m²")

# Add this before the pydeck chart
data_viz["price_formatted"] = data_viz["Price"].apply(lambda x: f"{x:,.0f} €")

# Enhanced scatter plot
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


st.write(
    """
    In order to appreciate the density of the ads, we can aggregate
    the entries in histogram bins in the following 3D map. The red high
    bars likely represent newly built apartment housing in which all the
    apartments have been advertised at the same time.
    """
)
midpoint = (np.average(data["lat"]), np.average(data["lon"]))
map_3d(data, midpoint[0], midpoint[1], 9.5)

st.header("The Model")
st.write(
    "As a default model for fast prototyping, we can train a random forest regressor to predict the price of the ads."
)
st.write(
    "The variables used to predict the price are: size, year of construction, total number of rooms, latitude and longitude."
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

forest = RandomForestRegressor()
forest.fit(X_train.values, y_train.values)
train_score, test_score = forest.score(X_train.values, y_train.values), forest.score(
    X_test.values, y_test.values
)
y_pred = forest.predict(X_test.values)

# Compute MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Display model performance metrics
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="R² Score (Test)",
        value=f"{test_score:.3f}",
    )

with col2:
    st.metric(
        label="Mean Absolute Error",
        value=f"{mae:,.0f} €",
    )

with col3:
    st.metric(
        label="Root Mean Square Error",
        value=f"{rmse:,.0f} €",
    )

# Add context for better understanding
st.markdown("**📖 What do these metrics mean?**")
st.markdown(
    f"""
- **R² Score**: The model explains {test_score*100:.1f}% of price variation. Higher is better (max 1.0).
- **MAE**: The average absolute error across all predictions. This is the typical prediction error.
- **RMSE**: Root mean square error - penalizes large errors more heavily than MAE.
"""
)

# Practical interpretation
percentage_error = (mae / y.mean()) * 100
st.info(
    f"💡 **In practical terms**: The average prediction error is {percentage_error:.1f}% of the typical property price ({y.mean():,.0f} €)."
)

st.header("🏠 Property Price Predictor")
st.write("Enter your property details below to get an instant price estimate:")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🏡 Property Details")

    # Calculate reasonable size range (using percentiles to avoid extreme outliers)
    size_5th = np.percentile(data["Size"], 5)
    size_95th = np.percentile(data["Size"], 95)

    # Checkbox to choose input type
    use_large_size = st.checkbox(f"Unusually large property (>{size_95th:.0f} m²)?")

    if use_large_size:
        # For large properties: number input
        size = st.number_input(
            "📐 Enter exact size (square meters):",
            min_value=size_95th,
            max_value=float(np.max(data["Size"])),
            value=size_95th + 20.0,  # Start slightly above threshold
            step=10.0,
            help=f"For very large properties (up to {np.max(data['Size']):.0f} m²)",
        )
    else:
        # For normal properties: slider
        size = st.slider(
            "📐 Size (square meters)",
            min_value=max(20.0, size_5th),  # At least 20 m², or 5th percentile
            max_value=size_95th,  # Up to 95th percentile
            value=80.0,
            step=5.0,
            help=f"Total living area. Range covers 90% of properties ({size_5th:.0f}-{size_95th:.0f} m²)",
        )

    year = st.slider(
        "🗓️ Year built",
        min_value=int(np.min(data["Year"])),
        max_value=int(np.max(data["Year"])),
        step=1,
        value=1990,
        help="Year when the property was constructed",
    )

    rooms = st.selectbox(
        "🚪 Total rooms",
        options=np.unique(data["Total_rooms"]).tolist(),
        index=2,  # Default to 3 rooms
        help="Total number of rooms (bedrooms, living room, kitchen, etc.)",
    )

with col2:
    st.subheader("📍 Location")

    # Location presets for easy selection
    location_presets = {
        "Helsinki Center": (60.1699, 24.9384),
        "Kamppi": (60.1688, 24.9316),
        "Kallio": (60.1844, 24.9505),
        "Punavuori": (60.1641, 24.9402),
        "Töölö": (60.1756, 24.9193),
        "Espoo Center": (60.2055, 24.6559),
        "Vantaa Center": (60.2934, 25.0378),
    }

    selected_location = st.selectbox(
        "📌 Choose area:",
        options=list(location_presets.keys()),
        index=0,  # Default to Helsinki Center
        help="Select a location in the Helsinki metropolitan area",
    )

    # Show a brief description of the selected area
    area_descriptions = {
        "Helsinki Center": "City center, close to everything",
        "Kamppi": "Shopping district, excellent transport",
        "Kallio": "Trendy neighborhood, vibrant nightlife",
        "Punavuori": "Upscale area, design district",
        "Töölö": "Quiet residential, near parks",
        "Espoo Center": "Modern suburb, tech hub",
        "Vantaa Center": "Airport area, good value",
    }

    st.caption(f"ℹ️ {area_descriptions[selected_location]}")

    # Add some spacing
    st.write("")
    st.write("")

# Set coordinates based on selected preset
latitude, longitude = location_presets[selected_location]
st.write(f"📍 Selected: **{selected_location}** ({latitude:.4f}, {longitude:.4f})")

st.header("🔮 Price Prediction")

# Calculate prediction
price_pred = forest.predict([[size, year, rooms, latitude, longitude]])
predicted_price = price_pred[0]

# Display prediction prominently
col1, col2 = st.columns([2, 1])

with col1:
    st.metric(
        label="💰 Estimated Property Price",
        value=f"{predicted_price:,.0f} €",
        help="Machine learning prediction based on your property specifications",
    )

    # Calculate price per square meter
    price_per_sqm = predicted_price / size
    st.metric(label="📏 Price per m²", value=f"{price_per_sqm:,.0f} €/m²")

with col2:
    # Show comparison to average market price
    avg_price = y.mean()
    price_difference = predicted_price - avg_price
    percentage_diff = (price_difference / avg_price) * 100

    if percentage_diff > 20:
        st.success("💰 **Above Market Average**")
    elif percentage_diff < -20:
        st.info("💸 **Below Market Average**")
    else:
        st.warning("📊 **Around Market Average**")

    st.metric(
        label="vs. Market Average",
        value=f"{avg_price:,.0f} €",
        delta=f"{percentage_diff:+.1f}%",
        help="Average price of all properties in the dataset",
    )

st.info(
    "💡 **Note**: This prediction is based on limited data. And, as always, _all models are wrong, some models are useful_. The model can help in getting a sense of the price range of the property, but it is not meant to be a precise prediction. There are many other factors (observable and not) that are not taken into account in this model."
)


st.header("The importance of the features")
st.write(
    """As important as the predictive power of the model,
    the importance that our model assigns to each feature is
    -for lack of a better word- important. We are also interested in the relationship that our
    model assigns between each independent variable and the target variable."""
)

st.subheader("Partial dependence")
st.write("One way to approach this is using partial dependence plots.")
st.image("./images/Partial_dependence.png")
st.write(
    "_If you're using the dark mode version of the website, this plot may have destroyed your retina. I apologize._"
)
st.markdown(
    """It's quite interesting to notice certain things:
- The size of the house is linearly correlated with the price. Nothing unexpected here.
- The price of the house has a very interesting relationship with the age of the housing. The lowest point is not at the oldest, but rather in the middle. It could be that the lowest peak corresponds to a period of time where a lot of affordable housing was built resulting in a lower price for those houses. Meanwhile, older housing which is still livable likely corresponds to higher standard of living housing.
- The number of total rooms is only linear up until 4 rooms. Larger number of rooms doesn't give a significant increase in price. It could be because a larger number of rooms does not always mean larger size, resulting in housing where a large number of people live in a small space.
- The smallest values of latitude are the priciest. Makes sense since the lower the latitutude, the souther. And the center of Helsinki and its best locations are placed in the south, next to the coast.
- Longitude (east-to-west) has a central band where the price is highest. Normal, given than the center of Helsinki is located in the middle with other properties on the sides."""
)

st.subheader("Feature importance")
st.write(
    """We are not only interested about how the input variables are related to the output but rather how much weight does
    each variable carry in the prediction.
    This is what the feature importance represents."""
)
st.image("./images/Feature_importance.png")
st.markdown(
    """As we can see, this model assigns an overwhelming **60%** (more actually) of the importance
    to the size of the housing. Other variables like the year of construction and the number of rooms carry much less weight.
    This could be a problem as our model is over reliant in this feature. A noisy input or some error in the size inputation
    would render the model inaccurate."""
)
st.markdown(
    """Interestingly, the latitude carries much more weight than the longitude.
    Meaning that North-to-South is a much larger indicator of price than West-to-East.
    This entails that closeness to the coast is much more importance than which side of the city you are in."""
)
st.write("To be continued...(?)")
