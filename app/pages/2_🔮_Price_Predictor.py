import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Price Predictor - Helsinki House Prices",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 Property Price Predictor")
st.write("Get instant price estimates for Helsinki metropolitan area properties")


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


@st.cache_resource
def train_model():
    data = load_data()
    data = data[data.Total_rooms.notna() & data.Latitude.notna()]

    X = data[["Size", "Year", "Total_rooms", "Latitude", "Longitude"]]
    y = data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    forest = RandomForestRegressor(random_state=42)
    forest.fit(X_train.values, y_train.values)

    return forest, data, y


with st.spinner("Loading model..."):
    forest, data, y = train_model()

st.success("✅ Model loaded successfully")

st.header("🏠 Property Details")
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
        "Kruununhaka": (60.1708, 24.9522),
        "Ullanlinna": (60.1580, 24.9420),
        "Eira": (60.1550, 24.9380),
        "Katajanokka": (60.1680, 24.9620),
        "Pasila": (60.1990, 24.9340),
        "Hakaniemi": (60.1790, 24.9510),
        "Sörnäinen": (60.1850, 24.9650),
        "Hermanni": (60.1920, 24.9720),
        "Arabianranta": (60.2100, 24.9800),
        "Vuosaari": (60.2090, 25.1420),
        "Espoo Center": (60.2055, 24.6559),
        "Tapiola": (60.1760, 24.8050),
        "Otaniemi": (60.1840, 24.8300),
        "Leppävaara": (60.2190, 24.8130),
        "Matinkylä": (60.1610, 24.7360),
        "Vantaa Center": (60.2934, 25.0378),
        "Tikkurila": (60.2920, 25.0440),
        "Myyrmäki": (60.2640, 24.8450),
        "Haaga": (60.2100, 24.8900),
        "Munkkiniemi": (60.1900, 24.8700),
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
        "Kruununhaka": "Historic district, government area",
        "Ullanlinna": "Elegant residential, embassy district",
        "Eira": "Prestigious waterfront area",
        "Katajanokka": "Island district, cruise terminal",
        "Pasila": "Business district, good connections",
        "Hakaniemi": "Market square area, authentic Helsinki",
        "Sörnäinen": "Industrial heritage, emerging area",
        "Hermanni": "Residential, family-friendly",
        "Arabianranta": "Waterfront living, modern development",
        "Vuosaari": "Eastern Helsinki, harbor area",
        "Espoo Center": "Modern suburb, tech hub",
        "Tapiola": "Garden city, cultural center",
        "Otaniemi": "University area, tech campus",
        "Leppävaara": "Shopping and transport hub",
        "Matinkylä": "Metro connection, coastal",
        "Vantaa Center": "Airport area, good value",
        "Tikkurila": "Cultural center, good transport",
        "Myyrmäki": "Residential suburb, affordable",
        "Haaga": "Green residential area",
        "Munkkiniemi": "Upscale residential, seaside",
    }

    st.caption(f"ℹ️ {area_descriptions[selected_location]}")

    # Add some spacing
    st.write("")
    st.write("")

# Set coordinates based on selected preset
latitude, longitude = location_presets[selected_location]
st.write(f"📍 Selected: **{selected_location}** ({latitude:.4f}, {longitude:.4f})")

st.header("🔮 Price Prediction")

price_pred = forest.predict([[size, year, rooms, latitude, longitude]])
predicted_price = price_pred[0]

col1, col2 = st.columns([2, 1])

with col1:
    st.metric(
        label="💰 Estimated Property Price",
        value=f"{predicted_price:,.0f} €",
        help="Machine learning prediction based on your property specifications",
    )

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

# Additional insights
st.header("📊 Market Context")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📈 Price Distribution")
    similar_size_properties = data[
        (data["Size"] >= size - 10) & (data["Size"] <= size + 10)
    ]
    if len(similar_size_properties) > 0:
        st.metric(
            f"Similar Size Properties ({size-10:.0f}-{size+10:.0f} m²)",
            f"{len(similar_size_properties)} found",
        )
        st.metric(
            "Average Price (Similar Size)",
            f"{similar_size_properties['Price'].mean():,.0f} €",
        )
    else:
        st.write("No similar size properties found in dataset")

with col2:
    st.subheader("🏗️ Age Factor")
    similar_year_properties = data[
        (data["Year"] >= year - 5) & (data["Year"] <= year + 5)
    ]
    if len(similar_year_properties) > 0:
        st.metric(
            f"Similar Age Properties ({year-5}-{year+5})",
            f"{len(similar_year_properties)} found",
        )
        st.metric(
            "Average Price (Similar Age)",
            f"{similar_year_properties['Price'].mean():,.0f} €",
        )
    else:
        st.write("No similar age properties found in dataset")

with col3:
    st.subheader("🚪 Room Count")
    similar_room_properties = data[data["Total_rooms"] == rooms]
    if len(similar_room_properties) > 0:
        st.metric(
            f"{int(rooms)}-Room Properties", f"{len(similar_room_properties)} found"
        )
        st.metric(
            f"Average Price ({int(rooms)} rooms)",
            f"{similar_room_properties['Price'].mean():,.0f} €",
        )
    else:
        st.write(f"No {int(rooms)}-room properties found in dataset")

# Quick comparison tool
st.header("⚖️ Quick Comparison")
st.write("See how different property characteristics affect the price:")

comparison_col1, comparison_col2 = st.columns(2)

with comparison_col1:
    st.subheader("📐 Size Impact")
    sizes_to_compare = [size - 20, size, size + 20]
    size_predictions = []

    for s in sizes_to_compare:
        if s > 0:
            pred = forest.predict([[s, year, rooms, latitude, longitude]])[0]
            size_predictions.append(pred)
            st.write(f"**{s:.0f} m²**: {pred:,.0f} €")
        else:
            st.write(f"**{s:.0f} m²**: Invalid size")

with comparison_col2:
    st.subheader("🗓️ Year Impact")
    years_to_compare = [year - 10, year, year + 10]
    year_predictions = []

    for y_comp in years_to_compare:
        pred = forest.predict([[size, y_comp, rooms, latitude, longitude]])[0]
        year_predictions.append(pred)
        st.write(f"**{y_comp}**: {pred:,.0f} €")
