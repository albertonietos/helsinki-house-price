import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Configure page
st.set_page_config(
    page_title="Helsinki House Price Predictor",
    page_icon="🏠",
    initial_sidebar_state="collapsed"
)

st.title('🏠 Helsinki House Price Predictor')
st.write('A project by [**Alberto Nieto**](https://www.linkedin.com/in/albertonietosandino/)')
DATA_URL = './data/cleaned/helsinki_house_price_cleaned.xls'

# DATA_URL = ('https://github.com/albertonietos/helsinki-house-price/tree/main/data/cleaned/helsinki_house_price_cleaned.xls')

st.header("The Data")
# Create a text element and let the reader know the data is loading.
# Load data into the dataframe.
@st.cache_data
def load_data():
	try:
		return pd.read_excel(DATA_URL, engine='xlrd')
	except FileNotFoundError:
		st.error("❌ Data file not found. Please check that the data file exists.")
		st.stop()
	except Exception as e:
		st.error(f"❌ Error loading data: {str(e)}")
		st.stop()

with st.spinner('Loading data...'):
	data = load_data()

st.success(f"✅ Successfully loaded {len(data)} property records")

st.write("""
	Below, you can take a look at the raw data that is used
	for all the analysis. The data was scraped from [**Etuovi**](https://www.etuovi.com/)
	by looking for the entries for the cities of Helsinki, Vantaa and Espoo.
	""")
if st.checkbox('Show raw data'):
	st.write(data)

# Clean data and inform user
initial_count = len(data)
data = data[data.Total_rooms.notna() & data.Latitude.notna()]
removed_count = initial_count - len(data)

if removed_count > 0:
    st.info(f"ℹ️ Removed {removed_count} properties with missing data (rooms or location)")
X = data[['Size', 'Year', 'Total_rooms', 'Latitude', 'Longitude']]
y = data['Price']
data = data.rename(columns={"Latitude": "lat", "Longitude": "lon"})

st.write("""
	An image is worth a 1000 comma separated values,
	so next I give you a visual representation of the location
	of those ads. Each point represents an ad for a house or apartment.
	""")
st.map(data, zoom=9)

def map_3d(data, lat, lon, zoom):
    st.write(pdk.Deck(
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
        ]
    ))

st.write("""
	In order to appreciate the density of the ads, we can aggregate
	the entries in histogram bins in the following 3D map. The red high
	bars likely represent newly built apartment housing in which all the
	apartments have been advertised at the same time. 
	""")
midpoint = (np.average(data["lat"]), np.average(data["lon"]))
map_3d(data, midpoint[0], midpoint[1], 9.5)

st.header("The Model")
st.write("As a default model for fast prototyping, we can train a random forest regressor to predict the price of the ads.")
st.write("The variables used to predict the price are: size, year of construction, total number of rooms, latitude and longitude.")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

forest = RandomForestRegressor()
forest.fit(X_train.values, y_train.values)
train_score, test_score = forest.score(X_train.values, y_train.values), forest.score(X_test.values, y_test.values)

st.write(f"The model achieves an $R^2$ of {train_score:.2f} on the train set and an $R^2$ of {test_score:.2f} in the test set.")

st.header("🏠 Property Price Predictor")
st.write("Enter your property details below to get an instant price estimate:")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🏡 Property Details")
    
    size = st.slider(
        '📐 Size (square meters)',
        min_value=np.min(data["Size"]), 
        max_value=np.max(data["Size"]),
        value=80.0,
        step=5.0,
        help="Total living area in square meters"
    )
    
    year = st.slider(
        '🗓️ Year built',
        min_value=int(np.min(data["Year"])),
        max_value=int(np.max(data["Year"])),
        step=1,
        value=1990,
        help="Year when the property was constructed"
    )
    
    rooms = st.selectbox(
        '🚪 Total rooms',
        options=np.unique(data["Total_rooms"]).tolist(),
        index=2,  # Default to 3 rooms
        help="Total number of rooms (bedrooms, living room, kitchen, etc.)"
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
        "Vantaa Center": (60.2934, 25.0378)
    }
    
    selected_location = st.selectbox(
        "📌 Choose area:",
        options=list(location_presets.keys()),
        index=0,  # Default to Helsinki Center
        help="Select a location in the Helsinki metropolitan area"
    )
    
    # Show a brief description of the selected area
    area_descriptions = {
        "Helsinki Center": "City center, close to everything",
        "Kamppi": "Shopping district, excellent transport",
        "Kallio": "Trendy neighborhood, vibrant nightlife", 
        "Punavuori": "Upscale area, design district",
        "Töölö": "Quiet residential, near parks",
        "Espoo Center": "Modern suburb, tech hub",
        "Vantaa Center": "Airport area, good value"
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
        value=f"€{predicted_price:,.0f}",
        help="Machine learning prediction based on your property specifications"
    )
    
    # Calculate price per square meter
    price_per_sqm = predicted_price / size
    st.metric(
        label="📏 Price per m²",
        value=f"€{price_per_sqm:,.0f}/m²"
    )

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
        value=f"€{avg_price:,.0f}",
        delta=f"{percentage_diff:+.1f}%",
        help="Average price of all properties in the dataset"
    )

st.info("💡 **Note**: This prediction is based on limited data. Always consult real estate professionals for accurate valuations.")


st.markdown("**Note**: _All models are wrong, some models are useful._")

st.header("The importance of the features")
st.write("""As important as the predictive power of the model, 
	the importance that our model assigns to each feature is 
	-for lack of a better word- important. We are also interested in the relationship that our
	model assigns between each independent variable and the target variable.""")

st.subheader("Partial dependence")
st.write("One way to approach this is using partial dependence plots.")
st.image("./images/Partial_dependence.png")
st.write("_If you're using the dark mode version of the website, this plot may have destroyed your retina. I apologize._")
st.markdown("""It's quite interesting to notice certain things:
- The size of the house is linearly correlated with the price. Nothing unexpected here.
- The price of the house has a very interesting relationship with the age of the housing. The lowest point is not at the oldest, but rather in the middle. It could be that the lowest peak corresponds to a period of time where a lot of affordable housing was built resulting in a lower price for those houses. Meanwhile, older housing which is still livable likely corresponds to higher standard of living housing.
- The number of total rooms is only linear up until 4 rooms. Larger number of rooms doesn't give a significant increase in price. It could be because a larger number of rooms does not always mean larger size, resulting in housing where a large number of people live in a small space.
- The smallest values of latitude are the priciest. Makes sense since the lower the latitutude, the souther. And the center of Helsinki and its best locations are placed in the south, next to the coast.
- Longitude (east-to-west) has a central band where the price is highest. Normal, given than the center of Helsinki is located in the middle with other properties on the sides.""")

st.subheader("Feature importance")
st.write("""We are not only interested about how the input variables are related to the output but rather how much weight does 
	each variable carry in the prediction.
	This is what the feature importance represents.""")
st.image("./images/Feature_importance.png")
st.markdown("""As we can see, this model assigns an overwhelming **60%** (more actually) of the importance
	to the size of the housing. Other variables like the year of construction and the number of rooms carry much less weight.
	This could be a problem as our model is over reliant in this feature. A noisy input or some error in the size inputation
	would render the model inaccurate.""")
st.markdown("""Interestingly, the latitude carries much more weight than the longitude.
	Meaning that North-to-South is a much larger indicator of price than West-to-East.
	This entails that closeness to the coast is much more importance than which side of the city you are in.""")
st.write("To be continued...(?)")