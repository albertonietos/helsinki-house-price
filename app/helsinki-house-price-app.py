import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title('Helsinki house pricing')
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

#with st.echo():
forest = RandomForestRegressor()
forest.fit(X_train.values, y_train.values)
train_score, test_score = forest.score(X_train.values, y_train.values), forest.score(X_test.values, y_test.values)

st.write(f"The model achieves an $R^2$ of {train_score:.2f} on the train set and an $R^2$ of {test_score:.2f} in the test set.")

st.header("The Prediction")
st.write("With that knowledge, feel free to try the model in the following prompt to see an estimation of the possible price")
size = st.slider(
	'Insert the size of the house or apartment (square meters)',
	min_value=np.min(data["Size"]), 
	max_value=np.max(data["Size"]),
	value=80.5)
year = st.slider(
	'Insert the year of construction',
	min_value=int(np.min(data["Year"])),
	max_value=int(np.max(data["Year"])),
	step=1,
	value=1975)
rooms = st.select_slider(
	'Insert the total number of rooms',
	options=np.unique(data["Total_rooms"]).tolist(),
	value=3)
latitude = st.slider(
	'Insert the approximate latitude',
	min_value=np.min(data["lat"]), 
	max_value=np.max(data["lat"]),
	value=60.2276)
longitude = st.slider(
	'Insert the approximate longitude',
	min_value=np.min(data["lon"]), 
	max_value=np.max(data["lon"]),
	value=24.7440)
price_pred = forest.predict([[size, year, rooms, latitude, longitude]])
st.write(f'Given the limited data that the model is trained on, the price for \
		housing with the given characteristics is estimated as {price_pred[0]:.2f} €.')


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