import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title('Helsinki house pricing')
st.write('A project by [**Alberto Nieto**](https://www.linkedin.com/in/albertonietosandino/)')
DATA_URL = ('./data/cleaned/helsinki_house_price_cleaned.xls')

#DATA_URL = ('github.com/albertonietos/helsinki-house-price/tree/main/data/cleaned/helsinki_house_price_cleaned.xls')

st.header("The Data")
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load data into the dataframe.
@st.cache
def load_data():
	return pd.read_excel(DATA_URL, index_col=0)
data = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

st.write("""
	Below, you can take a look at the raw data that is used
	for all the analysis. The data was scraped from [**Etuovi**](https://www.etuovi.com/)
	by looking for the entries for the cities of Helsinki, Vantaa and Espoo.
	""")
if st.checkbox('Show raw data'):
	st.write(data)

data = data[data.Total_rooms.notna() & data.Latitude.notna()]
X = data[['Size', 'Year', 'Total_rooms', 'Latitude', 'Longitude']]
y = data['Price']
data.rename(columns={"Latitude": "lat", "Longitude": "lon"}, inplace=True)

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
forest.fit(X_train, y_train)
train_score, test_score = forest.score(X_train, y_train), forest.score(X_test, y_test)

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
st.write(f'Given the limited data that the model is trained on, the price for housing with the given characteristics is estimated as {price_pred[0]:.2f} €.')


st.markdown("**Note**: _All models are wrong, some models are useful._")