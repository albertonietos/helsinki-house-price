import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title('Helsinki house pricing')

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

st.write('Here, you can take a look at the raw data that is used for all the analysis.')
st.write(data)

data = data[data.Total_rooms.notna() & data.Latitude.notna()]
X = data[['Size', 'Year', 'Total_rooms', 'Latitude', 'Longitude']]
y = data['Price']

st.write("An image is worth a 1000 comma separated values, so next I give you a visual representation of the location of those ads.")
data.rename(columns={"Latitude": "lat", "Longitude": "lon"}, inplace=True)


st.map(data)

st.header("The Model")
st.write("As a default model for fast prototyping, we can train a random forest regressor to predict the price of the ads.")
st.write("The variables used to predict the price are: size, year of construction, total number of rooms, latitude and longitude.")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#with st.echo():
forest = RandomForestRegressor()
forest.fit(X_train, y_train)
train_score, test_score = forest.score(X_train, y_train), forest.score(X_test, y_test)

st.write("The model achieves an $R^2$ of ", train_score, " on the train set and an $R^2$ of ", test_score, " in the test set.")

st.header("The Prediction")
st.write("With that knowledge, feel free to try the model in the following prompt to see an estimation of the possible price")
size = st.number_input('Insert the size of the house or apartment (square meters)', value=80.5)
year = st.number_input('Insert the year of construction', value=1975)
rooms = st.number_input('Insert the total number of rooms', value=3)
latitude = st.number_input('Insert the approximate latitude', value=60.2276)
longitude = st.number_input('Insert the approximate longitude', value=24.7440)

price_pred = forest.predict([[size, year, rooms, latitude, longitude]])
st.write('Given the limited data that the model is trained on, the price for housing with the given characteristics is estimated as ', price_pred[0], "€.")


st.markdown("**Note**: _All models are wrong, some models are useful._")