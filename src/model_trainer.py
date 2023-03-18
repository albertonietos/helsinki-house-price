import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestRegressor()

    def load_data(self):
        data = pd.read_excel(self.data_path, engine='xlrd')
        data = data[data.Total_rooms.notna() & data.Latitude.notna()]
        self.X = data[['Size', 'Year', 'Total_rooms', 'Latitude', 'Longitude']]
        self.y = data['Price']

    def train_model(self):
        self.model.fit(self.X.values, self.y.values)

    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

if __name__ == '__main__':
    # example usage
    model = ModelTrainer('./data/cleaned/helsinki_house_price_cleaned.xls')
    model.load_data()
    model.train_model()
    model.save_model('./models/model.pkl')
