import pickle
import numpy as np

class HousePricePredictor:
    def __init__(self, model_path):
        # load the trained model from file
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
    def predict(self, size, year, total_rooms, latitude, longitude):
        # create a 1x5 numpy array for the input features
        x = np.array([[size, year, total_rooms, latitude, longitude]])
        
        # use the model to predict the price
        y_pred = self.model.predict(x)
        
        # return the predicted price
        return y_pred[0]
