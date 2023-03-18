from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Create a class to gather all the methods for data cleaning
class DataCleaner:
    def __init__(self, data):
        self.data = data

    def add_house_type_and_description(self):
        """
        Method to add a column with the type of house and a column with the description
        using the Title column
        """
        self.data["House_type", "Description"] = self.data.Title.str.spilt("|", expand=True)
        self.data.drop("Title", axis=1, inplace=True)

    def clean_house_type(self):
        """
        Method to clean the House_type column
        """
        self.data.House_type = self.data.House_type.str.strip()

    def add_rooms_from_description(self):
        """
        Method to add the number of rooms from the Description column
        """
        # the '-' is for those where two numbers are specified (e.g. '3-4')
        self.data["Total_rooms"] = self.data.Description.str.split('h|H|-|k|mh|\(|,|x|\+', expand=True)[0]  
        # remove characters for type of house
        self.data.Total_rooms = self.data.Total_rooms.str.replace(r'KT|RT|PT', '') 
        # remove empty spaces
        self.data.Total_rooms = self.data.Total_rooms.str.strip()
        self.data.Total_rooms = self.data.Total_rooms.str.extract('(\d+)', expand=False)

    def add_location_from_address(self):
        """
        Method to add the location of the house from the Address column
        """
        geolocator = Nominatim(user_agent="adfgasdf")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        self.data['location'] = self.data['name'].apply(geocode)
        self.data['point'] = self.data['location'].apply(lambda loc: tuple(loc.point) if loc else None)

    def add_latitude_andlongitude(self):
        """
        Method to add the latitude and longitude of the house
        """
        self.data["Latitude", "Longitude", "Altitude"] = self.data.point.tolist()