from sklearn.externals import joblib
import numpy as np
from dataset_generator import DatasetInfoGenerator, DARSKY_KEY, getCurrentWeather, getAvgMonthWeather
from geopy.geocoders import Nominatim
from darksky import forecast
from datetime import datetime

def getCurrentWeatherCategory(city, country):
    # Load list of countries, capitals and weather in countries
    dg = DatasetInfoGenerator()
    countries,capitals,month_weather = dg.getAllData()
    countryIdx = countries.index(country)
    
    # Find lat and lng for the city
    geolocator = Nominatim()
    location = geolocator.geocode("{}, {}".format(city, country))
    if hasattr(location, 'latitude') == False:
        return False
    
    # Get current weather using Darsky API    
    weather = getCurrentWeather(location.latitude, location.longitude)  
    
    day = weather.currently
    hasBadPrecip = hasattr(day, 'precipType') and (day.precipType == 'rain' or day.precipType == 'sleet')
    dt = datetime.fromtimestamp(day.time)
    
    avg_weather_month = getAvgMonthWeather(dt.strftime('%B'), month_weather[countryIdx])
    
    model = joblib.load('cache/trained_classifier.pkl')
    pred = model.predict(np.array([[avg_weather_month, day.temperature, day.precipProbability, day.precipIntensity, hasBadPrecip]], dtype=np.float64))
    
    return pred[0]    
    
result = getCurrentWeatherCategory('Oslo', 'Norway')

if result == 3:
    print("Perfect weather for tourist")
elif result == 2:
    print("Good weather for tourist")
elif result == 1:
    print("Bad weather for tourist")
else:
    print("Error")