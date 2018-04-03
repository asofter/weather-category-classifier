from darksky import forecast
from geopy.geocoders import Nominatim
from datetime import datetime
import urllib.request, urllib.parse, json 
import pickle
from pathlib import Path
import csv as csv

DARSKY_KEY = ""

def saveVars(fileName, var):
    with open(fileName, 'wb') as output:
        pickle.dump(var, output, pickle.HIGHEST_PROTOCOL)

def getSavedVars(fileName):
    with open(fileName, 'rb') as input:
        return pickle.load(input)

class DatasetInfoGenerator(object):
    TRAVEL_BRIEFING_API_URL = 'https://travelbriefing.org/countries.json'
    REST_COUNTRIES_API_URL = 'https://restcountries.eu/rest/v2/name/{}'
    COUNTRIES_FILE = 'cache/countries.pkl'
    COUNTRY_CAPITALS_FILE = 'cache/country_capitals.pkl'
    COUNTRY_WEATHER_FILE = 'cache/country_weather.pkl'

    def __init__(self):
        self.countries = []
        self.capitals = []
        self.weather = []
        
        self.loadAll()
        
    def loadAll(self):
        self.__getTravelBriefingInfo()
        self.__getRestCountriesInfo()
        
    def getAllData(self):
        return self.countries, self.capitals, self.weather
        
    def __getRestCountriesInfo(self):
        my_file = Path(self.COUNTRY_CAPITALS_FILE)
        if my_file.is_file():
            self.capitals = getSavedVars(self.COUNTRY_CAPITALS_FILE)
        else:
            capitals = []
            for country in self.countries:
                try:
                    with urllib.request.urlopen(self.REST_COUNTRIES_API_URL.format(urllib.parse.quote(country))) as url:
                        data = json.loads(url.read().decode())
                        capitals.append(data[0]['capital'])
                except urllib.error.HTTPError as err:
                    capitals.append("")
                
            self.capitals = capitals
            saveVars(self.COUNTRY_CAPITALS_FILE, capitals)
      
    def __getTravelBriefingInfo(self):
        my_file1 = Path(self.COUNTRIES_FILE)
        my_file2 = Path(self.COUNTRY_WEATHER_FILE)
        if my_file1.is_file() and my_file2.is_file():
            self.countries = getSavedVars(self.COUNTRIES_FILE)
            self.weather = getSavedVars(self.COUNTRY_WEATHER_FILE)
        else:
            countries = []
            weather = []
            
            with urllib.request.urlopen(self.TRAVEL_BRIEFING_API_URL) as url:
                data = json.loads(url.read().decode())
                for item in data:
                    countries.append(item['name'])
                    
                    with urllib.request.urlopen(item['url']) as url:
                        city_data = json.loads(url.read().decode())
                        city_weather = [[month, city_data['weather'][month]['tAvg']] for month in city_data['weather']]
                        weather.append(city_weather)
           
            self.countries = countries
            self.weather = weather
            saveVars(self.COUNTRIES_FILE, countries)
            saveVars(self.COUNTRY_WEATHER_FILE, weather)

dg = DatasetInfoGenerator()
countries,capitals,weather = dg.getAllData()

def getCurrentWeather(lat, lng):
    return forecast(DARSKY_KEY, lat, lng, units='si', exclude='minutely,hourly,daily,alerts,flags')

def loadWeather():
    geolocator = Nominatim()
    result = []
    for idx, country in enumerate(countries):
        location = geolocator.geocode("{}, {}".format(capitals[idx], country))
        if hasattr(location, 'latitude') == False:
            result.append(0)
            continue
        
        weather = getCurrentWeather(location.latitude, location.longitude)
        
        day = weather.currently
        dt = datetime.fromtimestamp(day.time)
        day = dict(date = dt.strftime('%d.%m.%Y'),
                   month = dt.strftime('%B'),
                   temperature = day.temperature,
                   precipIntensity = day.precipIntensity,
                   precipProbability = day.precipProbability,
                   hasBadPrecip = hasattr(day, 'precipType') and (day.precipType == 'rain' or day.precipType == 'sleet')
                   )
        result.append(day)
    return result
            
def getWeather():
    file_name = 'cache/weather_forecast.pkl'
    my_file = Path(file_name)
    if my_file.is_file():
        return getSavedVars(file_name)
    else:
        weather = loadWeather()
        saveVars(file_name, weather)
        return weather

def getAvgMonthWeather(monthName, monthWeather):
    for month, avgTemp in monthWeather:
        if month == monthName:
            return round(float(avgTemp), 2)
            break
    return 0.0

fcst = getWeather()

def collectToDataset():
    with open('dataset.csv', 'w') as csvfile:
        fieldnames = ['country', 'date', 'tempAvgMonth', 'temp', 'precipProb', 'precipIntens', 'hasBadPrecip', 'rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
     
        writer.writeheader()
        for idx, country in enumerate(countries):
            try:
                if type(fcst[idx]) is dict:
                    current_forecast = fcst[idx]
                else: 
                    continue
            except IndexError:
                continue
            
            row = dict(country = country)
            
            row['date'] = current_forecast['date']
            row['temp'] = round(float(current_forecast['temperature']), 2)
            row['precipIntens'] = float(current_forecast['precipIntensity'])
            row['precipProb'] = float(current_forecast['precipProbability'])
            row['hasBadPrecip'] = int(current_forecast['hasBadPrecip'])
            
            row['tempAvgMonth'] = getAvgMonthWeather(current_forecast['month'], weather[idx])
                    
            writer.writerow(row)
            
# collectToDataset()
    
    