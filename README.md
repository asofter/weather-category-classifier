# Simple weather category classifier
Simple weather category (perfect, good, bad) classifier writter in Python. It can be used for travel-related website to recommend if the weather is OK for tourist sightseeing.

## Contains

 - Dataset generator. It takes data from TravelBriefing.org (country, average weather for country), RestCountries.eu (capital city of country) and Darsky.net (current weather in the capital of the country).
 - Classifier trainer. In this file we work with all the data to find the best classification algorithm and train the model.
 - Classifier demo. Demo file for the trained algorithm.
 
## Data

To find category of the weather (3 - perfect, 2 - good, 1 - bad) you need to provide: 

 - Average temperature for the country in this month.
 - Current temperature in the city.
 - Preciption probability.
 - Bool (1 or 0) if has sleet or rain preciption.
 - Preciption intensity

## Installation
```
pip install -r requirements.txt
```