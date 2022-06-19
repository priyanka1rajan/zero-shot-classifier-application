
import requests
from dataclasses import dataclass

@dataclass
class WeatherService:
    """
    This data class provides weather reports.
    """
    
    #default city
    city_name: str = "Cupertino"
        
    # api credentials from Openweathermap.org
    api_key: str = "3f983668759a16b5c652a0e5ab2b429e"
    base_url: str = "http://api.openweathermap.org/data/2.5/weather?"
    
    def get_weather_report(self):
            
        current_temperature = None; current_humidity=None; weather_description=None;

        url = self.base_url + "appid=" + self.api_key + "&q=" + self.city_name

        response = requests.get(url)

        x = response.json()

        #  x contains list of nested dictionaries
        if x["cod"] == 401:
            print("API Key not activiated yet...")

        elif x["cod"] == 404:
            print(" City Not Found ")

        else:
            # store the value of "main" key in variable y
            y = x["main"]

            current_temperature = round(y["temp"] -273.15,1) # celsius conversion

            current_pressure = round(y["pressure"],1)

            current_humidity = round(y["humidity"],1)

            # store the value of "weather" key in variable z
            z = x["weather"]

            weather_description = z[0]["description"]

        return current_temperature, current_humidity, weather_description

