import cartopy.feature as cfeature
from shapely.geometry import Point
from geopy.geocoders import Nominatim
import pandas as pd

# Load the CSV file into a DataFrame
lat_and_lon = pd.read_csv("./emulators/Climate_offline/data/worldcities.csv")


def query_lat_and_lon(city_name):
    """
    Retrieve the latitude and longitude of a place with the name.

    Args:
        city_name: The name of the place you would check the latitude and longitude
    """
    # Convert city_name to lowercase
    city_name_lower = city_name.lower()

    # Query the DataFrame for the city, converting the 'city' column to lowercase
    city = lat_and_lon[lat_and_lon["city"].str.lower() == city_name_lower]

    if not city.empty:
        lat = city.iloc[0]["lat"]
        lng = city.iloc[0]["lng"]
        return 0, f"latitude: {lat}, longitude: {lng}."
    else:
        return 0, "City not found."


def is_land_or_sea(lat, lon):
    point = Point(lon, lat)

    land = cfeature.NaturalEarthFeature("physical", "land", "110m")

    for geom in land.geometries():
        if geom.contains(point):
            return f"lat: {lat}, lon: {lon} is on land.", 1

    return f"lat: {lat}, lon: {lon} is on sea.", 0


def get_city_coordinates(city_name):
    geolocator = Nominatim(user_agent="geoapiExercises")

    location = geolocator.geocode(city_name)

    if location:
        return {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "bounding_box": location.raw.get("boundingbox", None),
        }
    else:
        return None


if __name__ == "__main__":
    print(query_lat_and_lon("London"))
    print(is_land_or_sea(51.5074, 0.1278))
    print(get_city_coordinates("London"))
