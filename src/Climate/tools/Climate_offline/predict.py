import numpy as np
import hashlib
import xarray as xr
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# Load the saved predictions
output_path = "./emulators/Climate_offline/data/predictions.npy"
predictions = np.load(output_path)
settings = ["ssp126", "ssp245", "ssp370", "ssp585"]


def ll2index(longitude, latitude):
    index_0 = int((90 - latitude) * 4)
    index_1 = int((longitude + 180) * 1439 / 360)
    return index_0, index_1


def stable_hash(number):
    number_str = str(number)

    hash_object = hashlib.sha256(number_str.encode())
    hash_hex = hash_object.hexdigest()

    return hash_hex


def location_summary(longitude, latitude):
    """
    Retrieve the temperature of a place in 1850, 1900, 1950, 2000, and predicted temperature under difference scenarios in 2050 and 2100.

    Args:
        longitude: The longitude of the place you would check the temperature for, a float from -180 to 180.
        latitude: The latitude of the place you would check the temperature for, a float from -90 to 90.
    """
    longitude = float(longitude)
    latitude = float(latitude)

    index_0, index_1 = ll2index(longitude, latitude)

    return (
        {
            "history": predictions[:, index_0, index_1].tolist(),
            "future": {
                setting: predictions[
                    165
                    + settings.index(setting) * 86 : 165
                    + settings.index(setting) * 86
                    + 86,
                    index_0,
                    index_1,
                ].tolist()
                for setting in settings
            },
        },
        f"Temperatur in 1850 is {predictions[0][index_0][index_1]}, temperatur in 1900 is {predictions[50][index_0][index_1]}, temperatur in 1950 is {predictions[100][index_0][index_1]}, temperatur in 2000 is {predictions[150][index_0][index_1]}.\nTemperatur in 2050 is {predictions[200][index_0][index_1]} under ssp126 scenario, {predictions[286][index_0][index_1]} under ssp245 scenario, {predictions[372][index_0][index_1]} under ssp370 scenario, {predictions[458][index_0][index_1]} under ssp585 scenario. Temperatur in 2100 is {predictions[250][index_0][index_1]} under ssp126 scenario, {predictions[336][index_0][index_1]} under ssp245 scenario, {predictions[422][index_0][index_1]} under ssp370 scenario, {predictions[508][index_0][index_1]} under ssp585 scenario.",
    )


def history_temperature(longitude, latitude, year):
    """
    Retrieve the temperature of a place from 1850 to 2014 with longitude and latitude.

    Args:
        longitude: The longitude of the place you would check the temperature for, a float from -180 to 180.
        latitude: The latitude of the place you would check the temperature for, a float from -90 to 90.
        year: The year you would check the temperature for, an integer from 1850 to 2014.
    """

    longitude = float(longitude)
    latitude = float(latitude)
    year = int(year)

    if year < 1850 or year > 2014:
        return "We only have history data from 1850 to 2014."

    index_0, index_1 = ll2index(longitude, latitude)

    return (
        predictions[year - 1850][index_0][index_1],
        f"The temperature is {predictions[year-1850][index_0][index_1]}.",
    )


def future_temperature(longitude, latitude, setting, year):
    """
    Retrieve the temperature of a place from 2015 to 2100 under different climate scenarios with longitude and latitude.

    Args:
        longitude: The longitude of the place you would check the temperature for, a float from -180 to 180.
        latitude: The latitude of the place you would check the temperature for, a float from -90 to 90.
        year: The year you would check the temperature for, an integer from 2015 to 2100.
        setting: Future climate scenarios, a string from ssp126, ssp245, ssp370, ssp585.
    """

    longitude = float(longitude)
    latitude = float(latitude)
    year = int(year)

    if year < 2015 or year > 2100:
        return "We only have future data from 2015 to 2100."

    index_0, index_1 = ll2index(longitude, latitude)

    return (
        predictions[165 + settings.index(setting) * 86 + (year - 2015)][index_0][
            index_1
        ],
        f"The temperature is {predictions[165 + settings.index(setting) * 86 + (year - 2015)][index_0][index_1]}.",
    )


def history_image(year, min_lon, max_lon, min_lat, max_lat, coastline, border):
    if max_lat - min_lat < 30 or max_lon - min_lon < 30:
        return "Both max_lat - min_lat and max_lon - min_lon should be bigger than 30."

    if year < 1850 or year > 2014:
        return "We only have history data from 1850 to 2014."

    return visualize(
        predictions[year - 1850], min_lon, max_lon, min_lat, max_lat, coastline, border
    )


def future_image(setting, year, min_lon, max_lon, min_lat, max_lat, coastline, border):
    if max_lat - min_lat < 30 or max_lon - min_lon < 30:
        return "Both max_lat - min_lat and max_lon - min_lon should be bigger than 30."

    if year < 2015 or year > 2100:
        return "We only have future data from 2015 to 2100."

    return visualize(
        predictions[165 + settings.index(setting) * 86 + (year - 2015)],
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        coastline,
        border,
    )

if __name__ == "__main__":
    print(ll2index(-172.3, -13.7))
