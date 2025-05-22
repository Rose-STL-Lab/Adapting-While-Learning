import ast
import yaml
import torch
import numpy as np
import xarray as xr
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from .mfrnp import Emulator
from .utils import get_test_data
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_config_pth = os.path.join(base_dir, "model", "tas_reanalysis.yaml")
model_checkpoint_pth = os.path.join(base_dir, "model", "checkpoints", "tas_model.pt")
model_z_dict_pth = os.path.join(base_dir, "model", "checkpoints", "z_dict.pth")
settings = ["ssp126", "ssp245", "ssp370", "ssp585"]

emulator = Emulator(model_config_pth, model_checkpoint_pth, model_z_dict_pth)


def ll2index(longitude, latitude):
    index_0 = int((90 - latitude) * 4)
    index_1 = int((longitude + 180) * 1439 / 360)
    return index_0, index_1


def ll2index2(longitude, latitude):
    mapped_longitude = int((longitude + 180) / 360 * 144)

    mapped_latitude = int((latitude + 90) / 180 * 96)

    mapped_longitude = max(0, min(mapped_longitude, 143))
    mapped_latitude = max(0, min(mapped_latitude, 95))

    return mapped_latitude, mapped_longitude


def diy_greenhouse(longitude, latitude, setting, year, delta_CO2=0, delta_CH4=0):
    """
    Predict the temperature of a place in the future under a specific climate scenario with DIY change of CO2 and CH4 based on the original setting.

    Args:
        longitude: The longitude of the place you would check the temperature for, a float from -180 to 180.
        latitude: The latitude of the place you would check the temperature for, a float from -90 to 90.
        setting: Future climate scenarios, a string from ssp126, ssp245, ssp370, ssp585.
        year: The year you would check the temperature for, an integer from 2015 to 2100.
        delta_CO2: The change of CO2 you would like to make, a float. CO2_after = CO2_before * (1 + delta_CO2).
        delta_CH4: The change of CH4 you would like to make, a float. CH4_after = CH4_before * (1 + delta_CH4).
    """
    year = int(year)
    longitude = float(longitude)
    latitude = float(latitude)
    delta_CO2 = float(delta_CO2)
    delta_CH4 = float(delta_CH4)

    index_0, index_1 = ll2index(longitude, latitude)

    if year < 2015 or year > 2100:
        return "We only have future data from 2015 to 2100."

    x = np.load(os.path.join(base_dir, "data", "x_all.npy"))[
        165 + settings.index(setting) * 86 + (year - 2015)
    ].reshape(1, 12)

    x[0, 0] *= 1 + delta_CO2
    x[0, 1] *= 1 + delta_CH4

    y = emulator.pred(x).reshape(721, 1440)

    return y[index_0][index_1], f"The temperature is {y[index_0][index_1]}."


def diy_aerosol(longitude, latitude, setting, year, delta_SO2, delta_BC, modify_points):
    """
    Predict the temperature of a place in the future under a specific climate scenario with DIY change of CO2 and CH4 based on the original setting.

    Args:
        longitude: The longitude of the place you would check the temperature for, a float from -180 to 180.
        latitude: The latitude of the place you would check the temperature for, a float from -90 to 90.
        setting: Future climate scenarios, a string from ssp126, ssp245, ssp370, ssp585.
        year: The year you would check the temperature for, an integer from 2015 to 2100.
        delta_SO2: The change of SO2 you would like to make, a float.
        delta_BC: The change of BC you would like to make, a float.
        modification_method: The method to modify the grid, a string from solute, add, multiply. If solute, the grid will be set to the new value. If add, the grid will be added by the value. If multiply, the grid will be multiplied by 1 + value.
    """
    longitude = float(longitude)
    latitude = float(latitude)
    year = int(year)
    delta_SO2 = float(delta_SO2)
    delta_BC = float(delta_BC)

    if year < 2015 or year > 2100:
        return "We only have future data from 2015 to 2100."

    modification_method = "percent"

    modify_points = ast.literal_eval(modify_points)
    modify_points = [
        ll2index2(longitude, latitude) for longitude, latitude in modify_points
    ]

    index_0, index_1 = ll2index(longitude, latitude)

    x = get_test_data(
        year - 2015, setting, "SO2", delta_SO2, modification_method, modify_points
    )
    if delta_SO2 != 0:
        x = get_test_data(
            year - 2015, setting, "SO2", delta_SO2, modification_method, modify_points
        )
    if delta_BC != 0:
        x = get_test_data(
            year - 2015, setting, "BC", delta_BC, modification_method, modify_points
        )

    x = x.iloc[year - 2015].values.reshape(1, -1)
    y = emulator.pred(x).reshape(721, 1440)

    return y[index_0][index_1], f"The temperature is {y[index_0][index_1]}."


def diy_aerosol_mean(setting, year, delta_SO2, delta_BC, modify_points):
    """
    Predict the temperature of a place in the future under a specific climate scenario with DIY change of CO2 and CH4 based on the original setting.

    Args:
        longitude: The longitude of the place you would check the temperature for, a float from -180 to 180.
        latitude: The latitude of the place you would check the temperature for, a float from -90 to 90.
        setting: Future climate scenarios, a string from ssp126, ssp245, ssp370, ssp585.
        year: The year you would check the temperature for, an integer from 2015 to 2100.
        delta_SO2: The change of SO2 you would like to make, a float.
        delta_BC: The change of BC you would like to make, a float.
        modification_method: The method to modify the grid, a string from solute, add, multiply. If solute, the grid will be set to the new value. If add, the grid will be added by the value. If multiply, the grid will be multiplied by 1 + value.
    """
    year = int(year)
    delta_SO2 = float(delta_SO2)
    delta_BC = float(delta_BC)

    if year < 2015 or year > 2100:
        return "We only have future data from 2015 to 2100."

    if isinstance(modify_points, str):
        try:
            modify_points = ast.literal_eval(modify_points)
        except:
            return "The format of modify_points is not correct. You should pass a list of (lon, lat) to the function. It should be like [(0,0), (1,1), (2,2)]."
    if not isinstance(modify_points, list):
        return "The format of modify_points is not correct. You should pass a list of (lon, lat) to the function. It should be like [(0,0), (1,1), (2,2)]."
    if not isinstance(modify_points[0], tuple) and not isinstance(
        modify_points[0], list
    ):
        return "The format of modify_points is not correct. You should pass a list of (lon, lat) to the function. It should be like [(0,0), (1,1), (2,2)]."

    for point in modify_points:
        if point[0] < -180 or point[0] > 180 or point[1] < -90 or point[1] > 90:
            return "The range of longitude is from -180 to 180, and the range of latitude is from -90 to 90."

    modify_points = [
        ll2index2(longitude, latitude) for longitude, latitude in modify_points
    ]

    modification_method = "percent"

    x = get_test_data(
        year - 2015, setting, "SO2", delta_SO2, modification_method, modify_points
    )
    if delta_SO2 != 0:
        x = get_test_data(
            year - 2015, setting, "SO2", delta_SO2, modification_method, modify_points
        )
    if delta_BC != 0:
        x = get_test_data(
            year - 2015, setting, "BC", delta_BC, modification_method, modify_points
        )

    x = x.iloc[year - 2015].values.reshape(1, -1)
    y = emulator.pred(x).reshape(721, 1440)

    return f"The average temperature is {np.mean(y)}.", np.mean(y)


def diff_diy_aerosol_mean(setting, year, delta_SO2, delta_BC, modify_points):
    return (
        0,
        diy_aerosol_mean(setting, year, delta_SO2, delta_BC, modify_points)[1]
        - diy_aerosol_mean(setting, year, 0, 0, modify_points)[1],
    )


def diy_greenhouse_summary(longitude, latitude, delta_CO2=0, delta_CH4=0):
    index_0, index_1 = ll2index(longitude, latitude)

    CO2_change = (
        f"The emission of CO2 is {'increased' if delta_CO2 > 0 else 'decreased'} by {delta_CO2}%."
        if delta_CO2 != 0
        else ""
    )
    CH4_change = (
        f"The emission of CH4 is {'increased' if delta_CH4 > 0 else 'decreased'} by {delta_CH4}%."
        if delta_CH4 != 0
        else ""
    )

    return_string = f"\nFollowing is the temperature under different scenarios if: {CO2_change}{CH4_change}\n\n"

    for setting in settings:
        for year in [2050, 2100]:
            x = np.load(os.path.join(base_dir, "data", "x_all.npy"))[
                165 + settings.index(setting) * 86 + (year - 2015)
            ].reshape(1, 12)

            x[0, 0] *= 1 + delta_CO2
            x[0, 1] *= 1 + delta_CH4
            y = emulator.pred(x).reshape(721, 1440)

            return_string += f"the temperature in {year} under {setting} scenario is {y[index_0][index_1]},"

    return_string = return_string[:-1] + "."
    return return_string


if __name__ == "__main__":
    print(diy_greenhouse(-84.08, 9.9325, "ssp245", 2093, 0, -0.05))
    print(diy_greenhouse(-84.08, 9.9325, "ssp245", 2093, 0, -0.05))
    print(diy_greenhouse(-84.08, 9.9325, "ssp245", 2093, 0, -0.05))
    print(diy_greenhouse(-84.08, 9.9325, "ssp245", 2093, 0, -0.05))
