# from functions.functions import *
import pandas as pd
import argparse
import random
import json
import numpy as np

import sys
import os

# sys.path.append(os.path.join(os.path.dirname(__file__), 'tools/Climate_online'))
# from tools.emulators import *

questions = []

climate_settings = ["ssp126", "ssp245", "ssp370", "ssp585"]


def build(city):
    pass


sea_routes = [
    ("Shanghai", (31.2286, 121.4747), "Los Angeles", (34.1141, -118.4068)),
    ("Rotterdam", (51.92, 4.48), "New York", (40.6943, -73.9249)),
    ("Singapore", (1.3, 103.8), "Dubai", (25.2631, 55.2972)),
    ("Hong Kong", (22.3, 114.2), "Hamburg", (53.55, 10.0)),
    ("Busan", (35.18, 129.075), "Long Beach", (33.7977, -118.167)),
    ("Ningbo", (29.8603, 121.6245), "Antwerp", (51.2178, 4.4003)),
    ("Tokyo", (35.6897, 139.6922), "Seattle", (47.6211, -122.3244)),
    ("Kaohsiung", (22.615, 120.2975), "Sydney", (-33.8678, 151.21)),
    ("Shenzhen", (22.5415, 114.0596), "London", (51.5072, -0.1275)),
    ("Guangzhou", (23.13, 113.26), "Marseille", (43.2964, 5.37)),
    ("Qingdao", (36.0669, 120.3827), "Vancouver", (49.25, -123.1)),
    ("Tianjin", (39.1336, 117.2054), "Auckland", (-36.8406, 174.74)),
    ("Jakarta", (-6.175, 106.8275), "Mumbai", (19.0761, 72.8775)),
    ("Manila", (14.5958, 120.9772), "Melbourne", (-37.8142, 144.9631)),
    ("Ho Chi Minh City", (10.7756, 106.7019), "Barcelona", (41.3828, 2.1769)),
    ("Colombo", (6.9344, 79.8428), "Rotterdam", (51.92, 4.48)),
    ("Frankfurt", (50.1106, 8.6822), "New York", (40.6943, -73.9249)),
    ("Rio de Janeiro", (-22.9111, -43.2056), "Cape Town", (-33.9253, 18.4239)),
    ("St. Petersburg", (27.7931, -82.6652), "Istanbul", (41.0136, 28.955)),
    ("Mombasa", (-4.05, 39.6667), "Karachi", (24.86, 67.01)),
    ("Jeddah", (21.5433, 39.1728), "Singapore", (1.3, 103.8)),
    ("Alexandria", (31.1975, 29.8925), "Genoa", (44.4111, 8.9328)),
    ("Dakar", (14.6928, -17.4467), "Casablanca", (33.5333, -7.5833)),
    ("Amsterdam", (52.3728, 4.8936), "San Francisco", (37.7558, -122.4449)),
    ("Bangkok", (13.7525, 100.4942), "Yokohama", (35.4442, 139.6381)),
    ("Panama City", (8.9833, -79.5167), "Miami", (25.784, -80.2101)),
    ("Lisbon", (38.7253, -9.15), "Houston", (29.786, -95.3885)),
    ("Tallinn", (59.4372, 24.7453), "Helsinki", (60.1708, 24.9375)),
    ("Oslo", (59.9133, 10.7389), "Copenhagen", (55.6761, 12.5683)),
]

air_routes = [
    ("Beijing", (39.9067, 116.3975), "New York", (40.6943, -73.9249)),
    ("Tokyo", (35.6897, 139.6922), "Los Angeles", (34.1141, -118.4068)),
    ("Seoul", (37.56, 126.99), "Mexico City", (19.4333, -99.1333)),
    ("Mumbai", (19.0761, 72.8775), "Riyadh", (24.6333, 46.7167)),
    ("Moscow", (55.7558, 37.6172), "Istanbul", (41.0136, 28.955)),
    ("Sydney", (-33.8678, 151.21), "Jakarta", (-6.175, 106.8275)),
    ("London", (51.5072, -0.1275), "Cairo", (30.0444, 31.2358)),
    ("Berlin", (52.52, 13.405), "Johannesburg", (-26.2044, 28.0456)),
    ("Paris", (48.8567, 2.3522), "Casablanca", (33.5333, -7.5833)),
    ("Rome", (41.8933, 12.4828), "Athens", (37.9842, 23.7281)),
    ("Madrid", (40.4169, -3.7033), "Buenos Aires", (-34.6033, -58.3817)),
    ("Amsterdam", (52.3728, 4.8936), "Brussels", (50.8467, 4.3525)),
    ("Kuala Lumpur", (3.1478, 101.6953), "Hanoi", (21.0, 105.85)),
    ("Bangkok", (13.7525, 100.4942), "Manila", (14.5958, 120.9772)),
    ("Auckland", (-36.8406, 174.74), "Suva", (-18.1416, 178.4419)),
    ("Santiago", (-33.4372, -70.6506), "Lima", (-12.06, -77.0375)),
    ("Bogota", (40.8751, -74.0293), "Quito", (-0.22, -78.5125)),
    ("Caracas", (10.4806, -66.9036), "Port of Spain", (10.6667, -61.5167)),
    ("Accra", (5.55, -0.2), "Abidjan", (5.3167, -4.0333)),
    ("Nairobi", (-1.2864, 36.8172), "Dar es Salaam", (-6.8161, 39.2803)),
    ("Muscat", (23.6139, 58.5922), "Islamabad", (33.6931, 73.0639)),
    ("Doha", (25.2867, 51.5333), "Kuwait City", (29.3697, 47.9783)),
    ("Colombo", (6.9344, 79.8428), "Male", (4.1753, 73.5089)),
    ("Lisbon", (38.7253, -9.15), "Dublin", (53.35, -6.2603)),
    ("Stockholm", (59.3294, 18.0686), "Helsinki", (60.1708, 24.9375)),
    ("Copenhagen", (55.6761, 12.5683), "Oslo", (59.9133, 10.7389)),
    ("Warsaw", (52.23, 21.0111), "Vilnius", (54.6872, 25.28)),
    ("Riga", (56.9489, 24.1064), "Tallinn", (59.4372, 24.7453)),
    ("Nassau", (25.0781, -77.3386), "Kingston", (17.9714, -76.7931)),
]

sea_routes = [
    (
        f"{i[0]} (lon: {i[1][1]}, lat: {i[1][0]})",
        f"{i[2]} (lon: {i[3][1]}, lat: {i[3][0]})",
    )
    for i in sea_routes
]

air_routes = [
    (
        f"{i[0]} (lon: {i[1][1]}, lat: {i[1][0]})",
        f"{i[2]} (lon: {i[3][1]}, lat: {i[3][0]})",
    )
    for i in air_routes
]

questions = []

for gas in ["SO2", "BC"]:
    for setting in climate_settings:
        for increase in [0.03, 0.05, 0.1, 0.2]:
            for city1, city2 in sea_routes:
                questions.append(
                    f"Design a new maritime route between {city1} and {city2} that would increase {gas} levels along the route by {increase * 100}%. Propose a route that would minimize the global average temperature increase in {random.randint(2020, 2099)} under {setting}. Present your answer as a list of coordinates (longitude, latitude) representing key points along the route. Format your response as follows: [(longitude_1, latitude_1), (longitude_2, latitude_2), ..., (longitude_n, latitude_n)]. Include at least the starting point, endpoint, and any significant waypoints. Ensure that the distance between any two consecutive points in your list is no less than 2 degrees in either latitude or longitude. Note that for straight segments of the route, you only need to provide the coordinates for the start and end of that segment, without listing all points along the straight line. The route will be automatically connected based on the nodes you provide."
                )

# for gas in ["SO2", "BC"]:
#     for setting in climate_settings:
#         for increase in [0.03, 0.05, 0.1, 0.2]:
#             for city1, city2 in sea_routes:
#                 questions.append(
#                     f"Choose an airline transfer station between {city1} and {city2} that would increase {gas} levels along the route by {increase * 100}%. Propose a route that would minimize the global average temperature increase in {random.randint(2020, 2099)} under {setting}. Present your answer as a list of coordinates (longitude, latitude) representing key points along the route. Format your response as follows: [(longitude_1, latitude_1), (longitude_2, latitude_2), ..., (longitude_n, latitude_n)]. Include at least the starting point, endpoint, and any significant waypoints. Ensure that the distance between any two consecutive points in your list is no less than 2 degrees in either latitude or longitude. Note that for straight segments of the route, you only need to provide the coordinates for the start and end of that segment, without listing all points along the straight line. The route will be automatically connected based on the nodes you provide."
#                 )

data = [{"question": q} for q in questions]

if __name__ == "__main__":
    random.shuffle(questions)
    print(len(questions))
    with open("", "w") as f:
        json.dump(data, f, indent=4)
