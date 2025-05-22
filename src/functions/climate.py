functions_climate = {
    "location_summary": {
        "name": "location_summary",
        "description": "Retrieve the temperature of a place in 1850, 1900, 1950, 2000, and predicted temperature under difference scenarios in 2050 and 2100.",
        "parameters": {
            "type": "object",
            "properties": {
                "longitude": {
                    "type": "number",
                    "description": "The longitude of the place you would check the temperature for, a float from -180 to 180.",
                },
                "latitude": {
                    "type": "number",
                    "description": "The latitude of the place you would check the temperature for, a float from -90 to 90.",
                },
            },
            "required": ["longitude", "latitude"],
        },
    },
    "history_temperature": {
        "name": "history_temperature",
        "description": "Retrieve the temperature of a place from 1850 to 2014 with longitude and latitude.",
        "parameters": {
            "type": "object",
            "properties": {
                "longitude": {
                    "type": "number",
                    "description": "The longitude of the place you would check the temperature for, a float from -180 to 180.",
                },
                "latitude": {
                    "type": "number",
                    "description": "The latitude of the place you would check the temperature for, a float from -90 to 90.",
                },
                "year": {
                    "type": "number",
                    "description": "The year you would check the temperature for, an integer from 1850 to 2014.",
                },
            },
            "required": ["longitude", "latitude", "year"],
        },
    },
    "future_temperature": {
        "name": "future_temperature",
        "description": "Retrieve the temperature of a place from 2015 to 2100 under different climate scenarios with longitude and latitude.",
        "parameters": {
            "type": "object",
            "properties": {
                "longitude": {
                    "type": "number",
                    "description": "The longitude of the place you would check the temperature for, a float from -180 to 180.",
                },
                "latitude": {
                    "type": "number",
                    "description": "The latitude of the place you would check the temperature for, a float from -90 to 90.",
                },
                "year": {
                    "type": "number",
                    "description": "The year you would check the temperature for, an integer from 2015 to 2100.",
                },
                "setting": {
                    "type": "string",
                    "enum": ["ssp126", "ssp245", "ssp370", "ssp585"],
                    "description": "Future climate scenarios, a string from ssp126, ssp245, ssp370, ssp585.",
                },
            },
            "required": ["longitude", "latitude", "year", "setting"],
        },
    },
    # "history_image": {
    #     "name": "history_image",
    #     "description": "Generate and save an image of the temperature of a place from 1850 to 2014 with the interval of longitude and latitude, whose min and max should have a gap of at least 30.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "min_lon": {
    #                 "type": "number",
    #                 "description": "The minimum longitude of the place you would check the temperature for, a float from -180 to 180.",
    #             },
    #             "max_lon": {
    #                 "type": "number",
    #                 "description": "The maximum longitude of the place you would check the temperature for, a float from -180 to 180.",
    #             },
    #             "min_lat": {
    #                 "type": "number",
    #                 "description": "The minimum latitude of the place you would check the temperature for, a float from -90 to 90.",
    #             },
    #             "max_lat": {
    #                 "type": "number",
    #                 "description": "The maximum latitude of the place you would check the temperature for, a float from -90 to 90.",
    #             },
    #             "year": {
    #                 "type": "number",
    #                 "description": "The year you would check the temperature for, an integer from 1850 to 2014.",
    #             },
    #             "coastline": {
    #                 "type": "boolean",
    #                 "description": "Whether to show the coastline, a boolean.",
    #             },
    #             "border": {
    #                 "type": "boolean",
    #                 "description": "Whether to show the border, a boolean.",
    #             },
    #         },
    #         "required": [
    #             "min_lon",
    #             "max_lon",
    #             "min_lat",
    #             "max_lat",
    #             "year",
    #             "coastline",
    #             "border",
    #         ],
    #     },
    # },
    # "future_image": {
    #     "name": "future_image",
    #     "description": "Generate and save an image of the temperature of a place from 2015 to 2100 under different climate scenarios with the interval of longitude and latitude, whose min and max should have a gap of at least 30.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "min_lon": {
    #                 "type": "number",
    #                 "description": "The minimum longitude of the place you would check the temperature for, a float from -180 to 180.",
    #             },
    #             "max_lon": {
    #                 "type": "number",
    #                 "description": "The maximum longitude of the place you would check the temperature for, a float from -180 to 180.",
    #             },
    #             "min_lat": {
    #                 "type": "number",
    #                 "description": "The minimum latitude of the place you would check the temperature for, a float from -90 to 90.",
    #             },
    #             "max_lat": {
    #                 "type": "number",
    #                 "description": "The maximum latitude of the place you would check the temperature for, a float from -90 to 90.",
    #             },
    #             "year": {
    #                 "type": "number",
    #                 "description": "The year you would check the temperature for, an integer from 2015 to 2100.",
    #             },
    #             "setting": {
    #                 "type": "string",
    #                 "enum": ["ssp126", "ssp245", "ssp370", "ssp585"],
    #                 "description": "Future climate scenarios, a string from ssp126, ssp245, ssp370, ssp585.",
    #             },
    #             "coastline": {
    #                 "type": "boolean",
    #                 "description": "Whether to show the coastline, a boolean.",
    #             },
    #             "border": {
    #                 "type": "boolean",
    #                 "description": "Whether to show the border, a boolean.",
    #             },
    #         },
    #         "required": [
    #             "min_lon",
    #             "max_lon",
    #             "min_lat",
    #             "max_lat",
    #             "year",
    #             "setting",
    #             "coastline",
    #             "border",
    #         ],
    #     },
    # },
    "query_lat_and_lon": {
        "name": "query_lat_and_lon",
        "description": "Retrieve the latitude and longitude of a place with the name.",
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": {
                    "type": "string",
                    "description": "The name of the place you would check the latitude and longitude for, a string.",
                },
            },
            "required": ["city_name"],
        },
    },
    "diy_greenhouse": {
        "name": "diy_greenhouse",
        "description": "Predict the temperature of a place in the future under a specific climate scenario with DIY change of CO2 and CH4 based on the original setting.",
        "parameters": {
            "type": "object",
            "properties": {
                "longitude": {
                    "type": "number",
                    "description": "The longitude of the place you would check the temperature for, a float from -180 to 180.",
                },
                "latitude": {
                    "type": "number",
                    "description": "The latitude of the place you would check the temperature for, a float from -90 to 90.",
                },
                "setting": {
                    "type": "string",
                    "enum": ["ssp126", "ssp245", "ssp370", "ssp585"],
                    "description": "Future climate scenarios, a string from ssp126, ssp245, ssp370, ssp585.",
                },
                "year": {
                    "type": "number",
                    "description": "The year you would check the temperature for, an integer from 2015 to 2100.",
                },
                "delta_CO2": {
                    "type": "number",
                    "description": "The change of CO2 you would like to make, a float. CO2_after = CO2_before * (1 + delta_CO2).",
                },
                "delta_CH4": {
                    "type": "number",
                    "description": "The change of CH4 you would like to make, a float. CH4_after = CH4_before * (1 + delta_CH4).",
                },
            },
            "required": [
                "longitude",
                "latitude",
                "setting",
                "year",
                "delta_CO2",
                "delta_CH4",
            ],
        },
    },
    "diy_aerosol": {
        "name": "diy_aerosol",
        "description": "Predict the temperature of a place in the future under a specific climate scenario with DIY change of SO2 and BC based on the original setting.",
        "parameters": {
            "type": "object",
            "properties": {
                "longitude": {
                    "type": "number",
                    "description": "The longitude of the place you would check the temperature for, a float from -180 to 180.",
                },
                "latitude": {
                    "type": "number",
                    "description": "The latitude of the place you would check the temperature for, a float from -90 to 90.",
                },
                "setting": {
                    "type": "string",
                    "enum": ["ssp126", "ssp245", "ssp370", "ssp585"],
                    "description": "Future climate scenarios, a string from ssp126, ssp245, ssp370, ssp585.",
                },
                "year": {
                    "type": "number",
                    "description": "The year you would check the temperature for, an integer from 2015 to 2100.",
                },
                "delta_SO2": {
                    "type": "number",
                    "description": "The change of SO2 you would like to make, a float.",
                },
                "delta_BC": {
                    "type": "number",
                    "description": "The change of BC you would like to make, a float.",
                },
                "modify_points": {
                    "type": "number",
                    "description": "The points along with the line or curve to modify the grid, a pair or pairs of longitude and latitude. If only one pair, it should be a string formatted as [(longitude, latitude)]. If multiple pairs, it should be a string formatted as [(longitude1, latitude1, (longitude2, latitude2), ...]. Longitude is a number in the range of -180 to 180, and latitude is a number in the range of -90 to 90.",
                },
            },
            "required": [
                "longitude",
                "latitude",
                "setting",
                "year",
                "delta_SO2",
                "delta_BC",
                "modification_method",
                "modify_points",
            ],
        },
    },
    "land_or_sea": {
        "name": "land_or_sea",
        "description": "Determine whether a place is on land or sea with latitude and longitude.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "The latitude of the place you would check, a float from -90 to 90.",
                },
                "longitude": {
                    "type": "number",
                    "description": "The longitude of the place you would check, a float from -180 to 180.",
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
    "diy_aerosol_mean": {
        "name": "diy_aerosol_mean",
        "description": "Predict the average temperature of the world in the future under a specific climate scenario with DIY change of SO2 and BC based on the original setting.",
        "parameters": {
            "type": "object",
            "properties": {
                "setting": {
                    "type": "string",
                    "enum": ["ssp126", "ssp245", "ssp370", "ssp585"],
                    "description": "Future climate scenarios, a string from ssp126, ssp245, ssp370, ssp585.",
                },
                "year": {
                    "type": "number",
                    "description": "The year you would check the temperature for, an integer from 2015 to 2100.",
                },
                "delta_SO2": {
                    "type": "number",
                    "description": "The change of SO2 you would like to make, a float. SO2_after = SO2_before * (1 + delta_SO2).",
                },
                "delta_BC": {
                    "type": "number",
                    "description": "The change of BC you would like to make, a float. BC_after = BC_before * (1 + delta_BC).",
                },
                "modify_points": {
                    "type": "number",
                    "description": "Points along the line or curve to modify the grid, specified as longitude-latitude pairs. For a single point, use the format '[(longitude, latitude)]'. For multiple points, use '[(longitude1, latitude1), (longitude2, latitude2), ...]'. Longitude ranges from -180 to 180, and latitude from -90 to 90.",
                },
            },
            "required": [
                "setting",
                "year",
                "delta_SO2",
                "delta_BC",
                "modification_method",
                "modify_points",
            ],
        },
    },
    "is_land_or_sea": {
        "name": "is_land_or_sea",
        "description": "Query whether a place is on land or sea with latitude and longitude.",
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "The latitude of the place you would check, a float from -90 to 90.",
                },
                "lon": {
                    "type": "number",
                    "description": "The longitude of the place you would check, a float from -180 to 180.",
                },
            },
            "required": ["lat", "lon"],
        },
    },
}
