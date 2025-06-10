import sys
import os

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(src_dir)

from utils.utils import *
from functions.functions import *
from utils.make_problem_utils import *
import pandas as pd
import argparse
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "tools/Climate_online"))
from tools.emulators import *

questions = []

climate_settings = ["ssp126", "ssp245", "ssp370", "ssp585"]


def generate_question(cities, country=None):
    global questions
    if country:
        history_indexs = random.sample(range(0, 165), 3)
        for history_index in history_indexs:
            temperatures = [
                history_temperature(city[0], city[1], 1850 + history_index)
                for city in cities
            ]
            temperatures = [i[0] for i in temperatures]
            hottest_city = cities[temperatures.index(max(temperatures))]
            options, correct_option = generate_simple_question(cities, hottest_city)
            questions.append(
                {
                    "Question": f"For {', '.join([city[-1] for city in cities])}, which city has the highest temperature in {1850+history_index}?",
                    "Options": options,
                    "Correct": correct_option,
                    "Solution": f"In {1850+history_index}, the temperature of {cities[0][-1]} is {temperatures[0]}, {cities[1][-1]} is {temperatures[1]}, {cities[2][-1]} is {temperatures[2]}, and {cities[3][-1]} is {temperatures[3]}.\nTherefore, the city with the highest temperature is {hottest_city[-1]}. The answer is {correct_option}.",
                }
            )
        history_indexs = random.sample(range(0, 165), 3)
        for history_index in history_indexs:
            temperatures = [
                history_temperature(city[0], city[1], 1850 + history_index)
                for city in cities
            ]
            temperatures = [i[0] for i in temperatures]
            coldest_city = cities[temperatures.index(min(temperatures))]
            options, correct_option = generate_simple_question(cities, coldest_city)
            questions.append(
                {
                    "Question": f"For {', '.join([city[-1] for city in cities])}, which city has the lowest temperature in {1850+history_index}?",
                    "Options": options,
                    "Correct": correct_option,
                    "Solution": f"In {1850+history_index}, the temperature of {cities[0][-1]} is {temperatures[0]}, {cities[1][-1]} is {temperatures[1]}, {cities[2][-1]} is {temperatures[2]}, and {cities[3][-1]} is {temperatures[3]}.\nTherefore, the city with the lowest temperature is {coldest_city[-1]}. The answer is {correct_option}.",
                }
            )
        for scenario in climate_settings:
            future_indexs = random.sample(range(0, 86), 3)
            for future_index in future_indexs:
                temperatures = [
                    future_temperature(city[0], city[1], scenario, 2015 + future_index)
                    for city in cities
                ]
                temperatures = [i[0] for i in temperatures]
                hottest_city = cities[temperatures.index(max(temperatures))][-1]
                options, correct_option = generate_simple_question(
                    [city[-1] for city in cities], hottest_city
                )
                questions.append(
                    {
                        "Question": f"For {', '.join([city[-1] for city in cities])}, which city has the highest temperature in {2015+future_index} under {scenario}?",
                        "Options": options,
                        "Correct": correct_option,
                        "Solution": f"In {2015+future_index}, under {scenario}, the temperature of {cities[0][-1]} is {temperatures[0]}, {cities[1][-1]} is {temperatures[1]}, {cities[2][-1]} is {temperatures[2]}, and {cities[3][-1]} is {temperatures[3]}.\nTherefore, the city with the highest temperature is {hottest_city}. The answer is {correct_option}.",
                    }
                )
            future_indexs = random.sample(range(0, 86), 3)
            for future_index in future_indexs:
                temperatures = [
                    future_temperature(city[0], city[1], scenario, 2015 + future_index)
                    for city in cities
                ]
                temperatures = [i[0] for i in temperatures]
                coldest_city = cities[temperatures.index(min(temperatures))][-1]
                options, correct_option = generate_simple_question(
                    [city[-1] for city in cities], coldest_city
                )
                questions.append(
                    {
                        "Question": f"For {', '.join([city[-1] for city in cities])}, which city has the lowest temperature in {2015+future_index} under {scenario}?",
                        "Options": options,
                        "Correct": correct_option,
                        "Solution": f"In {2015+future_index}, under {scenario}, the temperature of {cities[0][-1]} is {temperatures[0]}, {cities[1][-1]} is {temperatures[1]}, {cities[2][-1]} is {temperatures[2]}, and {cities[3][-1]} is {temperatures[3]}.\nTherefore, the city with the lowest temperature is {coldest_city}. The answer is {correct_option}.",
                    }
                )
    for city in cities:
        # simple
        history_indexs = random.sample(range(0, 165), 5)
        for history_index in history_indexs:
            temp, _ = location_summary(city[0], city[1])
            temp = temp["history"][history_index]
            options, correct_option = generate_number_choice(temp)
            questions.append(
                {
                    "Question": f"What is the average temperature of {city[-1]} in {1850+history_index}?",
                    "Options": options,
                    "Correct": correct_option,
                    "Solution": f"The average temperature of {city[-1]} in {1850+history_index} is {temp}. The answer is {correct_option}.",
                }
            )
        history_indexs = random.sample(range(0, 165), 5)
        for history_index in history_indexs:
            temp, _ = location_summary(city[0], city[1])
            temp = temp["history"][history_index]
            options, correct_option = generate_interval_choices(temp)
            questions.append(
                {
                    "Question": f"What is the average temperature of {city[-1]} in {1850+history_index}?",
                    "Options": options,
                    "Correct": correct_option,
                    "Solution": f"The average temperature of {city[-1]} in {1850+history_index} is {temp}. The answer is {correct_option}.",
                }
            )
        for scenario in climate_settings:
            future_indexs = random.sample(range(0, 86), 3)
            for future_index in future_indexs:
                temp, _ = location_summary(city[0], city[1])
                temp = temp["future"][scenario][future_index]
                options, correct_option = generate_number_choice(temp)
                questions.append(
                    {
                        "Question": f"What is the average temperature of {city[-1]} in {2015+future_index} under {scenario}?",
                        "Options": options,
                        "Correct": correct_option,
                        "Solution": f"The average temperature of {city[-1]} in {2015+future_index} under {scenario} is {temp}. The answer is {correct_option}.",
                    }
                )
            future_indexs = random.sample(range(0, 86), 3)
            for future_index in future_indexs:
                temp, _ = location_summary(city[0], city[1])
                temp = temp["future"][scenario][future_index]
                options, correct_option = generate_interval_choices(temp)
                questions.append(
                    {
                        "Question": f"What is the average temperature of {city[-1]} in {2015+future_index} under {scenario}?",
                        "Options": options,
                        "Correct": correct_option,
                        "Solution": f"The average temperature of {city[-1]} in {2015+future_index} under {scenario} is {temp}. The answer is {correct_option}.",
                    }
                )
        for scenario in climate_settings:
            future_indexs = random.sample(range(0, 86), 3)
            for future_index in future_indexs:
                temperature_ssp126, _ = future_temperature(
                    city[0], city[1], "ssp126", 2015 + future_index
                )
                temperature_ssp245, _ = future_temperature(
                    city[0], city[1], "ssp245", 2015 + future_index
                )
                temperature_ssp370, _ = future_temperature(
                    city[0], city[1], "ssp370", 2015 + future_index
                )
                temperature_ssp585, _ = future_temperature(
                    city[0], city[1], "ssp585", 2015 + future_index
                )
                temp_min = min(
                    temperature_ssp126,
                    temperature_ssp245,
                    temperature_ssp370,
                    temperature_ssp585,
                )
                temp_max = max(
                    temperature_ssp126,
                    temperature_ssp245,
                    temperature_ssp370,
                    temperature_ssp585,
                )
                options, correct_option = generate_interval_choices_(temp_min, temp_max)
                questions.append(
                    {
                        "Question": f"What is the range of temperature of {city[-1]} in {2015+future_index} under different climate settings?",
                        "Options": options,
                        "Correct": correct_option,
                        "Solution": f"In {2015+future_index}, the temperature of {city[-1]} under ssp126 is {temperature_ssp126}, under ssp245 is {temperature_ssp245}, under ssp370 is {temperature_ssp370}, and under ssp585 is {temperature_ssp585}.\nTherefore, in {2015+future_index}, the minimum temperature of {city[-1]} is {temp_min} and the maximum temperature of {city[-1]} is {temp_max}. The range is {temp_min} to {temp_max}. The answer is {correct_option}.",
                    }
                )
        # trend
        history_temperatures, _ = location_summary(city[0], city[1])
        history_temperatures = history_temperatures["history"]
        trend_choices, correct_option = generate_trend_quali(history_temperatures)
        questions.append(
            {
                "Question": f"What is the trend of temperature in {city[-1]} from 1850 to 2000?",
                "Options": trend_choices,
                "Correct": correct_option,
                "Solution": f"The temperature of {city[-1]} is {history_temperatures[0]} in 1850, {history_temperatures[25]} in 1875, {history_temperatures[50]} in 1900, {history_temperatures[75]} in 1925, {history_temperatures[100]} in 1950, {history_temperatures[125]} in 1975, and {history_temperatures[150]} in 2000.\nTherefore, the trend of temperature in {city[-1]} from 1850 to 2000 is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}.",
            }
        )
        trend_choices, correct_option = generate_trend_quant(history_temperatures)
        questions.append(
            {
                "Question": f"What is the trend of temperature in {city[-1]} from 1850 to 2000?",
                "Options": trend_choices,
                "Correct": correct_option,
                "Solution": f"The temperature of {city[-1]} is {history_temperatures[0]} in 1850, {history_temperatures[25]} in 1875, {history_temperatures[50]} in 1900, {history_temperatures[75]} in 1925, {history_temperatures[100]} in 1950, {history_temperatures[125]} in 1975, and {history_temperatures[150]} in 2000.\nTherefore, the trend of temperature in {city[-1]} from 1850 to 2000 is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}.",
            }
        )
        for scenario in climate_settings:
            future_temperatures, _ = location_summary(city[0], city[1])
            future_temperatures = future_temperatures["future"][scenario]
            trend_choices, correct_option = generate_trend_quali(future_temperatures)
            questions.append(
                {
                    "Question": f"What is the trend of temperature in {city[-1]} from 2015 to 2100 under {scenario}?",
                    "Options": trend_choices,
                    "Correct": correct_option,
                    "Solution": f"Under {scenario}, the temperature of {city[-1]} is {future_temperatures[0]} in 2015, {future_temperatures[10]} in 2025, {future_temperatures[35]} in 2050, {future_temperatures[60]} in 2075, and {future_temperatures[85]} in 2100.\nTherefore, the trend of temperature in {city[-1]} from 2015 to 2100 under {scenario} is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}.",
                }
            )
            trend_choices, correct_option = generate_trend_quant(future_temperatures)
            questions.append(
                {
                    "Question": f"What is the trend of temperature in {city[-1]} from 2015 to 2100 under {scenario}?",
                    "Options": trend_choices,
                    "Correct": correct_option,
                    "Solution": f"Under {scenario}, the temperature of {city[-1]} is {future_temperatures[0]} in 2015, {future_temperatures[10]} in 2025, {future_temperatures[35]} in 2050, {future_temperatures[60]} in 2075, and {future_temperatures[85]} in 2100.\nTherefore, the trend of temperature in {city[-1]} from 2015 to 2100 under {scenario} is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}.",
                }
            )
        future_indexs = random.sample(range(0, 86), 3)
        for future_index in future_indexs:
            temperature_ssp126, _ = future_temperature(
                city[0], city[1], "ssp126", 2015 + future_index
            )
            temperature_ssp245, _ = future_temperature(
                city[0], city[1], "ssp245", 2015 + future_index
            )
            temperature_ssp370, _ = future_temperature(
                city[0], city[1], "ssp370", 2015 + future_index
            )
            temperature_ssp585, _ = future_temperature(
                city[0], city[1], "ssp585", 2015 + future_index
            )
            temperature_temp = [
                temperature_ssp126,
                temperature_ssp245,
                temperature_ssp370,
                temperature_ssp585,
            ]
            if (
                temperature_ssp126
                < temperature_ssp245
                < temperature_ssp370
                < temperature_ssp585
            ):
                random_index = random.randint(0, 3)
                choices, answer = generate_simple_question(
                    climate_settings, climate_settings[random_index]
                )
                questions.append(
                    {
                        "Question": f"What is the minimum level of agreement we should support if we want to control the temperature of {city[-1]} in {2015+future_index} under {temperature_temp[random_index] + 0.01}?",
                        "Options": choices,
                        "Correct": answer,
                        "Solution": f"In {2015 + future_index}, the temperature of {city[-1]} under ssp126 is {temperature_ssp126}, under ssp245 is {temperature_ssp245}, under ssp370 is {temperature_ssp370}, and under ssp585 is {temperature_ssp585}.\nTherefore, the minimum level of agreement we should support if we want to control the temperature of {city[-1]} in {2015+future_index} is {climate_settings[random_index]}. The answer is {answer}.",
                    }
                )
            else:
                max_index = temperature_temp.index(max(temperature_temp))
                choices, answer = generate_simple_question(
                    climate_settings, climate_settings[max_index]
                )
                questions.append(
                    {
                        "Question": f"Under which climate setting, the temperature of {city[-1]} in {2015+future_index} is the highest?",
                        "Options": choices,
                        "Correct": answer,
                        "Solution": f"In {2015 + future_index}, the temperature of {city[-1]} under ssp126 is {temperature_ssp126}, under ssp245 is {temperature_ssp245}, under ssp370 is {temperature_ssp370}, and under ssp585 is {temperature_ssp585}.\nTherefore, the climate setting that makes the temperature of {city[-1]} in {2015+future_index} the highest is {climate_settings[max_index]}. The answer is {answer}.",
                    }
                )
        for scenario in climate_settings:
            for gas in ["CO2", "CH4"]:
                future_indexs = random.sample(range(0, 86), 3)
                for future_index in future_indexs:
                    variance = [x / 20 for x in range(-20, 20)]
                    variance = random.sample(variance, 5)
                    for v in variance:
                        if v == 0:
                            continue
                        if gas == "CO2":
                            temperature, _ = future_temperature(
                                city[0], city[1], scenario, 2015 + future_index
                            )
                            temperature_var, _ = diy_greenhouse(
                                city[0], city[1], scenario, 2015 + future_index, v, 0
                            )
                            choices, correct = generate_number_choice(temperature_var)
                            questions.append(
                                {
                                    "Question": f"What is the temperature of {city[-1]} in {2015+future_index} under {scenario} if the emission of CO2 is {'increased' if v > 0 else 'decreased'} by {int(v*100)}%?",
                                    "Options": choices,
                                    "Correct": correct,
                                    "Solution": f"The temperature of {city[-1]} in {2015+future_index} under {scenario} is {temperature}.\nIf the emission of CO2 is {'increased' if v > 0 else 'decreased'} by {int(v*100)}%, the temperature will be {temperature_var}. The answer is {correct}.",
                                }
                            )
                        else:
                            temperature, _ = future_temperature(
                                city[0], city[1], scenario, 2015 + future_index
                            )
                            temperature_var, _ = diy_greenhouse(
                                city[0], city[1], scenario, 2015 + future_index, 0, v
                            )
                            choices, correct = generate_number_choice(temperature_var)
                            questions.append(
                                {
                                    "Question": f"What is the temperature of {city[-1]} in {2015+future_index} under {scenario} if the emission of CH4 is {'increased' if v > 0 else 'decreased'} by {int(v*100)}%?",
                                    "Options": choices,
                                    "Correct": correct,
                                    "Solution": f"The temperature of {city[-1]} in {2015+future_index} under {scenario} is {temperature}.\nIf the emission of CH4 is {'increased' if v > 0 else 'decreased'} by {int(v*100)}%, the temperature will be {temperature_var}. The answer is {correct}.",
                                }
                            )
                for future_index in future_indexs:
                    variance = [x / 20 for x in range(-20, 20)]
                    variance = random.sample(variance, 5)
                    for v in variance:
                        if v == 0:
                            continue
                        if gas == "CO2":
                            temperature, _ = future_temperature(
                                city[0], city[1], scenario, 2015 + future_index
                            )
                            temperature_var, _ = diy_greenhouse(
                                city[0], city[1], scenario, 2015 + future_index, v, 0
                            )
                            choices, correct = generate_number_choice(
                                temperature_var - temperature
                            )
                            questions.append(
                                {
                                    "Question": f"How much will the temperature of {city[-1]} in {2015+future_index} under {scenario} change if the emission of CO2 is {'increased' if v > 0 else 'decreased'} by {int(v*100)}%?",
                                    "Options": choices,
                                    "Correct": correct,
                                    "Solution": f"The temperature of {city[-1]} in {2015+future_index} under {scenario} is {temperature}.\nIf the emission of CO2 is {'increased' if v > 0 else 'decreased'} by {int(v*100)}%, the temperature will be {temperature_var}. Therefore, the temperature will change by {temperature_var-temperature}. The answer is {correct}.",
                                }
                            )
                        else:
                            temperature, _ = future_temperature(
                                city[0], city[1], scenario, 2015 + future_index
                            )
                            temperature_var, _ = diy_greenhouse(
                                city[0], city[1], scenario, 2015 + future_index, 0, v
                            )
                            choices, correct = generate_number_choice(
                                temperature_var - temperature
                            )
                            questions.append(
                                {
                                    "Question": f"How much will the temperature of {city[-1]} in {2015+future_index} under {scenario} change if the emission of CH4 is {'increased' if v > 0 else 'decreased'} by {int(v*100)}%?",
                                    "Options": choices,
                                    "Correct": correct,
                                    "Solution": f"The temperature of {city[-1]} in {2015+future_index} under {scenario} is {temperature}.\nIf the emission of CH4 is {'increased' if v > 0 else 'decreased'} by {int(v*100)}%, the temperature will be {temperature_var}. Therefore, the temperature will change by {temperature_var-temperature}. The answer is {correct}.",
                                }
                            )


def generate(world):
    if world == "climate":
        lat_and_lon = pd.read_csv(
            "./tools/Climate_offline/data/worldcities.csv"
        )
        country_city_counts = lat_and_lon.groupby("country")["city"].nunique()
        eligible_countries = country_city_counts[country_city_counts > 4].index.tolist()

        if eligible_countries:
            selected_country = random.choice(eligible_countries)
            country_cities = lat_and_lon[lat_and_lon["country"] == selected_country][
                "city"
            ].unique()
            random_cities = random.sample(list(country_cities), 4)

            cities = []
            for city in random_cities:
                city_data = lat_and_lon[
                    (lat_and_lon["country"] == selected_country)
                    & (lat_and_lon["city"] == city)
                ].iloc[0]
                cities.append((city_data["lng"], city_data["lat"], city))

            print(cities, selected_country)
            generate_question(cities, selected_country)


if __name__ == "__main__":
    for _ in range(10000):
        generate("climate")
        with open("climate_questions.json", "w") as f:
            json.dump(questions, f, cls=NumpyEncoder, indent=4)