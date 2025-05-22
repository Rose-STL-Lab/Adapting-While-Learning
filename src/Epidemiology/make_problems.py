import numpy as np
import random
import json
from utils import *
from tqdm import tqdm
import random

# Define lists of features and counties (same as before)
input_features = [
    'seasonality_min', 'omega_community_interventions', 'omega_work_interventions',
    'omega_school_interventions', 'omega_home_interventions', 'alpha_school_interventions',
    'transit_commute_interventions', 'international_travel_interventions',
    'domestic_travel_interventions', 'R0'
]
input_features = [i.replace('_', ' ') for i in input_features]

counties = [
    'Alameda', 'Alpine', 'Amador', 'Butte', 'Calaveras', 'Colusa', 'Contra Costa', 
    'Del Norte', 'El Dorado', 'Fresno', 'Glenn', 'Humboldt', 'Imperial', 'Inyo', 
    'Kern', 'Kings', 'Lake', 'Lassen', 'Los Angeles', 'Madera', 'Marin', 'Mariposa', 
    'Mendocino', 'Merced', 'Modoc', 'Mono', 'Monterey', 'Napa', 'Nevada', 'Orange', 
    'Placer', 'Plumas', 'Riverside', 'Sacramento', 'San Benito', 'San Bernardino', 
    'San Diego', 'San Francisco', 'San Joaquin', 'San Luis Obispo', 'San Mateo', 
    'Santa Barbara', 'Santa Clara', 'Santa Cruz', 'Shasta', 'Sierra', 'Siskiyou', 
    'Solano', 'Sonoma', 'Stanislaus', 'Sutter', 'Tehama', 'Trinity', 'Tulare', 
    'Tuolumne', 'Ventura', 'Yolo', 'Yuba'
]

output_features = [
    'prevalence_CA_state_total_Latent', 'prevalence_CA_state_total_Infectious_symptomatic',
    'prevalence_CA_state_total_Infectious_asymptomatic', 'prevalence_CA_state_total_Hospitalized',
    'prevalence_CA_state_total_ICU', 'prevalence_CA_state_total_Removed_asymptomatic',
    'prevalence_CA_state_total_Removed_symptomatic', 'prevalence_CA_state_total_Home_asymptomatic',
    'prevalence_CA_state_total_Home_mild', 'prevalence_CA_state_total_Home_severe',
    'prevalence_CA_state_total_Removed_hospitalized', 'prevalence_CA_state_total_Deaths_hospitalized',
    'incidence_CA_state_total_Latent', 'incidence_CA_state_total_Infectious_symptomatic',
    'incidence_CA_state_total_Infectious_asymptomatic', 'incidence_CA_state_total_Hospitalized',
    'incidence_CA_state_total_ICU', 'incidence_CA_state_total_Removed_asymptomatic',
    'incidence_CA_state_total_Removed_symptomatic', 'incidence_CA_state_total_Home_asymptomatic',
    'incidence_CA_state_total_Home_mild', 'incidence_CA_state_total_Home_severe',
    'incidence_CA_state_total_Removed_hospitalized', 'incidence_CA_state_total_Deaths_hospitalized'
]

interest_features = [
    'prevalence_CA_state_total_Infectious_symptomatic',
    'prevalence_CA_state_total_Infectious_asymptomatic', 'prevalence_CA_state_total_Hospitalized',
    'prevalence_CA_state_total_ICU', 'prevalence_CA_state_total_Deaths_hospitalized',
    'incidence_CA_state_total_Infectious_symptomatic',
    'incidence_CA_state_total_Infectious_asymptomatic', 'incidence_CA_state_total_Hospitalized',
    'incidence_CA_state_total_ICU', 'incidence_CA_state_total_Deaths_hospitalized'
]

output_features = [i.replace('_', ' ') for i in output_features]

interest_features = [i.replace('_', ' ') for i in interest_features]

def are_all_values_identical(arr):
    """Check if all values in the array are identical."""
    return np.all(arr == arr[0])

def are_values_identical_across_days(x, scenario, feature_index):
    """Check if the values for a given feature are identical across all days and counties."""
    return np.all(x[scenario, :, :, feature_index] == x[scenario, 0, 0, feature_index])

def emulate(y, scenario, day, output_feature):
    output_index = output_features.index(output_feature)

    answer = y[scenario, day, output_index]
    
    return answer

qa_pairs = []

def load_data_and_generate_qa():
    data = np.load('test.npz')
    x = data['x']
    y = data['y']
    
    for _ in tqdm(range(1000)):
        scenario = random.randint(0, 10)
        
        # Construct question
        question = "Given the following information:\n\n"
        
        question += "Input Features:\n"
        for i, feature in enumerate(input_features):
            question += f"{i}: {feature}\n"
        question += "\n"
        
        question += "Counties:\n"
        for i, county_name in enumerate(counties):
            question += f"{i}: {county_name}\n"
        question += "\n"

        question += "Controllable Features:\n"

        for i, feature in enumerate(input_features):
            if are_values_identical_across_days(x, scenario, i):
                question += f"All days and counties: {feature} = {x[scenario, 0, 0, i]:.2f}\n"

        # Then, process the remaining features day by day
        for t in range(0, 28, 7):  # 28 days
            question += f"\nDay {t}:\n"
            for i, feature in enumerate(input_features):
                if not are_values_identical_across_days(x, scenario, i):
                    feature_values = x[scenario, t, :, i]
                    if are_all_values_identical(feature_values):
                        # If values are identical across counties for a specific day, summarize per day
                        question += f"  All counties: {feature} = {feature_values[0]:.2f}\n"
                    else:
                        # Otherwise, list each county's feature value for that day
                        question += feature + "\n"
                        for j, county_name in enumerate(counties):
                            question += f"{x[scenario, t, j, i]:.2f} "
                        question += "\n"

        question += "Output Features:\n"
        for i, feature in enumerate(output_features):
            question += f"{i}: {feature}\n"
        question += "\n"

        question += "Initial fratures:\n"
        question += ", ".join(f"{y[scenario, 0, i]:.2f}" for i in range(len(output_features)))
        question += "\n\n"
        
        scenario = random.randint(0, x.shape[0] - 1)
        day = random.randint(1, 28) 
        output_feature = random.choice(interest_features)
        
        answer = emulate(y, scenario, day, output_feature)
        options, correct_option = generate_number_choice(answer)
        
        qa_pairs.append({
            "question": question + f"Given the scenario data, what will be the value of {output_feature} on day {day}?",
            "options": options,
            "correct_option": correct_option,
            "scenario": scenario
        })
        
        start_day = random.randint(1, 27)
        end_day = random.randint(start_day + 1, 28)
        values = [emulate(y, scenario, d, output_feature) for d in range(start_day, end_day)]
        min_val, max_val = min(values), max(values)
        
        if min_val != max_val:
            options, correct_option = generate_interval_choices_(min_val, max_val)
            
            qa_pairs.append({
                "question": question + f"Given the scenario data, what is the range of values for {output_feature} between day {start_day} and day {end_day}?",
                "options": options,
                "correct_option": correct_option,
                "scenario": scenario
            })
        
        values = [emulate(y, scenario, d, output_feature) for d in range(29)]
        if values[0] != 0:
            options, correct_option = generate_trend_quant(values)
            
            qa_pairs.append({
                "question": question + f"Given the scenario data, what is the overall trend of {output_feature} over the 29-day period?",
                "options": options,
                "correct_option": correct_option,
                "scenario": scenario
            })

        extreme_choices, extreme_correct_option, extreme_type = generate_extreme_options(values)
        
        qa_pairs.append({
            "question": question + f"Given the scenario data for {output_feature}, on which day does the {extreme_type} value occur?",
            "options": extreme_choices,
            "correct_option": extreme_correct_option,
            "scenario": scenario
        })

load_data_and_generate_qa()

random.shuffle(qa_pairs)
qa_pairs = qa_pairs[:8000]

with open("qa_pairs3.json", 'w') as f:
    json.dump(qa_pairs, f, cls=NumpyEncoder,indent=4)