import json
import random

import numpy as np
from tqdm import tqdm

input_features = [
    'seasonality_min', 'omega_community_interventions', 'omega_work_interventions',
    'omega_school_interventions', 'omega_home_interventions', 'alpha_school_interventions',
    'transit_commute_interventions', 'international_travel_interventions',
    'domestic_travel_interventions', 'R0'
]
input_features = [i.replace('_', ' ') for i in input_features]

changeable_features = [
    'omega_community_interventions', 'omega_work_interventions',
    'omega_school_interventions', 'omega_home_interventions', 'alpha_school_interventions',
    'transit_commute_interventions', 'international_travel_interventions',
    'domestic_travel_interventions'
]
changeable_features = [i.replace('_', ' ') for i in input_features]

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

output_features = [i.replace('_', ' ') for i in output_features]

feature_mean = {
    "seasonality min": 0.8382,
    "omega community interventions": 0.5604,
    "omega work interventions": 0.4459,
    "omega school interventions": 0.7659,
    "omega home interventions": 0.9995,
    "alpha school interventions": 0.5090,
    "transit commute interventions": 0.6806,
    "international travel interventions": 0.3430,
    "domestic travel interventions": 0.3430,
    "R0": 2.4415
}

interest_features = [
    'prevalence_CA_state_total_Hospitalized',
    'prevalence_CA_state_total_ICU',
    'prevalence_CA_state_total_Deaths_hospitalized',
    'prevalence_CA_state_total_Home_severe',
    'incidence_CA_state_total_Hospitalized',
    'incidence_CA_state_total_ICU',
    'incidence_CA_state_total_Deaths_hospitalized',
    'incidence_CA_state_total_Home_severe'
]

interest_features = [i.replace('_', ' ') for i in interest_features]

BUDGET = 3  # Fractional budget for adjustment

qa_pairs = []

def are_all_values_identical(arr):
    """Check if all values in the array are identical."""
    return np.all(arr == arr[0])

def are_values_identical_across_days(x, scenario, feature_index):
    """Check if the values for a given feature are identical across all days and counties."""
    return np.all(x[scenario, :, :, feature_index] == x[scenario, 0, 0, feature_index])

def generate_open_questions(x, y):
    """Generate open-ended questions with controllable input features using absolute budgets."""
    for _ in tqdm(range(2000)):
        scenario = random.randint(0, 20)
        
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

        question += "Initial features:\n"
        question += ", ".join(f"{y[scenario, 0, i]:.2f}" for i in range(len(output_features)))
        question += "\n\n"
        
        # Randomly select a controllable input feature
        controllable_feature_index = random.randint(0, len(input_features) - 1)
        controllable_feature = input_features[controllable_feature_index]
        
        # Get the mean value of the selected feature
        mean_value = feature_mean[controllable_feature]
        
        # Calculate the absolute budget based on the mean value
        absolute_budget = BUDGET * mean_value
        max_adjustment_per_county = absolute_budget / 5  # Constraint on each county adjustment
        
        # Randomly select a target output feature to optimize
        target_feature = random.choice(interest_features)
        target_index = output_features.index(target_feature)
        
        for _ in ["peak", "final"]:

            question += (
                f"You are given an total budget of {absolute_budget:.4f} to adjust "
                f"'{controllable_feature}' across all counties. Each county's adjustment "
                f"cannot exceed {max_adjustment_per_county:.4f}. How would you allocate this budget "
                f"to optimize the {_} value of '{target_feature}' by the last day?\n\n"
            )

            question += (
                "Your answer should be in the following format:\n\n"
                "Action: Simulate/Answer\nThought: Describe your reasoning for allocating the budget.\n"
                "Answer: {\n"
                "    \"county_name_1\": x.xx,\n"
                "    \"county_name_2\": x.xx,\n"
                "    ...\n"
                "}\nNow begin!"
            )

            # Append question to QA pairs
            qa_pairs.append({
                "question": question,
                "scenario": scenario,
                "target_feature": target_feature,
                "category": _,
                "input_feature": controllable_feature,
                "absolute_budget": absolute_budget,
                "max_adjustment_per_county": max_adjustment_per_county
            })

# Load data
data = np.load('')
x = data['x']
y = data['y']

# Generate open-ended questions
generate_open_questions(x, y)

# Save the questions
with open("open_questions.json", "w") as f:
    json.dump(qa_pairs, f, indent=4)