functions_pandemic = {
    "diy_scenario": {
    "name": "emulate_online",
    "description": "Emulates an online model, modifying features by delta and predicting peak day and value.",
    "parameters": {
        "type": "object",
        "properties": {
            "delta_str": {
                "type": "string",
                "description": "String representing a list of nested arrays in the format '[[index, value], [index, value]]'."
            },
            "feature": {
                "type": "integer",
                "description": "Name of the input feature to affect.",
                "enum": [
    'seasonality_min', 'omega_community_interventions', 'omega_work_interventions',
    'omega_school_interventions', 'omega_home_interventions', 'alpha_school_interventions',
    'transit_commute_interventions', 'international_travel_interventions',
    'domestic_travel_interventions', 'R0'
]
            },
            "out": {
                "type": "integer",
                "description": "The output name.",
                "enum": [
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
            }
        },
        "required": ["delta_str", "feature", "out"]
    }
},
    "emulate": {
    "name": "emulate",
    "description": "Emulates the pandemic by giving the output feature that you want. This fuction will give you the value of the feature in following days.",
    "parameters": {
        "type": "object",
        "properties": {
            "out": {
                "type": "integer",
                "description": "The output name.",
                "enum": [
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
            }
        },
        "required": ["out"]
    }
}
}