# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import argparse
# import yaml
# from lib import utils
# from lib.utils import load_graph_data
# from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
# import random
import numpy as np
# import os

# with open('./tools/data/model/dcrnn_cov.yaml') as f:
#     supervisor_config = yaml.safe_load(f)

# graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
# sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data('/home/test/test03/IDM/test/PGLLM/src/emulators/pandemic/data/sensor_graph/adj_mx.pkl')

# i=0
# np.random.seed(i)
# random.seed(i)
# max_itr = 1 #12
# data = utils.load_dataset(**supervisor_config.get('data'))

# supervisor = DCRNNSupervisor(random_seed=i, iteration=0, max_itr = max_itr, 
#         adj_mx=adj_mx, **supervisor_config)

input_features = [
    'seasonality_min', 'omega_community_interventions', 'omega_work_interventions',
    'omega_school_interventions', 'omega_home_interventions', 'alpha_school_interventions',
    'transit_commute_interventions', 'international_travel_interventions',
    'domestic_travel_interventions', 'R0'
]

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

# def main():
#     print(data['x_test'].shape)
#     print(data['y_test'].shape)
#     print(data['x0_test'].shape)
#     supervisor._data = data
#     supervisor.load_model(0,0,277)
#     mae_metric, rmse_metric, result = supervisor.evaluate(dataset='test')
#     print(mae_metric, rmse_metric, result["prediction"].shape)

data_off = np.load('./tools/data/data/test.npz')
y_off = data_off['y']

def emulate(scenario, out):
    out = output_features.index(out)
    answer = y_off[scenario, :, out]
    return "\n".join([f"Day {i}: {a}" for i, a in enumerate(answer)])

# def diy_scenario(scenario, delta, feature, out):
#     feature = input_features.index(feature)
#     out = output_features.index(out)
#     for i in delta:
#         data['x_test'][scenario, :, i[0], feature] += i[1]
#     supervisor._data = data
#     supervisor.load_model(0,0,277)
#     mae_metric, rmse_metric, result = supervisor.evaluate(dataset='test')
#     prediction = result["prediction"][:, scenario, out]
    
#     peak_index = np.argmax(prediction)
#     peak_value = prediction[peak_index]
    
#     return f"Peak occurs at day {peak_index + 1} with a value of {peak_value:.4f}"
    
if __name__ == '__main__':
    # print(diy_scenario(233, [[1, 2], [2, 3]], 0, 0))
    # print(diy_scenario(233, [[1, 3], [2, 2]], 0, 0))
    print(emulate(1, 1, 1))
