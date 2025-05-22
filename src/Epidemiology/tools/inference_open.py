import json
import numpy as np
import torch
from tqdm import tqdm
import argparse
import yaml
from lib import utils
from lib.utils import load_graph_data, DataLoader
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
import random
import os

device = torch.device("cuda:1")

# Global variables
supervisor = None
data = None
x = None
y = None

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

input_features = [
    'seasonality_min', 'omega_community_interventions', 'omega_work_interventions',
    'omega_school_interventions', 'omega_home_interventions', 'alpha_school_interventions',
    'transit_commute_interventions', 'international_travel_interventions',
    'domestic_travel_interventions', 'R0'
]

input_features = [i.replace('_', ' ') for i in input_features]

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

def load_test_data():
    """Load test data into global variables."""
    global data, x, y
    data = np.load('')
    x = data['x']
    y = data['y']

def init_model(config_filename, model_seed=0, model_iteration=0, model_epoch=277):
    """Initialize the global model."""
    global supervisor
    
    # Set random seeds
    np.random.seed(model_seed)
    random.seed(model_seed)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load configuration
    with open(config_filename) as f:
        supervisor_config = yaml.safe_load(f)

    # Load graph data
    graph_pkl_filename = supervisor_config["data"].get("graph_pkl_filename")
    sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

    # Load data and initialize supervisor
    data = utils.load_dataset(**supervisor_config.get("data"))
    supervisor = DCRNNSupervisor(
        random_seed=model_seed,
        iteration=model_iteration,
        max_itr=1,
        adj_mx=adj_mx,
        **supervisor_config
    )
    supervisor._data = data

    # Load model
    supervisor.load_model(model_seed, model_iteration, model_epoch)

def process_model_response(response_text):
    """Parse the model's text response into county adjustments."""
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        json_str = response_text[start_idx:end_idx]
        adjustments = json.loads(json_str)
        return adjustments
    except:
        print("Failed to parse model response")
        return None

def modify_inputs(scenario, adjustments, feature_name):
    """Modify the input tensor based on model's suggested adjustments."""
    global x
    modified_x = x.copy()
    feature_idx = input_features.index(feature_name)
    
    for county_name, adjustment in adjustments.items():
        county_idx = counties.index(county_name)
        modified_x[scenario, :, county_idx, feature_idx] += adjustment
            
    return modified_x

def evaluate_scenario(scenario, modified_x=None):
    """Evaluate a specific scenario using the model."""
    global supervisor, x, y
    
    # Use modified_x if provided, otherwise use global x
    current_x = modified_x if modified_x is not None else x
    
    # Apply the same normalization as in training
    current_y = np.log(y + 1.0)
    
    # Prepare inputs for one scenario
    test_x = current_x[scenario:scenario+1]  # Shape: [1, seq_len, num_nodes, input_dim]
    test_y = current_y[scenario:scenario+1, 1:]  # Shape: [1, seq_len-1, num_nodes]
    test_x0 = current_y[scenario:scenario+1, 0]  # Shape: [1, num_nodes]
    
    # Create DataLoader for the single scenario
    test_loader = DataLoader(
        test_x, 
        test_y, 
        test_x0,
        batch_size=1,
        shuffle=False
    )
    
    # Get the iterator
    test_iterator = test_loader.get_iterator()
    
    # Get the first (and only) batch
    x_batch, y_batch, x0_batch = next(test_iterator)
    
    # Reshape input tensor to match model expectations
    seq_len, num_nodes, input_dim = x_batch.shape[1], x_batch.shape[2], x_batch.shape[3]
    x_reshaped = np.transpose(x_batch, (1, 0, 2, 3))
    x_reshaped = x_reshaped.reshape(seq_len, 1, num_nodes * input_dim)
    
    # Similarly reshape y and x0
    y_reshaped = np.transpose(y_batch, (1, 0, 2))
    x0_reshaped = x0_batch.reshape(1, -1)
    
    with torch.no_grad():
        supervisor.dcrnn_model.eval()
        outputs = supervisor.dcrnn_model(
            torch.from_numpy(x_reshaped).float().to(device),
            torch.from_numpy(y_reshaped).float().to(device),
            torch.from_numpy(x0_reshaped).float().to(device),
            test=True,
            z_mean_all=supervisor.z_mean_all,
            z_var_temp_all=supervisor.z_var_temp_all
        )
    
    # Convert predictions back to original scale
    predictions = outputs[0].cpu().numpy()
    predictions = np.exp(predictions) - 1.0
    
    return predictions

def test_single_scenario(scenario, sample_response, input_feature, output_feature):
    """Test the model with a single scenario."""    
    
    feature_idx = output_features.index(output_feature)
    original_outputs = evaluate_scenario(scenario)
    
    adjustments = process_model_response(sample_response)
    if adjustments:
        modified_x = modify_inputs(scenario, adjustments, input_feature)
        modified_outputs = evaluate_scenario(scenario, modified_x)

        original_feature = original_outputs[0, :, feature_idx]
        modified_feature = modified_outputs[0, :, feature_idx]
        
        results = {
            "scenario": scenario,
            "original_outputs": original_outputs,
            "modified_outputs": modified_outputs,
            "adjustments": adjustments,
            "feature_comparison": {
                "feature_name": output_feature,
                "peak_comparison": {
                    "original_peak": float(np.max(original_feature)),
                    "modified_peak": float(np.max(modified_feature)),
                    "peak_difference": float(np.max(modified_feature) - np.max(original_feature))
                },
                "tail_comparison": {
                    "original_tail": float(original_feature[-1]),
                    "modified_tail": float(modified_feature[-1]),
                    "tail_difference": float(modified_feature[-1] - original_feature[-1])
                }
            }
        }
        
        return results
    else:
        return None

init_model("")
load_test_data()
    
def main():
    scenario = 0
    response = '{"Los Angeles": 0.1, "San Francisco": 0.05}'
    input_feature = "omega community interventions"
    output_feature = "prevalence CA state total ICU"

    results = test_single_scenario(scenario, response, input_feature, output_feature)
    print(results)