import numpy as np

# Define the output features in the same order as in your data
outs = [
    'prevalence CA state total Latent',
    'prevalence CA state total Infectious symptomatic',
    'prevalence CA state total Infectious asymptomatic',
    'prevalence CA state total Hospitalized',
    'prevalence CA state total ICU',
    'prevalence CA state total Removed asymptomatic',
    'prevalence CA state total Removed symptomatic',
    'prevalence CA state total Home asymptomatic',
    'prevalence CA state total Home mild',
    'prevalence CA state total Home severe',
    'prevalence CA state total Removed hospitalized',
    'prevalence CA state total Deaths hospitalized',
    'incidence CA state total Latent',
    'incidence CA state total Infectious symptomatic',
    'incidence CA state total Infectious asymptomatic',
    'incidence CA state total Hospitalized',
    'incidence CA state total ICU',
    'incidence CA state total Removed asymptomatic',
    'incidence CA state total Removed symptomatic',
    'incidence CA state total Home asymptomatic',
    'incidence CA state total Home mild',
    'incidence CA state total Home severe',
    'incidence CA state total Removed hospitalized',
    'incidence CA state total Deaths hospitalized'
]

# Load the data globally so emulate does not need to accept y as an argument
data = np.load('./tools/data/data/test.npz')
y = data['y']

def emulate(scenario, day, out):
    """
    Returns the value for a specific scenario, day, and output feature.
    
    Args:
        scenario (int): Index of the scenario.
        day (int): Index of the day.
        out (str): Name of the output feature (must match the names in outs).
        
    Returns:
        float: The value from the y array.
        
    Raises:
        ValueError: If the output feature is not in the outs list.
    """
    if out not in outs:
        raise ValueError(f"Output feature '{out}' not found in outs list.")
    output_index = outs.index(out)
    return y[scenario, day, output_index]