"""
Minor Cannabinoids
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 8/23/2023
Updated: 8/23/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Files:

    - [CA Lab Results 2023-05-30](https://cannlytics.page.link/ca-lab-results-2023-05-30)
    - [MA Lab Results 2023-05-30](https://cannlytics.page.link/ma-lab-results-2023-05-30)

Data Sources:

    - [Glass House Farms Strains](https://glasshousefarms.org/strains/)
    - [MCR Labs Test Results](https://reports.mcrlabs.com)

"""
# Standard imports:
import ast
import json

# External imports:
from cannlytics.utils import convert_to_numeric
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns


#-----------------------------------------------------------------------
# Get latest lab results.
#-----------------------------------------------------------------------

DATAFILES = {
    'ca': 'data/ca-lab-results-2023-05-30.xlsx',
    'ma': 'data/ma-lab-results-2023-05-30.xlsx',
}

# Read in CA data.
ca_results = pd.read_excel(DATAFILES['ca'])
ca_results['state'] = 'ca'

# Read in MA data.
ma_results = pd.read_excel(DATAFILES['ma'])
ma_results['state'] = 'ma'


#-----------------------------------------------------------------------
# Standardize the data.
#-----------------------------------------------------------------------

def get_result_value(
        results,
        analyte,
        key='key',
        value='value',
        method='list',
    ):
    """Get the value for an analyte from a list of standardized results."""
    # Ensure that the results are a list.
    try:
        result_list = json.loads(results)
    except:
        try:
            result_list = ast.literal_eval(results)
        except:
            result_list = []
    if not isinstance(result_list, list):
        return None

    # DataFrame method.
    if method == 'df':
        result_data = pd.DataFrame(result_list)
        if result_data.empty:
            return None
        result = result_data.loc[result_data[key] == analyte, value]
        try:
            return convert_to_numeric(result, strip=True)
        except:
            return result

    # List method.
    for result in result_list:
        if result[key] == analyte:
            try:
                return convert_to_numeric(result[value], strip=True)
            except:
                return result[value]

# Define cannabinoids of interest.
cannabinoids = ['thc', 'thca', 'delta_8_thc', 'thcv', 'thcva',
                'cbg', 'cbga', 'cbn', 'cbdv', 'cbdva',  'cbc', 'cbca',
                'cbcv', 'cbt', 'cbl', 'cbla']


# Get CA minor cannabinoids.
results = ca_results['results']
for cannabinoid in cannabinoids:
    ca_results[cannabinoid] = pd.to_numeric(
        results.apply(lambda x: get_result_value(x, cannabinoid)),
        errors='coerce',
    )

# Get MA minor cannabinoids.
results = ma_results['results']
for cannabinoid in cannabinoids:
    ma_results[cannabinoid] = pd.to_numeric(
        results.apply(lambda x: get_result_value(x, cannabinoid)),
        errors='coerce',
    )


#-----------------------------------------------------------------------
# Analyze the data.
#-----------------------------------------------------------------------


def calculate_detection_rate(dataframe, cannabinoid):
    """Calculate the detection rate of a cannabinoid in a dataframe."""
    detected = dataframe[cannabinoid].notna().sum()
    total_samples = len(dataframe)
    return round(detected / total_samples, 4)


def calculate_average_concentration(dataframe, cannabinoid):
    """Calculate the average concentration when detected."""
    detected_values = dataframe[dataframe[cannabinoid].notna()]
    return round(detected_values[cannabinoid].mean(), 4)


# Restrict the sample to flower only.
ca_results = ca_results[ca_results['product_type'] != 'Plant (Enhanced/Infused Preroll)']
ma_results = ma_results[ma_results['product_type'] == 'flower']

# Lists to store the data.
states = ['ca', 'ma']
detection_rates = {}
average_concentrations = {}

# Calculate the detection rates and average concentrations.
for state in states:
    state_data = eval(f"{state}_results")
    detection_rates[state] = {}
    average_concentrations[state] = {}
    for cannabinoid in cannabinoids:
        try:
            detection_rates[state][cannabinoid] = calculate_detection_rate(state_data, cannabinoid)
            average_concentrations[state][cannabinoid] = calculate_average_concentration(state_data, cannabinoid)
        except KeyError:
            pass


#-----------------------------------------------------------------------
# Visualize the data.
#-----------------------------------------------------------------------

# # Setup plotting style.
# plt.rcParams.update({
#     'font.family': 'Times New Roman',
#     'font.size': 24,
# })

# Example: Plot detection rates for a given cannabinoid in all states.
cannabinoid_to_plot = 'thcva'

# Plot detection rates by state.
plt.bar(detection_rates.keys(), [detection_rates[state][cannabinoid_to_plot] for state in states])
plt.title(f"Detection rates of {cannabinoid_to_plot}")
plt.ylabel('Detection rate')
plt.show()

# Plot average concentrations by state.
plt.bar(average_concentrations.keys(), [average_concentrations[state][cannabinoid_to_plot] for state in states])
plt.title(f"Average concentrations of {cannabinoid_to_plot}")
plt.ylabel('Concentration (%)')
plt.show()


#-----------------------------------------------------------------------
# Visualize ratios.
#-----------------------------------------------------------------------

def compute_total(dataframe, main_compound, acidic_form):
    """Compute total compound value: main_compound + 0.877 * acidic_form."""
    return dataframe[main_compound] + 0.877 * dataframe[acidic_form]


# Define colors for each state
colors = {
    'ca': 'purple',
    'ma': 'green'
}

# Handle different naming conventions.
ca_results['thc'] = ca_results['delta_9_thc']

# Compute total THC and total CBG.
for state in states:
    state_data = eval(f"{state}_results")
    state_data['total_cbg'] = compute_total(state_data, 'cbg', 'cbga')
    state_data['total_thc'] = compute_total(state_data, 'thc', 'thca')

# Create a scatter plot with color-coded points for each state
for state in states:
    state_data = eval(f"{state}_results")
    plt.scatter(state_data['total_cbg'], state_data['total_thc'], alpha=0.5, color=colors[state], label=state.upper())


# Configure plot.
plt.title('Total THC to Total CBG in CA and MA Flower')
plt.xlabel('Total CBG (%)')
plt.ylabel('Total THC (%)')
plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('total_thc_to_total_cbg.png', dpi=300, bbox_inches='tight')
plt.show()
