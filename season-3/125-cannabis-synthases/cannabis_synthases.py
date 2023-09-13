"""
Cannabinoid Synthases
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 8/30/2023
Updated: 8/30/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Files:

    - [CA Lab Results 2023-08-30](https://cannlytics.page.link/ca-lab-results-2023-08-30)
    - [FL Lab Results 2023-06-12](https://cannlytics.page.link/ma-lab-results-2023-06-12)
    - [MA Lab Results 2023-08-30](https://cannlytics.page.link/ma-lab-results-2023-08-30)

Data Sources:

    - [Florida Labs](https://knowthefactsmmj.com/cmtl/)
    - [Florida Licenses](https://knowthefactsmmj.com/mmtc/)
    - [Kaycha Labs](https://yourcoa.com)
    - [The Flowery](https://support.theflowery.co)
    - [TerpLife Labs](https://www.terplifelabs.com)
    - [Glass House Farms Strains](https://glasshousefarms.org/strains/)
    - [MCR Labs Test Results](https://reports.mcrlabs.com)

"""
# Standard imports:

# External imports:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns


#-----------------------------------------------------------------------
# Get latest lab results.
#-----------------------------------------------------------------------

DATAFILES = {
    'ca': 'data/ca-lab-results-2023-08-30.xlsx',
    'fl': 'data/fl-lab-results-2023-06-12.xlsx',
    'ma': 'data/ma-lab-results-2023-08-30.xlsx',
}

# Read in CA data.
ca_results = pd.read_excel(DATAFILES['ca'], sheet_name='Values')
ca_results['state'] = 'ca'

# Read in FL data.
fl_results = pd.read_excel(DATAFILES['fl'], sheet_name='Values')
fl_results['state'] = 'fl'


#-----------------------------------------------------------------------
# Standardize the data.
#-----------------------------------------------------------------------


#-----------------------------------------------------------------------
# Calculate summary statistics.
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


# Define cannabinoids of interest.
cannabinoids = ['delta_9_thc', 'thca', 'delta_8_thc', 'thcv', 'thcva',
                'cbg', 'cbga', 'cbn', 'cbdv', 'cbdva',  'cbc', 'cbca']

# Define terpenes of interest.
terpenes = [
    'beta_caryophyllene',
    'linalool',
    'beta_myrcene',
    'alpha_pinene',
    'beta_pinene',
    'd_limonene',
]


# Restrict the sample to flower only.
ca_results = ca_results.loc[ca_results['product_type'] != 'Plant (Enhanced/Infused Preroll)']
fl_results = fl_results.loc[fl_results['product_type'].str.contains('flower', case=False)]

# Lists to store the data.
states = ['ca', 'fl']
detection_rates = {}
average_concentrations = {}

# Calculate the detection rates and average concentrations.
for state in states:
    state_data = eval(f"{state}_results")
    detection_rates[state] = {}
    average_concentrations[state] = {}
    for analyte in cannabinoids + terpenes:
        try:
            detection_rates[state][analyte] = calculate_detection_rate(state_data, analyte)
            average_concentrations[state][analyte] = calculate_average_concentration(state_data, analyte)
        except KeyError:
            print(f'{analyte} not measured in {state}.', )
            pass


#-----------------------------------------------------------------------
# Visualize the data.
#-----------------------------------------------------------------------

# # Setup plotting style.
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})

# Example: Plot detection rates for a given cannabinoid in all states.
cannabinoid_to_plot = 'thcva'

# Plot detection rates by state.
data = detection_rates
analytes = list(data['fl'].keys())
states = list(data.keys())
bar_width = 0.35
index = np.arange(len(analytes))
fig, ax = plt.subplots(figsize=(15, 8))
for i, state in enumerate(states):
    values = [data[state][analyte] * 100 for analyte in analytes]
    ax.bar(index + i * bar_width, values, bar_width, label=state)
ax.set_ylabel('Percentage (%)')
ax.set_title('Analyte detection rate by state and analyte')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(analytes, rotation=45, ha='right')
ax.legend()
fig.tight_layout()
plt.show()

# Plot average concentrations by state.
data = average_concentrations
analytes = list(data['fl'].keys())[4:]
states = list(data.keys())
bar_width = 0.35
index = np.arange(len(analytes))
fig, ax = plt.subplots(figsize=(15, 8))
for i, state in enumerate(states):
    values = [data[state][analyte] for analyte in analytes]
    ax.bar(index + i * bar_width, values, bar_width, label=state)
ax.set_ylabel('Percentage (%)')
ax.set_title('Analyte concentrations by state and analyte')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(analytes, rotation=45, ha='right')
ax.legend()
fig.tight_layout()
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
    'fl': 'green'
}

# Compute total THC and total CBG.
for state in states:
    state_data = eval(f"{state}_results")
    state_data['total_cbg'] = compute_total(state_data, 'cbg', 'cbga')

# Create a scatter plot with color-coded points for each state
plt.figure(figsize=(9, 6))
for state in states:
    state_data = eval(f"{state}_results")
    plt.scatter(state_data['total_cbg'], state_data['total_thc'], alpha=0.5, color=colors[state], label=state.upper())
plt.title('Total THC to Total CBG in CA and FL Flower')
plt.xlabel('Total CBG (%)')
plt.ylabel('Total THC (%)')
plt.ylim(0)
plt.xlim(0)
plt.grid(True)
plt.legend(loc='upper right')
fig.set_facecolor('white')
plt.savefig('figures/total_thc_to_total_cbg.png', dpi=300, bbox_inches='tight')
plt.show()


#-----------------------------------------------------------------------
# Time series.
#-----------------------------------------------------------------------

# Convert date columns to datetime
ca_results['date_tested'] = pd.to_datetime(ca_results['date_tested'], format='mixed', errors='coerce')
fl_results['date_tested'] = pd.to_datetime(fl_results['date_tested'], format='mixed', errors='coerce')

# Define compounds of interest
compounds = cannabinoids + terpenes
del compounds[compounds.index('cbdva')]
del compounds[compounds.index('thcva')]
del compounds[compounds.index('cbca')]

# Resample data by month and take mean
ca_monthly_avg = ca_results.set_index('date_tested').resample('M')[compounds].mean()
fl_monthly_avg = fl_results.set_index('date_tested').resample('M')[compounds].mean()

# Plotting
for compound in compounds[3:]:
    plt.figure(figsize=(15, 10))
    plt.plot(ca_monthly_avg.index, ca_monthly_avg[compound], label=f"CA {compound}", linewidth=4)
    plt.plot(fl_monthly_avg.index, fl_monthly_avg[compound], '--', label=f"FL {compound}", linewidth=4)
    plt.title(f'{compound.upper()} Trend Over Time Averaged By Month')
    plt.xlabel('Date', fontsize=32)
    plt.ylabel('Average Concentration (%)', fontsize=32)
    plt.ylim(0)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.tight_layout()
    fig.set_facecolor('white')
    plt.savefig(f'figures/{compound}-trending.png', dpi=300, bbox_inches='tight')
    plt.show()


#-----------------------------------------------------------------------
# ANOVA.
#-----------------------------------------------------------------------



#-----------------------------------------------------------------------
# MANOVA.
#-----------------------------------------------------------------------




