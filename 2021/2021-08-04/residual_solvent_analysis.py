"""
Residual Solvent Analysis of Concentrates in Washington State
Cannabis Data Science

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 7/28/2021
Updated: 8/4/2021
License: MIT License <https://opensource.org/licenses/MIT>
Data Sources:
    - WSLCB Traceability Data: https://lcb.app.box.com/s/fnku9nr22dhx04f6o646xv6ad6fswfy9?page=1
Resources:
    - WSLCB Traceability Licensee Data Guide: https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf
    - WSLCB Traceability Lab Data Guide: https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_TestingLabUser.pdf
"""

# External imports
import pandas as pd
import matplotlib.pyplot as plt

# Silence pandas errors (not the best practice)
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

#--------------------------------------------------------------------------
# Read in the data. (Download the data to a .datasets folder in the root directory.)
#--------------------------------------------------------------------------
    
# Import the data.
data = pd.read_csv(
    '../.datasets/LabResults_0.csv',
    encoding='utf-16',
    sep='\t',
)
data = data.sort_index()
print('Number of observations: %i' % len(data))

# Find all of the sample types.
sample_types = list(data.intermediate_type.unique())

#--------------------------------------------------------------------------
# Rule 1. Look at residual solvents in concentrates.
#--------------------------------------------------------------------------

# Define concentrate types.
concentrate_types = [
    'hydrocarbon_concentrate',
    'non-solvent_based_concentrate',
    'co2_concentrate',
    'food_grade_solvent_concentrate',
    'ethanol_concentrate',
]

# Define residual solvents.
solvents = [
    'solvent_acetone_ppm',
    'solvent_benzene_ppm',
    'solvent_cyclohexane_ppm',
    'solvent_chloroform_ppm',
    'solvent_dichloromethane_ppm',
    'solvent_ethyl_acetate_ppm',
    'solvent_hexanes_ppm',
    'solvent_isopropanol_ppm',
    'solvent_methanol_ppm',
    'solvent_pentanes_ppm',
    'solvent_toluene_ppm',
    'solvent_xylene_ppm',
    'solvent_heptanes_ppm',
    'solvent_butanes_ppm',
    'solvent_heptane_ppm',
    'solvent_propane_ppm',
]

#--------------------------------------------------------------------------
# Calculate failure rates
#--------------------------------------------------------------------------       
        
# Calculate failure rates for each solvent.
limits = pd.read_excel('./data/residual_solvents_limits.xlsx', index_col=0)
    
# Calculate the failure rate for the type.
for analyte in solvents:
    limit_wa = limits.loc[analyte].limit_wa
    limit_ok = limits.loc[analyte].limit_ok
    limit_fl = limits.loc[analyte].limit_fl
    
    fails_wa = data.loc[data[analyte] > limit_wa, analyte]
    fails_ok = data.loc[data[analyte] > limit_ok, analyte]
    fails_fl = data.loc[data[analyte] > limit_fl, analyte]
    
    if len(fails_wa) != len(fails_ok) or len(fails_wa) != len(fails_fl):
        print('\n--------------------------')
        print('Different failure rates: %s' % analyte)
        print('Number of failures in Washington: %i' % len(fails_wa))
        print('Number of failures in Oklahoma: %i' % len(fails_ok))
        print('Number of failures in Florida: %i' % len(fails_fl))

#--------------------------------------------------------------------------
# Deep dive into a specific analyte.
#--------------------------------------------------------------------------  

# Define a specific analyte.
analyte = 'solvent_acetone_ppm'
analyte_name = 'Acetone'
limit_wa = limits.loc[analyte].limit_wa
limit_ok = limits.loc[analyte].limit_ok
limit_fl = limits.loc[analyte].limit_fl

# Look at a specific analyte.
fails_wa = data.loc[(data[analyte] > limit_wa) & (data[analyte] < 20000)]
fails_ok = data.loc[(data[analyte] > limit_ok) & (data[analyte] < 20000)]
fails_fl = data.loc[(data[analyte] > limit_fl) & (data[analyte] < 20000)]

# Restrict the data to (rational) measurements.
upper_range = data.loc[(data[analyte] > 0) & (data[analyte] < 20000)]

#--------------------------------------------------------------------------
# Plot the analyte observations.
#--------------------------------------------------------------------------  

# Create a figure.
fig, ax = plt.subplots(figsize=(14, 8))

# Plot observations over time.
plt.scatter(
    pd.to_datetime(upper_range.updated_at),
    upper_range[analyte],
    alpha=0.5
)

# Plot limits.
plt.axhline(y=limit_wa, color='#ffa600', linestyle='-', label='WA limit')
plt.axhline(y=limit_ok, color='#FF9500', linestyle='-', label='OK limit')
plt.axhline(y=limit_fl, color='#e53a23ff', linestyle='-', label='FL limit')

# Show the Legend
ax.legend(loc='upper left', fontsize=14)

# Show the plot.
plt.title('%s Measurements in Washington State' % analyte_name, fontsize=21)
ax.set_ylabel('%s (ppm)' % analyte_name, fontsize=18)
ax.set_xlabel('Tested At', fontsize=18)
ax.grid(axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
fig.savefig('%s_observations.png' % analyte, bbox_inches='tight', dpi=300)

#--------------------------------------------------------------------------
# Plot all analyte observations.
#-------------------------------------------------------------------------- 

def plot_residual_solvent(data, limits, analyte, analyte_name, x_axis='updated_at'):
    """Plot residual solvent observations."""
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.scatter(
        pd.to_datetime(data[x_axis]),
        data[analyte],
        alpha=0.5
    )
    
    # Identify the limits.
    limit_wa = limits.loc[analyte].limit_wa
    limit_ok = limits.loc[analyte].limit_ok
    limit_fl = limits.loc[analyte].limit_fl

    # Plot limits.
    plt.axhline(y=limit_wa, color='#ffa600', linestyle='-', label='WA limit')
    plt.axhline(y=limit_ok, color='#FF9500', linestyle='-', label='OK limit')
    plt.axhline(y=limit_fl, color='#e53a23ff', linestyle='-', label='FL limit')
    
    # Show the Legend
    ax.legend(loc='upper left', fontsize=14)
    
    # Show the plot.
    plt.title('%s Measurements in Washington State' % analyte_name, fontsize=21)
    ax.set_ylabel('%s (ppm)' % analyte_name, fontsize=18)
    ax.set_xlabel('Tested At', fontsize=18)
    ax.grid(axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    fig.savefig('./figures/%s_observations.png' % analyte, bbox_inches='tight', dpi=300)

# Plot all occurrences of residual solvents and their limits.
for analyte in solvents:
    observations = data.loc[(data[analyte] > 0) & (data[analyte] < 20000)]
    analyte_name = analyte.split('_')[1].title()
    plot_residual_solvent(observations, limits, analyte, analyte_name)
