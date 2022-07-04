"""
Residual Solvent Analysis for Concentrates in Washington State
Cannabis Data Science

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 7/28/2021
Updated: 7/28/2021
License: MIT License <https://opensource.org/licenses/MIT>
Data Sources:
    - WSLCB Traceability Data: https://lcb.app.box.com/s/fnku9nr22dhx04f6o646xv6ad6fswfy9?page=1
Resources:
    - WSLCB Traceability Licensee Data Guide: https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf
    - WSLCB Traceability Lab Data Guide: https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_TestingLabUser.pdf
"""
import pandas as pd

# Silence pandas errors (not the best practice)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    'concentrate_for_inhalation',
    'non-solvent_based_concentrate',
    'infused_mix',
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

# Look at solvents present in the various concentrate types.
for concentrate_type in concentrate_types:
    
    # Isolate type data.
    type_data = data.loc[data.intermediate_type == concentrate_type]  
    print('\n--------------')
    print(concentrate_type.replace('_', ' ').title())
    
    # Calculate the average for each solvent for the type.
    for solvent in solvents:
        solvent_name = solvent.split('_').pop(1).title()
        average = type_data[solvent].mean()
        if average:
            print('%s average: %.2f ppm' % (solvent_name, average))
        else:
            print('%s average: n/a' % solvent_name)
        
        
# Calculate failure rates for each concentrate type and each solvent.
limits = pd.read_excel('./data/wa_limits.xlsx', index_col=0)
for concentrate_type in concentrate_types:
    
    # Isolate type data.
    type_data = data.loc[data.intermediate_type == concentrate_type]  
    print('\n--------------')
    print(concentrate_type.replace('_', ' ').title())
    
    # Calculate the failure rate for the type.
    for solvent in solvents:
        limit = limits.loc[solvent].limit
        failure_rate = type_data.loc[type_data[solvent] > limit, solvent].count() /\
                   type_data[solvent].count() * 100
        solvent_name = solvent.split('_').pop(1).title()
        if average:
            print('%s failure rate: %.2f%%' % (solvent_name, failure_rate))
        else:
            print('%s failure rate: n/a' % solvent_name)


#--------------------------------------------------------------------------
# Bonus. Look at mycotoxin failure rates in residual solvents in concentrates.
#--------------------------------------------------------------------------

# Define mycotoxins
mycotoxins = [
    'mycotoxin_aflatoxins_ppb',
    'mycotoxin_ochratoxin_ppb',
]

limits = pd.read_excel('./data/wa_limits.xlsx', index_col=0)
for sample_type in sample_types:
    
    # Isolate type data.
    try:
        type_data = data.loc[data.intermediate_type == sample_type]  
        print('\n--------------')
        print(sample_type.replace('_', ' ').title())
    except AttributeError:
        continue
    
    # Calculate the failure rate for the type.
    for analyte in mycotoxins:
        limit = limits.loc[analyte].limit
        failure_rate = type_data.loc[type_data[analyte] > limit, analyte].count() /\
                   type_data[analyte].count() * 100
        name = analyte.split('_').pop(1).title()
        if average:
            print('%s failure rate: %.4f%%' % (name, failure_rate))
        else:
            print('%s failure rate: n/a' % name)

#--------------------------------------------------------------------------
# Bonus. Look at microbe failure rates in residual solvents in concentrates.
#--------------------------------------------------------------------------

# Define microbes
microbes = [
    'microbial_total_coliform_cfu_g',
    'microbial_bile_tolerant_cfu_g',
]

limits = pd.read_excel('./data/wa_limits.xlsx', index_col=0)
for sample_type in sample_types:
    
    # Isolate type data.
    try:
        type_data = data.loc[data.intermediate_type == sample_type]  
        print('\n--------------')
        print(sample_type.replace('_', ' ').title())
    except AttributeError:
        continue
    
    # Calculate the failure rate for the type.
    for analyte in microbes:
        limit = limits.loc[analyte].limit
        failure_rate = type_data.loc[type_data[analyte] > limit, analyte].count() /\
                   type_data[analyte].count() * 100
        name = analyte.split('_').pop(1).title()
        if average:
            print('%s failure rate: %.4f%%' % (name, failure_rate))
        else:
            print('%s failure rate: n/a' % name)
