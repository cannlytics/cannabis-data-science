"""
Valid Lab Results | Cannabis Data Science

Authors:
    UFO Software, LLC
    Keegan Skeate <keegan@cannlytics.com>
Created: Thursday, July 29, 2021 21:51
Updated: 8/10/2021
License GPLv3+: GNU GPL version 3 or later https://gnu.org/licenses/gpl.html This is free software: you are free to change and redistribute it. There is NO WARRANTY, to the extent permitted by law.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def get_lab_results_df():
    # reduce the size of the dataframe's memory footprint by specifying data types
    # comment out columns you are not using to further decrease the memory footprint
    col_dtypes = {'global_id' : 'string',
                 #'#mme_id' : 'category',
                 #'user_id' : 'string',
                 #'external_id' : 'string',
                 #'inventory_id' : 'string',
                 'status' : 'category',
                 #'testing_status' : 'category',
                 #'batch_id' : 'string',
                 #'parent_lab_result_id' : 'string',
                 #'og_parent_lab_result_id' : 'string',
                 #'copied_from_lab_id' : 'string',
                 #'lab_user_id' : 'string',
                 'type' : 'category',
                 #'foreign_matter' : 'bool',
                 #'moisture_content_percent' : 'float16', #if you are not using Dask change this to float16
                 #'growth_regulators_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'cannabinoid_status' : 'category',
                 #'cannabinoid_editor' : 'float32', #if you are not using Dask change this to float16
                 #'cannabinoid_d9_thca_percent': 'float16',
                 #'cannabinoid_d9_thca_mg_g' : 'float16', #if you are not using Dask change this to float16
                 #'cannabinoid_d9_thc_percent' : 'float16', #if you are not using Dask change this to float16
                 #'cannabinoid_d9_thc_mg_g' : 'float16', #if you are not using Dask change this to float16
                 #'cannabinoid_d8_thc_percent' : 'float16', #if you are not using Dask change this to float16
                 #'cannabinoid_d8_thc_mg_g' : 'float16', #if you are not using Dask change this to float16
                 #'cannabinoid_cbd_percent' : 'float16', #if you are not using Dask change this to float16
                 #'cannabinoid_cbd_mg_g' : 'float16', #if you are not using Dask change this to float16
                 #'cannabinoid_cbda_percent' : 'float16', #if you are not using Dask change this to float16
                 #'cannabinoid_cbda_mg_g' : 'float16', #if you are not using Dask change this to float16
                 #'cannabinoid_cbdv_percent' : 'float16', #if you are not using Dask change this to float16
                 #'cannabinoid_cbg_percent' : 'float16', #if you are not using Dask change this to float16
                 #'cannabinoid_cbg_mg_g' : 'float16', #if you are not using Dask change this to float16
                 #'terpenoid_pinene_percent' : 'float16', #if you are not using Dask change this to float16
                 #'terpenoid_pinene_mg_g' : 'float16', #if you are not using Dask change this to float16
                 #'microbial_status' : 'category',
                 #'microbial_editor' : 'string',
                 #'microbial_bile_tolerant_cfu_g' : 'float16', #if you are not using Dask change this to float16
                 #'microbial_pathogenic_e_coli_cfu_g' : 'float16', #if you are not using Dask change this to float16
                 #'microbial_salmonella_cfu_g' : 'float16', #if you are not using Dask change this to float16
                 #'mycotoxin_status' : 'category',
                 #'mycotoxin_editor' : 'string',
                 #'mycotoxin_aflatoxins_ppb' : 'float16', #if you are not using Dask change this to float16
                 #'mycotoxin_ochratoxin_ppb' : 'float16', #if you are not using Dask change this to float16
                 #'metal_status' : 'category',
                 #'metal_editor': 'string',
                 #'metal_arsenic_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'metal_cadmium_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'metal_lead_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'metal_mercury_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_status' : 'category',
                 #'pesticide_editor' : 'string',
                 #'pesticide_abamectin_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_acequinocyl_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_bifenazate_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_cyfluthrin_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_cypermethrin_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_etoxazole_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_flonicamid_ppm' : 'float', #if you are not using Dask change this to float16
                 #'pesticide_fludioxonil_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_imidacloprid_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_myclobutanil_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_spinosad_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_spirotetramet_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_thiamethoxam_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_trifloxystrobin_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_status' : 'category',
                 #'solvent_editor' : 'string',
                 #'solvent_butanes_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_heptane_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_propane_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'notes' : 'float32', #if you are not using Dask change this to float16
                 #'thc_percent' : 'float16', #if you are not using Dask change this to float16
                 'intermediate_type' : 'category',
                 #'moisture_content_water_activity_rate' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_acetone_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_benzene_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_cyclohexane_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_chloroform_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_dichloromethane_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_ethyl_acetate_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_hexanes_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_isopropanol_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_methanol_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_pentanes_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_toluene_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'solvent_xylene_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_acephate_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_acetamiprid_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_aldicarb_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_azoxystrobin_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_bifenthrin_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_boscalid_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_carbaryl_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_carbofuran_ppm' : 'float16', #if you are not using Dask change this to float16
                 #'pesticide_chlorantraniliprole_ppm' : 'float16' #if you are not using Dask change this to float16
                }

    date_cols = ['created_at',
                 #'deleted_at',
                 #'updated_at',
                 #'tested_at',
                 #'received_at' deprecated
                ]

    # combine the column names to load only the columns you are using
    cols = list(col_dtypes.keys()) + date_cols

    lab_results_df = pd.read_csv(file_path / 'LabResults_0.csv', sep = '\t', encoding = 'utf-16', usecols = cols, dtype = col_dtypes, parse_dates = date_cols, skipinitialspace = True)
    # all the datasets in the WA data use global_id but it has different meaning for each dataset which makes the data difficult to understand and causes issues with Pandas when trying to perform operations on more than one dataframe.
    lab_results_df.rename(columns={'global_id':'lab_results_id'}, inplace=True)
    # dataframe with rows from the origanal dataframe where the lab_results_id is nan
    null_lab_results_id_df = lab_results_df.loc[lab_results_df.lab_results_id.isna()]
    # drop rows with nan lab_results_ids
    lab_results_df.dropna(subset=['lab_results_id'], inplace=True)
    # exract the lab_id from the lab_results_id
    # lab_ids are embedded in the lab_results_id in the form "WAL##."
    lab_results_df['lab_id'] = lab_results_df.lab_results_id.map(lambda x: x[x.find('WAL') : x.find('.')])
    # dataframe with the rows that did not contain a valid lab_id in the lab_results_id
    # The lab_results_id does not contain a substring in the form of "WA##."
    invalid_lab_id_df = lab_results_df.loc[(lab_results_df.lab_id == '')]
    #remove the rows with invalid lab_ids from the dataframe
    lab_results_df = lab_results_df.loc[~(lab_results_df.lab_id == '')]
 
    return lab_results_df, null_lab_results_id_df, invalid_lab_id_df


if __name__ == '__main__':

    # change the file path to match where your data is stored
    file_path = Path('../.datasets')
    pd.set_option('display.max_columns', None)
    pd.options.display.float_format = "{:.2f}".format
    
    # Read in the data
    lab_results_df, null_lab_results_id_df, invalid_lab_id_df = get_lab_results_df()
