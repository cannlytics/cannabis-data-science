"""
Cannabinoid Analysis | Cannabis Data Science

Author: Keegan Skeate<keegan@cannlytics.com>
Created: 7/21/2021
Updated: 7/28/2021
License: MIT License <https://opensource.org/licenses/MIT>
Data Sources:
    - WSLCB Traceability Data: https://lcb.app.box.com/s/fnku9nr22dhx04f6o646xv6ad6fswfy9?page=1
Resources:
    - WSLCB Traceability Licensee Data Guide: https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf
    - WSLCB Traceability Lab Data Guide: https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_TestingLabUser.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cannabinoid_histogram(series, title, y_label, x_label, filename):
    """Plot a decent looking histogram for cannabinoid results.
    Args:
        series (Pandas Series): A series of data to plot a histogram.
        title (str): The title of the histogram.
        y_label (str): The Y-axis label.
        x_label (str): The X-axis label.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    series.plot.kde(ax=ax, legend=False)
    series.plot.hist(density=True, ax=ax)
    plt.title(title, fontsize=21)
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_xlabel(x_label, fontsize=18)
    ax.grid(axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0)
    plt.show()
    plt.savefig(filename, bbox_inches='tight', dpi=300)

if __name__ == '__main__':

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

    #--------------------------------------------------------------------------
    # Rule 1. Look at the data.
    #--------------------------------------------------------------------------

    # Find all of the sample types.
    sample_types = list(data.intermediate_type.unique())

    # Identify all flower samples.
    flower_data = data.loc[(data.intermediate_type == 'flower') |
                        (data.intermediate_type == 'flower_lots')]
    print('Number of flower samples: %i' % len(flower_data))

    # Identify all flower samples with CBG.
    cbg_present = flower_data.loc[flower_data.cannabinoid_d8_thc_percent > 0]
    print('Number of flower samples with CBG: %i' % len(cbg_present))

    # Calculate the percent of samples that test for Delta-8 THC.
    percent_of_flower_with_delta_8 = len(delta_8_thc) / len(flower_data) * 100
    print('Percent of flower samples with Delta-8 THC: %.2f' % percent_of_flower_with_delta_8)

    # Exclude outliers?
    upper_bound = delta_8_thc.cannabinoid_d8_thc_percent.quantile(0.95)
    delta_8_thc = delta_8_thc.loc[delta_8_thc.cannabinoid_d8_thc_percent < upper_bound]

    # Plot a histogram of delta-8 THC in flower.
    cannabinoid_histogram(
        delta_8_thc['cannabinoid_d8_thc_percent'],
        title='Distribution of Delta-8 THC in Flower in Washington State',
        y_label='Density',
        x_label='Delta-8 THC Concentration',
        filename='distribution_of_delta_8_thc_in_wa_flower.png'
    )


    #--------------------------------------------------------------------------
    # Bonus: Analysis of solid edibles.
    #--------------------------------------------------------------------------

    # Identify solid edible samples.
    solid_edibles = data.loc[data.intermediate_type == 'solid_edible']
    print('Number of solid edible samples: %i' % len(solid_edibles))

    # FIXME: Standardize mg of THC if mg of THC is missing but percent is given.
    # mg_g = solid_edibles.apply(lambda row: row.cannabinoid_d9_thc_percent * 10 if \
    #                            row.cannabinoid_d9_thc_percent else \
    #                                row.cannabinoid_d9_thc_mg_g, axis=1)

    # Optional: Plot of mg of THC in edibles.
    # solid_edibles.cannabinoid_d9_thc_mg_g.hist()

    # Calculate THC to CBD ratio.
    solid_edibles['thc_to_cbd_ratio'] = solid_edibles.cannabinoid_d9_thc_mg_g / \
                                        solid_edibles.cannabinoid_cbd_mg_g

    # Drop N/A.
    solid_edibles['thc_to_cbd_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    thc_to_cbd_ratio_data = solid_edibles.loc[solid_edibles['thc_to_cbd_ratio'].notna()]

    # Drop outliers.
    upper_bound = thc_to_cbd_ratio_data.thc_to_cbd_ratio.quantile(0.95)
    thc_to_cbd_ratio_data = thc_to_cbd_ratio_data.loc[thc_to_cbd_ratio_data.thc_to_cbd_ratio < upper_bound]

    # Plot a histogram of the THC to CBD ratio.
    cannabinoid_histogram(
        thc_to_cbd_ratio_data['thc_to_cbd_ratio'],
        title='THC to CBD Ratio in Solid Edibles in Washington State',
        y_label='Density',
        x_label='THC to CBD Ratio',
        filename='thc_to_cbd_ratio_for_wa_solid_edibles.png'
    )

    # Calculate the mean and standard deviation of the THC to CBD ratio.
    thc_to_cbd_ratio_data['thc_to_cbd_ratio'].describe()
