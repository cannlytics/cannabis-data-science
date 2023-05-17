"""
Benford's Law
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 12/21/2022
Updated: 12/21/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Source:

    - WSLCB
    URL: <https://lcb.box.com/s/xseghpsq2t4i1musxj6mgd7b8rhxe7bm>

"""
# Standard imports:
import gc

# External imports:
from cannlytics.data.ccrs import CCRS
from cannlytics.data.ccrs.constants import (
    CCRS_ANALYTES,
    CCRS_ANALYSES,
    CCRS_DATASETS,
)
from cannlytics.utils import camel_to_snake
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

# Specify where your data lives.
DATA_DIR = 'D:\\data\\washington\\ccrs-2022-11-22\\ccrs-2022-11-22'

# Initialize a CCRS client.
ccrs = CCRS(data_dir=DATA_DIR)

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#------------------------------------------------------------------------------
# Data curation
#------------------------------------------------------------------------------

# Read the lab result data.
def read_lab_results(datafile, limit=None):
    """Read lab results into a well-formatted DataFrame,
    mapping analyses from `test_name` into `key`, `type`, `units`.
    Future work: Allow users to specify which fields to read.
    """
    data = pd.read_csv(
        datafile,
        low_memory=False,
        nrows=limit,
        encoding='utf-16',
        sep='\t',
        parse_dates=CCRS_DATASETS['lab_results']['date_fields'],
        usecols=CCRS_DATASETS['lab_results']['fields'],
    )
    data.columns = [camel_to_snake(x) for x in data.columns]
    analyte_data = data['test_name'].map(CCRS_ANALYTES).values.tolist()
    data = data.join(pd.DataFrame(analyte_data))
    data['type'] = data['type'].map(CCRS_ANALYSES)
    return data


# Read all CCRS lab results.
results = pd.DataFrame()
datafiles = [
    f'{DATA_DIR}/LabResult_0/LabResult_0/LabResult_0.csv',
    f'{DATA_DIR}/LabResult_1/LabResult_1/LabResult_1.csv',
]
for datafile in datafiles:
    data = read_lab_results(datafile)
    results = pd.concat([results, data])

# Garbage cleaning.
del data
gc.collect()

# TODO: Exclude any test lab results.


#------------------------------------------------------------------------------
# Data exploration.
#------------------------------------------------------------------------------

# Plot total THC distribution.
thc = results.loc[(results['key'] == 'total_thc') & (results['units'] == 'percent')]
thc = thc.assign(value=pd.to_numeric(thc['test_value'], errors='coerce'))
thc.loc[(thc.value > 0) & (thc.value < 100)].value.hist(bins=100)
plt.title('Distribution of Total THC in WA Cannabis in 2022')
plt.show()

# Plot total CBD distribution.
cbd = results.loc[(results['key'] == 'total_cbd') & (results['units'] == 'percent')]
cbd = cbd.assign(value=pd.to_numeric(cbd['test_value'], errors='coerce'))
cbd.loc[(cbd.value > 0) & (cbd.value < 2)].value.hist(bins=100)
plt.title('Distribution of Total CBD in WA Cannabis in 2022')
plt.show()

# Plot moisture content distribution.
moisture = results.loc[(results['key'] == 'moisture_content')]
moisture = moisture.assign(value=pd.to_numeric(moisture['test_value'], errors='coerce'))
moisture.loc[(moisture.value > 0) & (moisture.value < 20)].value.hist(bins=100)
plt.title('Distribution of Moisture Content in WA Cannabis in 2022')
plt.show()

# Plot water activity distribution.
aw = results.loc[(results['key'] == 'water_activity')]
aw = aw.assign(value=pd.to_numeric(aw['test_value'], errors='coerce'))
aw.loc[(aw.value > 0) & (aw.value < 1)].value.hist(bins=100)
plt.title('Distribution of Water Activity in WA Cannabis in 2022')
plt.show()


#------------------------------------------------------------------------------
# Data analysis: Test if various analytes satisfy Benford's Law on a
# lab by lab basis.
#------------------------------------------------------------------------------

def first_decimal_place(value):
    """Return the first decimal place of a value."""
    try:
        return int(str(float(value)).split('.')[-1][0])
    except:
        return 0

# Identify the digit of the first decimal place.
thc['first_decimal_place'] = thc.value.apply(first_decimal_place)
cbd['first_decimal_place'] = cbd.value.apply(first_decimal_place)
moisture['first_decimal_place'] = moisture.value.apply(first_decimal_place)
aw['first_decimal_place'] = aw.value.apply(first_decimal_place)

# Plot the distributions.
thc.loc[thc.first_decimal_place > 0]['first_decimal_place'].hist(bins=100)
plt.show()

cbd.loc[cbd.first_decimal_place > 0]['first_decimal_place'].hist(bins=100)
plt.show()

moisture.loc[moisture.first_decimal_place > 0]['first_decimal_place'].hist(bins=100)
plt.show()

aw.loc[aw.first_decimal_place > 0]['first_decimal_place'].hist(bins=100)
plt.show()

# Plot the distributions by lab.
labs = (results.lab_licensee_id.unique())
analyte = cbd
for lab in labs:
    lab_thc = analyte.loc[analyte.lab_licensee_id == lab]
    analyte.loc[analyte.first_decimal_place > 0]['first_decimal_place'].hist(bins=100)
    plt.title('Distribution of Digits by Lab %s' % lab)
    plt.show()


#------------------------------------------------------------------------------
# Data visualization.
#------------------------------------------------------------------------------

# Define a series to plot.
series = thc
x_min, x_max = 0, 100
name = 'Total THC'
units = 'Percent'

# Plot a ridge plot.
series = series.loc[(series.value > x_min) & (series.value < x_max)]
palette = sns.color_palette(palette='coolwarm', n_colors=len(labs))
grid = sns.FacetGrid(
    series,
    row='lab_licensee_id',
    hue='lab_licensee_id',
    aspect=3,
    height=4,
    legend_out=False,
)
grid.map(
    sns.kdeplot,
    'value',
    clip_on=False,
    shade=True,
    alpha=0.7,
    lw=1,
)
grid.set_axis_labels(units, '')
grid.fig.subplots_adjust(top=0.95)
grid.fig.suptitle(f'Distribution of {name} Measured by Lab in WA in 2022')
plt.xlim(x_min, x_max)
plt.show()

# Print out mean and variance by lab.
for lab in labs:
    lab_series = series.loc[series.lab_licensee_id == lab].value
    lab_series = lab_series.loc[(lab_series > 0) & (lab_series < 40)]
    mean = lab_series.mean()
    variance = lab_series.var()
    print('Lab:', lab, 'Count:', len(lab_series), 'Mean:', round(mean, 2), 'Variance:', round(variance, 2))
