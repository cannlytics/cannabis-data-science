"""
Calculate Cannabis Waste Statistics | Project

Authors: Keegan Skeate, Charles Rice
Contact: <keegan@cannlytics.com>
Created: Thu Apr 29 14:38:48 2021
License: GPLv3 License

Resources:
    https://github.com/ufosoftwarellc/cannabis_data_science/blob/main/wa/analytics/waste.ipynb
    https://pandas.pydata.org/pandas-docs/stable/user_guide/scale.html
    https://docs.dask.org/en/latest/dataframe.html
    https://examples.dask.org/dataframes/01-data-access.html
    https://www.youtube.com/watch?v=0eEsIA0O1iE
    
"""

#-----------------------------------------------------------------------
# Dask experimentation
#-----------------------------------------------------------------------

# import dask.dataframe as dd
# from dask.distributed import Client

# Create a Dask client.
# client = Client(
#     n_workers=1,
#     threads_per_worker=4,
#     processes=False,
#     memory_limit='2GB'
# )
# client

# batches_df = dd.read_csv(
#     directory + data_file,
#     sep = '\t',
#     encoding = 'utf-8',
#     usecols=cols,
#     dtype=col_dtypes,
#     parse_dates=
#     date_cols
# )


#-----------------------------------------------------------------------
# Create daily series from large data set.
# https://stackoverflow.com/questions/29334463/how-can-i-partially-read-a-huge-csv-file
#-----------------------------------------------------------------------

import pandas as pd

# Read in a chunk at a time.
chunksize = 10000

# Define column data types to reduce memory footprint.
col_dtypes = {
    'global_id': 'string',
    'mme_id': 'string',
    'user_id': 'string',
    'external_id': 'string',
    'uom': 'category',
    #'planted_at': 'string', depreciated
    #'created_by_mme_id', 'string', Every entry is nan
    'num_plants': 'int32',
    'status': 'category',
    'strain_id': 'string',
    'is_parent_batch': 'bool',
    'is_child_batch': 'bool',
    'type': 'category',
    'harvest_stage': 'category',
    #'qty_accumulated_waste': 'float32', depreciated
    'qty_packaged_flower': 'float32',
    'qty_packaged_by_product': 'float32',
    'area_id': 'string',
    'origin': 'category',
    #'qty_cure': 'float32' depreciated
    'plant_stage': 'category',
    'flower_dry_weight': 'float32',
    'waste': 'float32',
    'other_dry_weight': 'float32',
    'flower_wet_weight': 'float32',
    'other_wet_weight': 'float32'
}

# Parse date columns.
date_cols = [
    'created_at',
    'updated_at',
    #'planted_at', depreciated
    'harvested_at',
    'batch_created_at',
    'packaged_completed_at',
    'deleted_at',
    'harvested_end_at'
]

# Define columns to load to reduce memory footprint
# by only loading the columns needed.
cols = list(col_dtypes.keys()) + date_cols

# Read in the data.
directory = r'E:\cannlytics\data_archive\leaf\Batches_0/'
data_file = 'Batches_0.csv'
filename = directory + data_file
batches_df = pd.read_csv(
    filename,
    sep='\t',
    encoding='utf-16',
    usecols=cols,
    dtype=col_dtypes,
    parse_dates=date_cols,
    skiprows=range(1, 2000000),
    nrows=chunksize,
)

# Change negative values to postitive values.
# batches_df.loc[(batches_df.waste < 0), 'waste'] = batches_df.loc[(batches_df.waste < 0), 'waste'] * -1.0
batches_df['waste'] = batches_df['waste'].abs()

# Fill in missing values where waste is present but
# the plant stage is missing using the most common value.
batches_df.loc[(batches_df.plant_stage.isna() & batches_df.waste > 0), 'plant_stage'] = 'harvested'

# Find the amount of waste for harvested product.
wasted_harvested_df =  batches_df[(batches_df.plant_stage == 'harvested') & (batches_df.waste > 0)].copy()

# Group the data by producer and date.
waste_by_producer = wasted_harvested_df.groupby(['mme_id', 'update_date']).agg({'waste': 'sum'})

# Waste generated during growing
growing_waste_df = batches_df[(batches_df.plant_stage == 'growing') & (batches_df.waste > 0)].copy()
growing_waste_df['update_date'] = growing_waste_df.updated_at.dt.date
growing_waste_df = growing_waste_df.groupby(['mme_id', 'update_date']).agg({'waste': 'sum'})

# Flower waste
wasted_flower_df = batches_df[(batches_df.plant_stage == 'flower') & (batches_df.waste > 0)].copy()
wasted_flower_df['update_date'] = wasted_flower_df.updated_at.dt.date
wasted_flower_df = wasted_flower_df.groupby(['mme_id', 'update_date']).agg({'waste': 'sum'})

# Seedling waste
seedling_waste_df = batches_df[(batches_df.plant_stage == 'seedling') & (batches_df.waste > 0)].copy()
seedling_waste_df['update_date'] = seedling_waste_df.updated_at.dt.date
seedling_waste_df = seedling_waste_df.groupby(['mme_id', 'update_date']).agg({'waste': 'sum'})


# TODO: Save all waste data


#-----------------------------------------------------------------------
# ARMA forecasting
# https://machinelearningmastery.com/make-sample-forecasts-arima-python/
#-----------------------------------------------------------------------

# from statsmodels.tsa.arima.model import ARIMA

# # fit model
# model = ARIMA(differenced, order=(7,0,1))
# model_fit = model.fit()
# # print summary of fit model
# print(model_fit.summary())

# # one-step out-of sample forecast
# forecast = model_fit.forecast()[0]

# # one-step out of sample forecast
# start_index = '1990-12-25'
# end_index = '1990-12-25'
# forecast = model_fit.predict(start=start_index, end=end_index)

# # multi-step out-of-sample forecast
# forecast = model_fit.forecast(steps=7)


#-----------------------------------------------------------------------
# Auto-ARIMA forecasting
# https://towardsdatascience.com/time-series-forecasting-using-auto-arima-in-python-bb83e49210cd
# https://github.com/alkaline-ml/pmdarima/blob/master/examples/quick_start_example.ipynb
#-----------------------------------------------------------------------

# from pmdarima.arima import auto_arima

# arima_model =  auto_arima(train, start_p=0, d=1, start_q=0, 
#                           max_p=5, max_d=5, max_q=5, start_P=0, 
#                           D=1, start_Q=0, max_P=5, max_D=5,
#                           max_Q=5, m=12, seasonal=True, 
#                           error_action='warn',trace = True,
#                           supress_warnings=True,stepwise = True,
#                           random_state=20,n_fits = 50 )

# Summary of the model
# arima_model.summary()

# Predictions
# prediction = pd.DataFrame(arima_model.predict(n_periods = 20),index=test.index)
# prediction.columns = ['predicted_sales']
# prediction

# Plot
# plt.figure(figsize=(8,5))
# plt.plot(train,label="Training")
# plt.plot(test,label="Test")
# plt.plot(prediction,label="Predicted")
# plt.legend(loc = 'Left corner')
# plt.show()
