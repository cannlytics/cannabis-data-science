"""
Draft Material.
Cannabis Data Science Meetup Group
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 1/11/2022
Updated: 1/19/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""


# # Define lab datasets.
# lab_datasets = ['LabResults_0', 'LabResults_1', 'LabResults_2']

# # Specify lab result fields.
# lab_result_fields = {
#     'global_id' : 'string',
#     'mme_id' : 'category',
#     'intermediate_type' : 'category',
#     'status' : 'category',
#     'inventory_id' : 'string',
#     'cannabinoid_status' : 'category',
#     'cannabinoid_d9_thca_percent': 'float16',
#     'cannabinoid_d9_thca_mg_g' : 'float16',
#     'cannabinoid_d9_thc_percent' : 'float16',
#     'cannabinoid_d9_thc_mg_g' : 'float16',
#     'cannabinoid_d8_thc_percent' : 'float16',
#     'cannabinoid_d8_thc_mg_g' : 'float16',
#     'cannabinoid_cbd_percent' : 'float16',
#     'cannabinoid_cbd_mg_g' : 'float16',
#     'cannabinoid_cbda_percent' : 'float16',
#     'cannabinoid_cbda_mg_g' : 'float16',
#     'cannabinoid_cbdv_percent' : 'float16',
#     'cannabinoid_cbg_percent' : 'float16',
#     'cannabinoid_cbg_mg_g' : 'float16',
#     'solvent_status' : 'category',
#     'solvent_acetone_ppm' : 'float16',
#     'solvent_benzene_ppm' : 'float16',
#     'solvent_butanes_ppm' : 'float16',
#     'solvent_chloroform_ppm' : 'float16',
#     'solvent_cyclohexane_ppm' : 'float16',
#     'solvent_dichloromethane_ppm' : 'float16',
#     'solvent_ethyl_acetate_ppm' : 'float16',
#     'solvent_heptane_ppm' : 'float16',
#     'solvent_hexanes_ppm' : 'float16',
#     'solvent_isopropanol_ppm' : 'float16',
#     'solvent_methanol_ppm' : 'float16',
#     'solvent_pentanes_ppm' : 'float16',
#     'solvent_propane_ppm' : 'float16',
#     'solvent_toluene_ppm' : 'float16',
#     'solvent_xylene_ppm' : 'float16',
#     #'foreign_matter' : 'bool',
#     #'microbial_status' : 'category',
#     #'microbial_bile_tolerant_cfu_g' : 'float16',
#     #'microbial_pathogenic_e_coli_cfu_g' : 'float16',
#     #'microbial_salmonella_cfu_g' : 'float16',
#     #'moisture_content_percent' : 'float16',
#     #'moisture_content_water_activity_rate' : 'float16',
#     #'mycotoxin_status' : 'category',
#     #'mycotoxin_aflatoxins_ppb' : 'float16',
#     #'mycotoxin_ochratoxin_ppb' : 'float16',
#     #'thc_percent' : 'float16',
#     #'notes' : 'float32',
#     # 'testing_status' : 'category',
#     # 'type' : 'category',
#     #'batch_id' : 'string',
#     #'parent_lab_result_id' : 'string',
#     #'og_parent_lab_result_id' : 'string',
#     #'copied_from_lab_id' : 'string',
#     #'external_id' : 'string',
#     #'lab_user_id' : 'string',
#     #'user_id' : 'string',
#     #'cannabinoid_editor' : 'float32',
#     #'microbial_editor' : 'string',
#     #'mycotoxin_editor' : 'string',
#     #'solvent_editor' : 'string',
# }
# lab_result_date_fields = ['created_at']
# lab_result_columns = list(lab_result_fields.keys()) + lab_result_date_fields

# # Read in the lab result data.
# shards = []
# for dataset in lab_datasets:
#     lab_data = pd.read_csv(
#         f'../.datasets/{dataset}.csv',
#         sep='\t',
#         encoding='utf-16',
#         usecols=lab_result_columns,
#         dtype=lab_result_fields,
#         parse_dates=lab_result_date_fields,
#         nrows=1000,
#     )
#     shards.append(lab_data)
#     del lab_data
#     gc.collect()

# # Aggregate lab data.
# lab_results = pd.concat(shards)
# del shards
# gc.collect()

# # Sort the data, removing null observations.
# lab_results.dropna(subset=['global_id'], inplace=True)
# lab_results.index = lab_results['global_id']
# lab_results = lab_results.sort_index()

# # Define a lab ID for each observation and remove attested lab results.
# lab_results['lab_id'] = lab_results['global_id'].map(lambda x: x[x.find('WAL'):x.find('.')])
# lab_results = lab_results.loc[lab_results.lab_id != '']

#------------------------------------------------------------------------------
# Combine with licensee data.
#------------------------------------------------------------------------------

# # Specify the licensee fields.
# licensee_fields = {
#     'global_id' : 'string',
#     'name': 'string',
#     'type': 'string',
#     # 'code': 'string',
#     # 'address1': 'string',
#     # 'address2': 'string',
#     # 'city': 'string',
#     # 'state_code': 'string',
#     # 'postal_code': 'string',
#     # 'country_code': 'string',
#     # 'phone': 'string',
#     # 'external_id': 'string',
#     # 'certificate_number': 'string',
#     # 'is_live': 'bool',
#     # 'suspended': 'bool',
# }
# licensee_date_fields = [
#     'created_at', # No records if issued before 2018-02-21.
#     # 'updated_at',
#     # 'deleted_at',
#     # 'expired_at',
# ]
# licensee_columns = list(licensee_fields.keys()) + licensee_date_fields

# # # Read in the licensee data.
# licensees = pd.read_csv(
#     '../.datasets/Licensees_0.csv',
#     sep='\t',
#     encoding='utf-16',
#     usecols=licensee_columns,
#     dtype=licensee_fields,
#     parse_dates=licensee_date_fields,
# )

# # Combine the data sets.
# licensees.rename(columns={
#     'global_id': 'licensee_id',
#     'created_at': 'license_created_at',
# }, inplace=True)
# del licensees['created_at']
# data = pd.merge(
#     left=lab_results,
#     right=licensees,
#     how='left',
#     left_on='mme_id',
#     right_on='licensee_id'
# )

#------------------------------------------------------------------------------
# Optional: Combine with inventory data.
#------------------------------------------------------------------------------

# # Specify the inventories fields.
# inventory_fields = {
#     'global_id' : 'string',
#     'strain_id': 'string',
#     'inventory_type_id': 'string',
#     # 'mme_id': 'string',
#     # 'user_id': 'string',
#     # 'external_id': 'string',
#     # 'area_id': 'string',
#     # 'batch_id': 'string',
#     # 'lab_result_id': 'string',
#     # 'lab_retest_id': 'string',
#     # 'is_initial_inventory': 'bool',
#     # 'created_by_mme_id': 'string',
#     # 'qty': 'float16',
#     # 'uom': 'string',
#     # 'additives': 'string',
#     # 'serving_num': 'float16',
#     # 'sent_for_testing': 'bool',
#     # 'medically_compliant': 'string',
#     # 'legacy_id': 'string',
#     # 'lab_results_attested': 'int',
#     # 'global_original_id': 'string',
# }
# inventory_date_fields = [
#     # 'created_at', # No records if issued before 2018-02-21.
#     # 'updated_at',
#     # 'deleted_at',
#     # 'inventory_created_at',
#     # 'inventory_packaged_at',
#     # 'lab_results_date',
# ]
# inventory_columns = list(inventory_fields.keys()) + inventory_date_fields

# # # Read in the licensee data.
# inventories = pd.read_csv(
#     '../.datasets/Inventories_0.csv',
#     sep='\t',
#     encoding='utf-16',
#     nrows=1000,
#     usecols=inventory_columns,
#     dtype=inventory_fields,
#     parse_dates=inventory_date_fields,
# )
# print(inventories)



# TODO: Match with inventories (by inventory_id -> global_id) data to get:

# TODO: 1. Read in a chunk of inventories data.
# fp = open('../.datasets/Inventories_0.csv')
# row_count = 0
# pos = {0: 0}
# line = fp.readline()
# while line:
#     row_count += 1
#     pos[row_count] = fp.tell()
#     line = fp.readline()
# fp.close()
# print('Lines:')

# chunksize = row_count // 30
# wanted_plts = [1,5,10,15,20,25,30]
# for i in wanted_plts:
#     fp.seek(pos[i*chunksize])  # this will bring you to the first line of the desired chunk
#     obj = pd.read_csv(
#         fp,
#         chunksize=chunksize,
#         sep='\t',
#         encoding='utf-16',
#         usecols=licensee_columns,
#         dtype=licensee_column_types,
#         parse_dates=licensee_date_columns,
#         # names=[!!<column names you have>!!]
#     )  # read your chunk lazily
#     # df = obj.get_chunk()  # convert to DataFrame object
#     print(df.head())
#     # plt.plot(df["Variable"]) # do something

# fp.close()  # Don't forget to close the file when finished.

    # TODO: 2. Process the chunk:
    # Iterate over lab results, finding matching inventory.
    # for index, row in lab_results.iterrows():
        # inventory_id = row['inventory_id]
        # result_inventory = inventories.loc[inventories['global_id'] == inventory_id]

        # Identify global_strain_id, global_inventory_type_id
        # labResults[0]['strain_name']?
        # Identify additives?
        # Add up qty, packed_qty, cost, value, uom?

        # Set the new variables in the lab_result observation.

# TODO: Save all of the aggregated data!
# lab_results.to_csv('./data/enhanced_lab_results.csv')



# Use global_batch_id to get strain_name?

# Optional: Get inventory type ID e.g WAxxx.TYxxx to get used_butane
# - name, description, ingredients, allergens, contains
# - net_weight, packed_qty, cost, value, uom
# - weight_per_unit_in_grams

# Get global_strain_id -> Match with strain data global_id to get:
# - name

#------------------------------------------------------------------------------
# SCRAP: Parsing inventories.
#------------------------------------------------------------------------------

# Assign values to lab results.
# for index, row in inventories.iterrows():
#     s = lab_results.loc[lab_results.global_for_inventory_id == row.global_id]
#     if len(s):
#         lab_results.loc[row.global_id , 'inventory_type_id'] = row.qty
#         lab_results.loc[row.global_id , 'qty'] = row.qty
#         lab_results.loc[row.global_id , 'uom'] = row.qty
#         print('Matched', s[0].global_for_inventory_id, 'to', row.global_id)
#         break

# SCRAP: Parsing inventory types.

# # Define inventory type fields.
# inventory_type_rows = 57_016_229
# inventory_type_fields = {
#     'global_id': 'string',
#     'name': 'string',
# }

# # Read inventory type names in chunks.
# chunk_size = 1_000_000
# read_rows = 0
# skiprows = None
# inventory_names = []
# while read_rows < inventory_type_rows:

#     if read_rows >= 5_000_000:
#         break

#     if read_rows:
#         skiprows = [i for i in range(1, read_rows)]

#     # Read in inventory types chunk.
#     inventory_types = pd.read_csv(
#         '../.datasets/InventoryTypes_0.csv',
#         sep='\t',
#         encoding='utf-16',
#         usecols=list(inventory_type_fields.keys()),
#         dtype=inventory_type_fields,
#         index_col='global_id',
#         skiprows=skiprows,
#         nrows=chunk_size,
#     )
#     inventory_types.rename(columns={'name': 'inventory_name'}, inplace=True)

#     # Keep the necessary fields.
#     inventory_names.append(inventory_types)
#     del inventory_types
#     gc.collect()

#     read_rows += chunk_size
#     print('Read:', read_rows)

# # Create a small file of inventory type names.
# inventory_type_names = pd.concat(inventory_names)
# inventory_type_names.to_csv('../.datasets/inventory_type_names.csv')


# SCRAP: Get file size.

# def get_blocks(files, size=65536):
#     """Get a block of a file by the given size."""
#     while True:
#         block = files.read(size)
#         if not block: break
#         yield block

# def get_number_of_rows(file_name, encoding='utf-16', errors='ignore'):
#     """
#     Read the number of lines in a large file.
#     Credit: glglgl, SU3 <https://stackoverflow.com/a/9631635/5021266>
#     License: CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0/>
#     """
#     with open(file_name, 'r', encoding=encoding, errors=errors) as f:
#         count = sum(bl.count('\n') for bl in get_blocks(f))
#         print('Number of rows:', count)
#         return count

# SCRAP: Reading in chunks of data.

# class ChunkReader:
#     """
#     Credit: Valentino <https://stackoverflow.com/a/56408765>
#     License: CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>
#     """

#     def __init__(self, filename, chunksize, n):
#         self.fo = open(filename)
#         self.chs = chunksize
#         self.skiplines = self.chs * n
#         self.header = next(self.fo)

#     def getchunk(self):
#         ll = list(islice(self.fo, self.chs))
#         if len(ll) == 0:
#             raise StopIteration
#         dd = list(islice(self.fo, self.skiplines))
#         return self.header + ''.join(ll)

#     def __iter__(self):
#         return self

#     def __next__(self):
#         return io.StringIO(self.getchunk())

#     def close(self):
#         self.fo.close()

#     def __del__(self):
#         self.fo.close()

# import pandas as pd

# chunksize = 1000
# file = '../.datasets/Licensees_0.csv'

# # Specify the licensee fields
# licensee_column_types = {
#     'global_id' : 'string',
#     'name': 'string',
#     'type': 'string',
#     # 'code': 'string',
#     # 'address1': 'string',
#     # 'address2': 'string',
#     # 'city': 'string',
#     # 'state_code': 'string',
#     # 'postal_code': 'string',
#     # 'country_code': 'string',
#     # 'phone': 'string',
#     # 'external_id': 'string',
#     # 'certificate_number': 'string',
#     # 'is_live': 'bool',
#     # 'suspended': 'bool',
# }

# # Specify the date columns.
# licensee_date_columns = [
#     'created_at', # No records if issued before 2018-02-21.
#     # 'updated_at',
#     # 'deleted_at',
#     # 'expired_at',
# ]

# # Specify all of the columns.
# licensee_columns = list(licensee_column_types.keys()) + licensee_date_columns

# reader = ChunkReader(file, chunksize, 1)
# for dfst in reader:
#     df = pd.read_csv(
#         dfst,
#         sep='\t',
#         encoding='utf-16',
#         usecols=licensee_columns,
#         dtype=licensee_column_types,
#         parse_dates=licensee_date_columns,
#     )
#     print(df.head()) #here I print to stdout, you can plot
# reader.close()

# SCRAP: Untried but useful looking way to read chunks of data.

# import pandas
# from functools import reduce

# def get_counts(chunk):
#     voters_street = chunk[
#         "Residential Address Street Name "]
#     return voters_street.value_counts()

# def add(previous_result, new_result):
#     return previous_result.add(new_result, fill_value=0)

# # MapReduce structure:
# chunks = pandas.read_csv("voters.csv", chunksize=1000)
# processed_chunks = map(get_counts, chunks)
# result = reduce(add, processed_chunks)

# result.sort_values(ascending=False, inplace=True)
# print(result)


# # Open enhanced lab results.
# enhanced_lab_results = pd.read_csv(
#     '../.datasets/enhanced_lab_results.csv',
#     # index_col='global_id',
#     dtype={
#         'global_id' : 'string',
#         'global_for_inventory_id': 'string',
#         'lab_id': 'string',
#         'inventory_type_id': 'string',
#         'strain_id': 'string',
#     }
# )

# Combine lab results merged with licensee data with enhanced lab results.
# lab_results = pd.merge(
#     left=lab_results,
#     right=enhanced_lab_results,
#     how='left',
#     left_on='global_id',
#     right_on='global_id',
# )

# Read in the enhanced lab results data.
# enhanced_lab_results = pd.read_csv(
#     '../.datasets/lab_results_with_strain_names.csv',
#     # index_col='global_id',
#     dtype={
#         'global_id' : 'string',
#         'global_for_inventory_id': 'string',
#         'lab_id': 'string',
#         'inventory_type_id': 'string',
#         'strain_id': 'string',
#     }
# )

