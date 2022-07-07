# Read lab results.
# dataset = 'LabResult_0'
# lab_results = pd.read_csv(
#     f'{DATA_DIR}/{folder}/{dataset}/{dataset}.csv',
#     usecols=model_names['lab_results']['fields'],
#     datecols=model_names['lab_results']['date_fields'],
# )
# lab_results.columns = [snake_case(x) for x in lab_results.columns]

# # Parse analyses!
# parsed_analyses = lab_results['TestName'].map(analyses).values.tolist()
# lab_results = lab_results.join(pd.DataFrame(parsed_analyses))
# lab_results['type'] = lab_results['type'].map(analysis_map)

# import csv
# import pandas as pd

# # Read and parse the licensees CSVs.
# # 1. If a row has a value in cell 22, shift 2 to the left,
# # condensing column 4 to 3 and column 6 to 5.
# # 2. If a row has a value in cell 21, shift 1 to the left,
# # condensing column 4 to 3.
# dataset = 'Licensee_0'
# datafile = f'{DATA_DIR}/{folder}/{dataset}/{dataset}.csv'
# csv_list = []
# with open(datafile, 'r', encoding='latin1') as csvf:
#     for line in csv.reader(csvf):
#         csv_list.append(line)
# headers = csv_list[:1][0]
# df = pd.DataFrame(csv_list[1:])
# csv_list = []
# for index, values in df.iterrows():
#     # FIXME: Some rows are even longer due to addresses.
#     if values[22]:
#         name = values[5] + values[6]
#         name = values[3] + values[4]
#         values.pop(6)
#         values.pop(4)
#     elif values[21]:
#         name = values[3] + values[4]
#         values.pop(4)
#     csv_list.append(values)
# licensees = pd.DataFrame(csv_list)
# licensees.columns = headers + [''] * (len(licensees.columns) - len(headers))
# licensees.drop('', axis=1, inplace=True)
# licensees.columns = [snake_case(x) for x in licensees.columns]
# licensees['license_issue_date'] = pd.to_datetime(licensees['license_issue_date'], errors='ignore')
# licensees['license_expiration_date'] = pd.to_datetime(licensees['license_expiration_date'], errors='ignore')
# licensees['created_date'] = pd.to_datetime(licensees['created_date'], errors='ignore')
# licensees['updated_date'] = pd.to_datetime(licensees['updated_date'], errors='ignore')
# licensees['license_number'] = licensees['license_number'].str.strip()
# licensees['name'] = licensees['name'].str.title()
# licensees['dba'] = licensees['dba'].str.title()
# licensees['city'] = licensees['city'].str.title()
# licensees['county'] = licensees['county'].str.title()

# Read licensee data.
# dataset = 'Licensee_0'
# datafile = f'{DATA_DIR}/{folder}/{dataset}/{dataset}.csv'
# licensees = pd.read_csv(datafile, sep='\t', lineterminator='\\r',)
# licensees.columns = ['headers']
# licensees = licensees['headers'].str.split(',', expand=True)
# headers = list(pd.read_csv(datafile, nrows=1).columns)
# licensees.columns = headers + [''] * (len(licensees.columns) - len(headers))
# licensees.drop('', axis=1, inplace=True)
# licensees.columns = [snake_case(x) for x in licensees.columns]
