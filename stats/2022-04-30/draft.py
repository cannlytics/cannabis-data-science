

# # Those that "Needed more mj to get same effect pst 12 mos".
# mrjndmor = show_ratio(panel, survey_fields, 'mrjndmor')

# # Those that "Using same amt mj had less effect past 12 mos".
# mrjlsefx = show_ratio(panel, survey_fields, 'mrjlsefx')

# # Those that "Want/try to cut down/stop using mj pst 12 mos".
# mrjcutdn = show_ratio(panel, survey_fields, 'mrjcutdn')

# # Those "Able to cut/stop using mj every time pst 12 mos"
# mrjcutev = show_ratio(panel, survey_fields, 'mrjcutev')

# # Those where "Mj cause prbs with emot/nerves past 12 mos".
# mrjemopb = show_ratio(panel, survey_fields, 'mrjemopb')

# # Those with "Any phys prbs caused/worsnd by mj pst 12 mos"
# mrjphlpb = show_ratio(panel, survey_fields, 'mrjphlpb')

# # Contd to use cannabis despite phys prbs
# mrjphctd = show_ratio(panel, survey_fields, 'mrjphctd')

# # Less activities b/c of mj use past 12 mos
# mrjlsact = show_ratio(panel, survey_fields, 'mrjlsact')

# # Mj cause sers prbs at home/work/sch pst 12 mos
# mrjserpb = show_ratio(panel, survey_fields, 'mrjserpb')

# # Using mj and do dangerous activities pst 12 mos
# mrjpdang = show_ratio(panel, survey_fields, 'mrjpdang')

# # Using mj cause prbs with law past 12 mos
# mrjlawtr = show_ratio(panel, survey_fields, 'mrjlawtr')

# # Using mj cause prbs w/family/friends pst 12 mos
# mrjfmfpb = show_ratio(panel, survey_fields, 'mrjfmfpb')

# # Contd to use mj despite prbs w/ fam/frnds
# mrjfmctd = show_ratio(panel, survey_fields, 'mrjfmctd')


#-----------------------------------------------------------------------
# Analyze consumer behavior.
#-----------------------------------------------------------------------

# At job or business last time bought cannabis
# mmbatjob1

# Buy last cannabis from store or dispensary
# mmbtdisp

# Give away any cannabis got last time for free
# mmfgive

# Give away any cannabis last time you grew it
# mmggive

#-----------------------------------------------------------------------
# Characteristics of consumption.
#-----------------------------------------------------------------------

# # Use mj or hashish even once yr before last
# # mrjyrbfr = show_ratio(panel, survey_fields, 'mrjyrbfr')


#-----------------------------------------------------------------------

# # Male vs female 30 day consumers.
# female_consumers = past_30_day_consumers.loc[
#     past_30_day_consumers['irsex'] == 2
# ]
# proportion = len(female_consumers) / len(past_30_day_consumers)
# print('Proportion of monthly female consumers: %.2f%%' % (proportion * 100))

# # Male vs female past year.
# female_consumers = past_year_consumers.loc[
#     past_year_consumers['irsex'] == 2
# ]
# proportion = len(female_consumers) / len(past_year_consumers)
# print('Proportion of female consumers in 2020: %.2f%%' % (proportion * 100))

#-----------------------------------------------------------------------

# # Optional: Look at time of last use for people who used
# # longer than a year ago.
# old_time_users = panel.loc[
#     (panel['mjever'] == 1) &
#     (panel['drvinmarj'] == 2)
# ]
# proportion = len(old_time_users) / len(panel)
# print('Proportion who used to consume before 2019: %.2f%%' % (proportion * 100))

# # Look how people last used cannabis.
# # ("How did you get last cannabis used").
# coding = codings['mmgetmj']
# users = panel.loc[panel['mmgetmj'].isin([1, 2, 3, 4])]
# users['source'] = users['mmgetmj'].map(coding)

# # Visualize how people get their cannabis.
# fig, ax = plt.subplots(figsize=(8,8), facecolor='silver')
# users['source'].value_counts().plot(kind='pie')
# plt.show()

# # Calculate percent of past 30 day consumers who grew their own.
# homegrowers = past_30_day_consumers.loc[past_30_day_consumers['mmgetmj'] == 4]
# proportion = len(homegrowers) / len(past_30_day_consumers)
# print('Proportion who consumed their own homegrow in the last 30 days: %.2f%%' % (proportion * 100))

# homegrowers = past_year_consumers.loc[past_year_consumers['mmgetmj'] == 4]
# proportion = len(homegrowers) / len(past_year_consumers)
# print('Proportion who consumed their own homegrow in the last year: %.2f%%' % (proportion * 100))

# # Estimate the number of homegrowers in the US!
# # proportion = len(homegrowers) / len(panel)
# # us_homegrowers = proportion * us_population
# # print('Number of US homegrowers:', format_millions(us_homegrowers))


#-----------------------------------------------------------------------
# Analyze policy effects on consumption.
#-----------------------------------------------------------------------

# # How difficult is it for consumers to get cannabis?
# coding = codings['difgetmrj']
# panel['difficulty'] = panel['difgetmrj'].map(coding)
# state_medical = panel.loc[panel['medmjpa2'] == 1]
# state_not_medical = panel.loc[panel['medmjpa2'] == 2]

# # Visualize how hard it is for people in medical states to get cannabis.
# fig, ax = plt.subplots(figsize=(8,8), facecolor='silver')
# users = state_medical.loc[state_medical['difgetmrj'].isin([1, 2, 3, 4, 5])]
# users['difficulty'].value_counts().plot(kind='pie')
# plt.title('Reported difficulty to get cannabis in medical states')
# plt.ylabel('')
# plt.show()

# # Visualize how hard it is for people in non-medical states to get cannabis.
# fig, ax = plt.subplots(figsize=(8,8), facecolor='silver')
# users = state_not_medical.loc[state_not_medical['difgetmrj'].isin([1, 2, 3, 4, 5])]
# users['difficulty'].value_counts().plot(kind='pie')
# plt.title('Reported difficulty to get cannabis in non-medical states')
# plt.ylabel('')
# plt.show()

#-----------------------------------------------------------------------
# Analyze sales statistics.
#-----------------------------------------------------------------------

# # Amount paid for last cannabis joints bought
# key = 'mmjnpctb1'
# coding = codings[key]
# options = range(1, 5)
# users = past_year_consumers.loc[past_year_consumers[key].isin(options)]
# users['amount'] = users[key].map(coding)
# fig, ax = plt.subplots(figsize=(8,8), facecolor='silver')
# users['amount'].value_counts().plot(
#     kind='pie',
#     autopct=lambda pct: "{:.1f}%".format(pct)
# )
# plt.title('Amount paid for last cannabis joints bought by US Consumers')
# plt.ylabel('')
# plt.show()

# # Amount paid for cannabis bought last time
# key = 'mmlspctb1'
# coding = codings[key]
# options = range(1, 13)
# users = past_year_consumers.loc[past_year_consumers[key].isin(options)]
# users['amount'] = users[key].map(coding)
# fig, ax = plt.subplots(figsize=(8, 8), facecolor='silver')
# users['amount'].value_counts(normalize=True).mul(100).plot(kind='bar')
# plt.title('Amount paid for cannabis bought last time by US Consumers')
# plt.ylabel('Percent (%)')
# plt.show()

# # Amount worth last cannabis joints traded for.
# key = 'mmtjwrcb1'
# coding = codings[key]
# options = range(1, 13)
# users = past_year_consumers.loc[past_year_consumers[key].isin(options)]
# users['amount'] = users[key].map(coding)
# fig, ax = plt.subplots(figsize=(8, 8), facecolor='silver')
# users['amount'].value_counts(normalize=True).mul(100).plot(kind='bar')
# plt.title('Amount worth last cannabis joints traded by US Consumers')
# plt.ylabel('Percent (%)')
# plt.show()

# # Amount the last cannabis traded for was worth
# key = 'mmtlwrcb1'
# coding = codings[key]
# options = range(1, 13)
# users = past_year_consumers.loc[past_year_consumers[key].isin(options)]
# users['amount'] = users[key].map(coding)
# fig, ax = plt.subplots(figsize=(8, 8), facecolor='silver')
# users['amount'].value_counts(normalize=True).mul(100).plot(kind='bar')
# plt.title('Amount the last cannabis traded for was worth by US Consumers')
# plt.ylabel('Percent (%)')
# plt.show()

# # Optional: Price category of last cannabis joints bought
# # mmjnpcat1


# Estimate the demand for sales in the US.

# Future work: Estimate price per gram and compare with prior data.



# Optional: Number of cannabis joints bought last time
# mmjntnum1


# Optional: Get economic variables (e.g. from Fed FRED) and
# correlate state economic variables, such as cannabis sales,
# with participation rates.

# Age of last use.
# invalid_ages = list(codings['mrjaglst'].keys())
# invalid_ages = [int(x) for x in invalid_ages]
# age_last_use = panel.loc[~panel['mrjaglst'].isin(invalid_ages)]
# age_last_use['mrjaglst'].hist(bins=100)
# plt.vlines(25, ymin=0, ymax=700)
# plt.title('Age of LastS Cannabis Consumers')
# plt.xlabel('Age')
# plt.show()
# print('N:', len(age_last_use))

def calculate_ratio(data, field, criterion=[1], comparison=[1, 2]):
    numerator = len(data.loc[data[field].isin(criterion)])
    denominator = len(data.loc[data[field].isin(comparison)])
    return  numerator / denominator


def show_ratio(data, fields, field):
    ratio = calculate_ratio(data, field)
    description = fields.loc[fields['key'] == field].iloc[0]['description']
    print(f'{description}: ' +  '%.2f%%' % (ratio * 100))
    return ratio

#-----------------------------------------------------------------------

# # Look at consumption rates of people who work vs. unemployed.
# # wrkhadjob | wrkdpstwk | wrknjbwks
# employed = panel.loc[panel['wrkdpstwk'] == 1]
# employed_consumers = employed.loc[employed['mjrec'] == 1]
# proportion = len(employed_consumers) / len(employed)
# print('Proportion of employed who partake: %.2f%%' % (proportion * 100))

# unemployed = panel.loc[panel['wrkdpstwk'] == 2]
# unemployed_consumers = unemployed.loc[unemployed['mjrec'] == 1]
# proportion = len(unemployed_consumers) / len(unemployed)
# print('Proportion of unemployed who partake: %.2f%%' % (proportion * 100))

# Number of times consumer cannabis in the past year.
# mjyrtot
# irmjfy

# Multiply monthly consumers by the number of days bought cannabis past 30 days.
# key = 'mmbt30dy'
# coding = codings[key]
# options = list(coding.keys())
# frequent_users = users.loc[
#     (~users[key].astype(str).isin(options)) &
#     (users['monthly_consumer'] == 1)
# ]
# monthly_amount = frequent_users['grams_bought'] * frequent_users[key]
# users.loc[frequent_users.index, 'grams_bought'] = monthly_amount
