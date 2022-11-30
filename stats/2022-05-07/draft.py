# # Find percent of population who consumed in last 30 days.
# past_30_day_consumers = panel.loc[panel['mjrec'] == 1]
# proportion = len(past_30_day_consumers) / len(panel)
# print('Proportion who consumed in last 30 days: %.2f%%' % (proportion * 100))

# past_year_consumers = panel.loc[(panel['mjrec'] == 1) | (panel['mjrec'] == 2)]
# proportion = len(past_year_consumers) / len(panel)
# print('Proportion who consumed in 2020: %.2f%%' % (proportion * 100))

# # Find the percent that have used medical cannabis in the past year.
# past_year_patients = panel.loc[panel['medmjyr'] == 1]
# proportion = len(past_year_patients) / len(panel)
# print('Proportion with medical cannabis perscription in 2020: %.2f%%' % (proportion * 100))

#-----------------------------------------------------------------------

# # Look at the age where people first used cannabis.
# invalid_ages = list(codings['mjage'].keys())
# invalid_ages = [int(x) for x in invalid_ages]
# first_use_age = sample.loc[~sample['mjage'].isin(invalid_ages)]
# ax = sns.displot(data=first_use_age, x='mjage', bins=100)
# plt.title('Age of First Cannabis Use of US Consumers')
# plt.xlabel('Age')
# plt.gcf().set_size_inches(18.5, 10.5)
# plt.vlines(25, ymin=0, ymax=2000)
# plt.show()
# print('N =', len(first_use_age))

# # Probability of first-use being greater than 25?
# older_25 = first_use_age.loc[first_use_age['mjage'] > 25]
# proportion = len(older_25) / len(first_use_age)
# print('Probability of first-time use being greater than 25: %.2f%%' % proportion)


#-----------------------------------------------------------------------
# Determine ages from use / age at past use.
# Problem: This introduces bias because it appears people who
# report using other substances are also more likely to report
# using cannabis.
#-----------------------------------------------------------------------

# age_fields = [
#     {'key': 'mrjaglst', 'indicator': 'mjrec'},
#     {'key': 'cigaglst', 'indicator': 'cigrec'},
#     {'key': 'smkaglast', 'indicator': 'smklssrec'},
#     {'key': 'cgraglst', 'indicator': 'cigarrec'},
#     {'key': 'alcaglst', 'indicator': 'alcrec'},
#     {'key': 'cigaglst', 'indicator': 'cigrec'},
#     {'key': 'cocaglst', 'indicator': 'cocrec'},
#     {'key': 'crkaglst', 'indicator': 'crakrec'},
#     {'key': 'heraglst', 'indicator': 'herrec'},
#     {'key': 'hallaglst', 'indicator': 'hallucrec'},
#     {'key': 'lsdaglst', 'indicator': 'lsdrec'},
#     {'key': 'pcpaglst', 'indicator': 'pcprec'},
#     {'key': 'ecstmoagl', 'indicator': 'ecstmorec'},
#     {'key': 'inhlaglst', 'indicator': 'inhalrec'},
#     {'key': 'methaglst', 'indicator': 'methamrec'},    
# ]


# def determine_age_from_field(data, codings, key, indicator):
#     """Get the age of people who used cannabis in the past month or year.
#     Remove invalid ages and returning a series of ages with only valid ages.
#     """
#     field = data.loc[(data[indicator] == 1) | (data[indicator] == 2)]
#     invalid_ages = list(codings[key].keys())
#     invalid_ages = [int(x) for x in invalid_ages]
#     field = field.loc[~field[key].isin(invalid_ages)]
#     return field[key]


# # Attempt to identify the age of the sample.
# panel['age'] = 0
# for age_field in age_fields:
#     key = age_field['key']
#     indicator = age_field['indicator']
#     ages = determine_age_from_field(panel, codings, key, indicator)
#     panel.loc[ages.index, 'age'] = ages

# proportion = len(panel.loc[panel['age'] > 0]) / len(panel)
# print('Identified ages for %.2f%% of the sample.' % proportion)


#-----------------------------------------------------------------------

# # Assign age cohort.
# panel['older_25'] = False
# panel.loc[panel['mrjaglst'] > 25, 'older_25'] = True

# # Identify the ages of people who used in the past year.
# past_year_ages = panel.loc[
#     (panel['mjrec'] == 1) |
#     (panel['mjrec'] == 2)
# ]
# invalid_ages = list(codings['mrjaglst'].keys())
# invalid_ages = [int(x) for x in invalid_ages]
# past_year_ages = past_year_ages.loc[~past_year_ages['mrjaglst'].isin(invalid_ages)]

# # Plot known ages of known consumers.
# ax = sns.displot(
#     data=past_year_ages,
#     x='mrjaglst',
#     hue='older_25',
#     bins=100,
#     legend=False,
# )
# plt.legend(labels=['Over 25', 'Under 25'])
# plt.title('Age of Cannabis Users in the US in 2020')
# plt.xlabel('Age')
# plt.gcf().set_size_inches(18.5, 10.5)
# plt.vlines(25, ymin=0, ymax=150)
# plt.show()
# print('N:', len(past_year_ages))

# # Probability of a cannabis consumer being older than 25?
# older_25 = past_year_ages.loc[past_year_ages['mrjaglst'] > 25]
# proportion = len(older_25) / len(past_year_ages)
# print('Probability of cannabis consumer being greater than 25: %.2f%%' % proportion)



# Look at the correlation amon explanatory variables.
# correlation = sample[list(predictors.values())].corr()
# sns.heatmap(correlation, annot=True, cmap='vlag_r')
# plt.show()



#-----------------------------------------------------------------------
# Analytics: Look at correlation with other interesting fields.
#-----------------------------------------------------------------------

# # Look at service members who partake in cannabis consumption.
# service = panel.loc[panel['service'] == 1]
# service_consumers = service.loc[
#     (service['mjrec'] == 1) |
#     (service['medmjyr'] == 1)
# ]
# proportion = len(service_consumers) / len(service)
# print('Proportion of service members who partake: %.2f%%' % (proportion * 100))


# Estimate the number of days consumers consume, who consumed in last 30 days.
# coding = codings['mr30est']
# options = ['1', '2', '3', '4', '5', '6']
# users = consume_monthly.loc[consume_monthly['mr30est'].astype(str).isin(options)]
# users['frequency'] = users['mr30est'].astype(str).map(coding)

# Look at monthly consumers vs.
# total # of days used alcohol in past 12 mos 
# sample['monthly_consumer'] = 0
# sample.loc[consume_monthly.index, 'monthly_consumer'] = 1
# sns.scatterplot(
#     data=sample,
#     x='alcyrtot',
#     y='monthly_consumer',
# )
# plt.show()


explanatory_variables = [
    # 'income',
    # 'catag6',
    # 'height',
    # 'weight',
    # 'service',
    # 'eduhighcat',
    # 'consumption',
    # 'monthly_consumer',
    # 'emerg_room',
    # 'height',
    # 'weight',
    # 'out_patient',
    # 'first_use_alcohol_revised',
    # 'alcohol_days',
    # 'unable_to_work',
    # 'mrjaglst',
    # 'height',
    # 'weight',
    # 'service',
    # 'older_25',
]


# Correlate consumption with various factors.
correlation = subsample[explanatory_variables + ['consumer', 'consumption']].corr()
sns.heatmap(correlation, annot=True, cmap='vlag_r')
plt.show()


# TODO: Code these variables.
# Identify factors that predict amount of use:
# - mrjaglst
# - income (bracket)
# - htinche2 (height in in.)
# - wtpound2 (weight in pounds)
# - cadrlast (# of drinks in the past month)


# Interest rate? Inflation?




# Predict the probability of being a 1st time consumer.

# Predict the probability of being a yearly user.

# Predict the probability of being a monthly user.

# Predict the amount of cannabis consumed per month, given probability
# of being a user (Heckman model!).

# Just for fun: Predict the probability of being a homegrower.


#-----------------------------------------------------------------------

# Correlate probability of consuming and consumption amount with
# various factors.


#-----------------------------------------------------------------------
# Health analysis
#-----------------------------------------------------------------------

# Question: Does consuming cannabis or the amount of cannabis consumed
# corretate with a greater or lesser likelihood of adverse health events?



# Optional: Remove missing values:
# - mrjaglst



# exog_variables=[
#     'height',
#     'weight',
#     'age_18_25_years_old',
#     'age_26_34_years_old',
#     'age_35_49_years_old',
#     'age_50_64_years_old',
#     'age_65_or_older',
#     'income_bracket_dollars_50_000_dollars_74_999',
#     'income_bracket_dollars_75_000_or_more',
#     'income_bracket_less_than_dollars_20_000',
# ]
# select = subsample[[
#     'height',
#     'weight',
#     'age_18_25_years_old',
#     'age_26_34_years_old',
#     'age_35_49_years_old',
#     'age_50_64_years_old',
#     'age_65_or_older',
#     'education_college_graduate',
#     'education_high_school_grad',
#     'education_less_high_school',
#     'education_some_colltoassoc_dg',
#     'income_bracket_dollars_50_000_dollars_74_999',
#     'income_bracket_dollars_75_000_or_more',
#     'income_bracket_less_than_dollars_20_000',
# ]]



# fig, ax = plt.subplots(figsize=(12, 8))
# ax = sns.displot(data=subsample, x='consumption', bins=100)
# sns.displot(data=predicted_consumption, bins=100, ax=ax)
# subsample['consumption'].hist(bins=100, density=True)


# predicted_consumption.hist(bins=100)

# sns.displot(
#     subsample,
#     x='predicted_consumption',
#     hue='predicted_consumer',
#     kind="kde"
# )


# TODO: Visualize consumption by various factors.
# g = sns.catplot(
#         x='seats', 
#         y='mileage_kmpl', 
#         data=cars,
#         palette='bright',
#         aspect=2,
#         inner=None,
#         kind='violin')
# sns.stripplot(
#     x='seats', 
#     y='mileage_kmpl', 
#     data=cars,
#     color='k', 
#     linewidth=0.2,
#     edgecolor='white',
#     ax=g.ax);



# consume_annually['grams_bought'] = median_bought
