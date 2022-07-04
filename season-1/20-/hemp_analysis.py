"""
Hemp Analysis | Cannabis Data Science

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 6/30/2021
Updated: 7/14/2021
License: MIT License <https://opensource.org/licenses/MIT>
Data Sources:
    - Midwestern Hemp Database: https://extension.illinois.edu/global/midwestern-hemp-database
Resources:
    - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html
    - https://realpython.com/logistic-regression-python/
    - https://www.geeksforgeeks.org/understanding-logistic-regression/
    - https://eml.berkeley.edu/reprints/mcfadden/zarembka.pdf
    - https://data.princeton.edu/wws509/notes/c6s3
    - http://keeganskeate.com/recent_work#conditional-logit
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report, confusion_matrix


def convert_day_number_to_date(day_num, year):
    """Convert day number to date in particular year.
    Source: https://www.geeksforgeeks.org/python-convert-day-number-to-date-in-particular-year/
    Args:
        day_num (int):
        year (int):
    """
    day_num.rjust(3 + len(day_num), '0')
    dt = datetime.strptime(f'{str(year)}-{day_num}', '%Y-%j')
    return dt.strftime('%m-%d-%Y')


#--------------------------------------------------------------------------
# Read in the data.
#--------------------------------------------------------------------------
    
# Import the data.
data = pd.read_excel(
    './data/midwestern_hemp_database.xlsx',
    sheet_name='Cultivar Cannabinoid Data',
)

#--------------------------------------------------------------------------
# Rule 1. Look at the data.
#--------------------------------------------------------------------------

# Calculate high THC failures (0 for pass, 1 for failure).
data['fail'] = (data['total_thc'] >= 0.3).astype(int)

# Look at failure rates conditional on state.
data['state'] = data['state'].str.strip()
states = list(data['state'].unique())
for state in states:
    avg = data.loc[data.state == state]['fail'].mean()
    print(state, 'average hemp failure rate:', avg)

# Calculate the number of days into the year sampling occurred.

# 1. Turn sampling date into datetime.
data['sampled'] = data['sample_date'] .str.strip() + ' ' + data['sample_year'].astype('str')
data['sampled_at'] = pd.to_datetime(data['sampled'], format='%b %d %Y')
data.index = data.sampled_at
data.sort_index()

# 2. Subtract sampling date from beginning of year date
data['start_of_year'] = data.apply (lambda row: datetime(row['sample_year'], 1, 1), axis=1)
data['sampled_at_days'] = data['sampled_at'] - data['start_of_year']
data['sampled_at_days']  = data['sampled_at_days'].dt.days

#--------------------------------------------------------------------------
# Analyze hemp failure rates (for high THC).
#--------------------------------------------------------------------------

# Regression of THC on days into year of sampling.
model = sm.formula.ols(
    formula='total_thc ~ sampled_at_days',
    data=data,
)
regression = model.fit()
print(regression.summary())

# Regression of CBD on days into year of sampling.
model = sm.formula.ols(
    formula='total_cbd ~ sampled_at_days',
    data=data,
)
regression = model.fit()
print(regression.summary())

#--------------------------------------------------------------------------
# Predict hemp failure rates (for high THC).
#--------------------------------------------------------------------------

# Logistic egression of failure on days into year of sampling.
data['constant'] = 1
Y = data[['fail']]
X = data[['constant', 'sampled_at_days']]
log_reg = sm.Logit(Y, X).fit()
print(log_reg.summary())

# See how well the logistic regression predicts failure.
Y_hat = log_reg.predict(X)
RMSE = np.sqrt(((Y_hat.values - Y.values) ** 2).mean())

# Plot probability of failing against number of days until sampling.
year_days = pd.Series(range(0, 365), name='year_days')
constant_hat = pd.Series(len(year_days) * [1], name='constant')
forecast = pd.concat([constant_hat, year_days], axis=1).reset_index()
X_hat = forecast[['constant', 'year_days']]
sample_date_predictions = pd.Series(log_reg.predict(X_hat))
sample_date_predictions.plot(
    title='Estimated probability of hemp failing for high THC hemp given days until sampling date',
)
plt.show()

# Plot the logistic regression.
year_range = pd.date_range('2020-01-01', '2020-12-30', freq='d')
sample_date_predictions.index = year_range
plt.title('Estimated Probability of Hemp Failing for High THC')
plt.xlabel('Sampled At')
plt.scatter(data['sampled_at'], data['fail'])
plt.plot(year_range, sample_date_predictions, color='red')
plt.show()


#--------------------------------------------------------------------------
# Repeat the analysis factoring in the state of cultivation.
#--------------------------------------------------------------------------

# Create state dummy variables
dummies = pd.get_dummies(data['state']).rename(columns=lambda x: 'in_' + str(x))
data = pd.concat([data, dummies], axis=1)
Y = data['fail']
X = data[[
    'sampled_at_days',
    # 'in_Indiana',
    # 'in_Michigan',
    # 'in_Wisconsin',
]]

model = LogisticRegression()
model.fit(X, Y)
predicted_classes = model.predict(X)
accuracy = accuracy_score(data['fail'], predicted_classes)
print('Accuracy:', accuracy)

# Plot confusion matrix.
cm = confusion_matrix(Y, model.predict(X))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Pass', 'Predicted Fail'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Pass', 'Actual Fail'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
