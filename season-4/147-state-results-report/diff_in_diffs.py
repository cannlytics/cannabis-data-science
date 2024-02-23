

import statsmodels.api as sm
import statsmodels.formula.api as smf


# TODO: Read CA results.


# TODO: Read FL results.



# Assuming `sample` is your DataFrame and it includes 'lab', 'date_tested', and your outcome variables
sample['post_compliance'] = (sample['date_tested'] >= compliance_date).astype(int)
sample['lab_id'] = sample['lab'].astype('category').cat.codes  # Convert lab to categorical codes if not already

# Example for Total THC
# Using formula API for convenience in specifying fixed effects
model = smf.ols('total_thc ~ post_compliance + C(lab_id) + post_compliance:C(lab_id)', data=sample).fit()

# Output model summary to console
print(model.summary())

# Export regression table to LaTeX
latex_output = model.summary().as_latex()
with open(f'{report_dir}/DiD_regression_total_thc.tex', 'w') as file:
    file.write(latex_output)

