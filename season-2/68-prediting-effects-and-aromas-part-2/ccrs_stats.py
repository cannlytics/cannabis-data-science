

# TODO: Summarize missing values.
data.info()
data.isnull().sum()
data.isna().sum()

# Select numeric.
data.select_dtypes(include='int64')

# Remove duplicates.
data.drop_duplicates(
    subset=None,
    keep='first',
    inplace=False,
    ignore_index=False,
)



# TODO: Determine the terpenes above and below the boiling points of THC / CBD.
# E.g. p-cymene has a boiling point above that of both THC and CBD.


# TODO: Match plants to licenses where possible.


# TODO: Count number of plants planted by licensee per month.

