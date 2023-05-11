


# Count licenses by business name.
group = data.groupby('business_dba_name')['license_number'].size().reset_index(name='count')

# Sanity check.
group.loc[group['business_dba_name'].str.contains('la mota', case=False)]
data.loc[data.business_dba_name.str.contains('la mota', case=False)]
