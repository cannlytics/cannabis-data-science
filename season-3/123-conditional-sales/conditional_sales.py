"""
Conditional Sales
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 8/9/2023
Updated: 8/9/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
import glob
import os

# External imports:
import matplotlib.pyplot as plt
import pandas as pd


# === Get the data ===

# Define where your data lives.
data_dir = 'D://data/washington/ccrs-stats/licensee_stats'

# Read licensee data.
# licensees = pd.read_csv('./licenses-wa-2023-08-09T06-49-02.csv')

# Read sales data by licensee.
all_sales = []
pattern = os.path.join(data_dir, "*", "sales-*-2023-*.xlsx")
paths_2023 = glob.glob(pattern)
for path in paths_2023:
    sales = pd.read_excel(path)
    all_sales.append(sales)
    print(f'Read: {path}')

# Aggregate all sales.
all_sales = pd.concat(all_sales)
print(f'Total items sold in the first half of 2023: {len(all_sales):,}')


# === Analyze the data ===

# Calculate the number of products sold.
total_products_sold = all_sales['product_name'].nunique()
print(f'Total products sold in the first half of 2023: {total_products_sold:,}')

# Calculate the number of strains sold.
total_strains_sold = all_sales['strain_name'].nunique()
print(f'Total strains sold in the first half of 2023: {total_strains_sold:,}')

# Calculate the total amount spent.
total_spent = all_sales['unit_price'].sum()
print(f'Total spent in the first half of 2023: ${total_spent:,.0f}')

# Calculate the total amount of discounts.
total_discounts = all_sales['discount'].sum()
print(f'Total discounts in the first half of 2023: ${total_discounts:,.0f}')

# Calculate the total amount of tax.
other_tax = all_sales['other_tax'].sum()
sales_tax = all_sales['sales_tax'].sum()
total_tax = other_tax + sales_tax
print(f'Total tax in the first half of 2023: ${total_tax:,.0f}')


# === Visualize the data ===

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


# Bar chart of product types sold.
totals_by_type = pd.DataFrame(all_sales.value_counts('inventory_type'))
proportion = totals_by_type / totals_by_type.sum() 
proportion[:10].plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Products sold in WA in the first half of 2023')
plt.xlabel('Inventory Type')
plt.ylabel('Proportion of Products Sold')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Plot sales over time by month.
month_dict = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June'
}
all_sales['month'] = all_sales['created_date'].dt.month
monthly_totals = all_sales.groupby('month', as_index=False)['unit_price'].sum()
monthly_totals[:-1]['unit_price'].div(1_000_000).plot(color='skyblue')
plt.title('Total Sales by Month in WA in 2023')
plt.xlabel('Month')
plt.ylabel('Total Sales (Millions of $)')
plt.xticks(rotation=0)
plt.ylim(0)
plt.xticks(ticks=range(6), labels=[month_dict[i] for i in range(1, 7)], rotation=45)
plt.tight_layout()
plt.show()
