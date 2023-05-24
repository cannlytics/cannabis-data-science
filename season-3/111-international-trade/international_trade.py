"""
International Cannabis Trade
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 5/10/2023
Updated: 5/11/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Source:

    - [South Africa Cannabis Licenses](https://www.sahpra.org.za/approved-licences/)

Resources:

    - [South Africa Cannabis Licensing Press Release](https://www.sahpra.org.za/press-releases/cannabis-licensing/)
    - [First-level Administrative Divisions, South Africa, 2015](https://earthworks.stanford.edu/catalog/stanford-js788dt6134)

"""
# Standard imports:
from datetime import datetime
from time import sleep

# External imports:
import matplotlib.pyplot as plt
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 21,
})


#------------------------------------------------------------------------------
# Get the data.
#------------------------------------------------------------------------------

# Open the licenses webpage.
url = 'http://www.sahpra.org.za/approved-licences/'
driver = webdriver.Edge()
driver.get(url)

# Click on the cultivations tab.
sleep(3)
link = driver.find_element(By.ID, 'ui-id-5')
link.click()

# Iterate over all of the pages.
data = []
iterate = True
while iterate:

    # Find the table.
    table = driver.find_element(By.CSS_SELECTOR, '[aria-label="Cannabis Cultivation licences"]')

    # Get all the rows from the table.
    rows = table.find_elements(By.TAG_NAME, 'tr')

    # Extract the data from each cell.
    for row in rows[3:]:
        cells = row.find_elements(By.TAG_NAME, 'td')
        row_data = {
            'business_legal_name': cells[0].text,
            'license_number': cells[1].text,
            'business_owner': cells[2].text,
            'premise_state': cells[3].text,
            'issue_date': cells[4].text,
            'expiration_date': cells[5].text,
        }
        data.append(row_data)

    # Click the next button, if not disabled.
    li = driver.find_element(By.CSS_SELECTOR, '[aria-label="next"]')
    action = li.find_element(By.TAG_NAME, 'a')
    style = li.get_attribute('class')
    if 'disabled' in style.split():
        iterate = False
    action.click()
    sleep(3)

# TODO: Get `data_refreshed_at`.

# Close the browser.
driver.close()

# Save the data.
date = datetime.now().strftime('%Y-%m-%d')
df = pd.DataFrame(data)
df.to_excel(f'south-africa-cannabis-licenses-{date}.xlsx', index=False)


#------------------------------------------------------------------------------
# Analyze the data.
#------------------------------------------------------------------------------

# Count the number of licenses by province.
counts = df.groupby('premise_state').count()['business_legal_name']
counts = counts.sort_values(ascending=False)

# Bar chart of number of licenses by province.
counts.plot(x='premise_state', y='Number', kind='bar')
plt.title('South Africa Cannabis Cultivation Licenses by Province')
plt.xlabel('Province')
plt.show()


#------------------------------------------------------------------------------
# Visualize the data.
#------------------------------------------------------------------------------

# Plotting dependencies:
from adjustText import adjust_text
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
import numpy as np

# Use LaTeX when rendering text.
plt.rc('text', usetex=True)
plt.rcParams['font.family'] = 'Times'

# Create the map outline.
figure = Basemap(
    projection='merc',
    llcrnrlat=-38,
    llcrnrlon=14,
    urcrnrlat=-22,
    urcrnrlon=34,
    resolution='l',
)

# Shade oceans
figure.drawlsmask(
    land_color='#ededed',
    ocean_color='#add8e6',
    lakes=True,
    resolution='l',
)

# Read shapefile and draw provinces.
shapefile_path = './south-africa-shapefile/js788dt6134'
figure.readshapefile(shapefile_path, 'provinces', linewidth=0.5)

# Create a colormap.
cmap = plt.cm.YlOrRd

# Normalize the counts of licenses.
norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())

# Iterate through provinces, match with counts, and then shade a polygon.
provinces, texts = [], []
for info, shape in zip(figure.provinces_info, figure.provinces):
    province_name = info['name_1'].replace('-', ' ')
    count = counts[province_name]
    color = cmap(norm(count))
    poly = plt.Polygon(shape, facecolor=color, edgecolor='k')
    plt.gca().add_patch(poly)

    # Annotate province names.
    if province_name not in provinces:
        provinces.append(province_name)
    else:
        continue
    if province_name == 'KwaZulu Natal':
        province_name = 'KwaZulu-Natal'
    x, y = np.array(shape).mean(axis=0)
    text = plt.text(x, y,
        province_name,
        fontsize=20,
        ha='center',
        va='center',
        color='black',
        bbox=dict(
            alpha=0.8,
            facecolor='white',
            edgecolor='black',
            boxstyle='round,pad=0.3'
        )
    )
    if province_name in ['North West', 'Mpumalanga', 'Gauteng']:
        texts.append(text)

# Adjust text positions to prevent overlaps
adjust_text(texts, ha='left')

# Add legend
num_colors = 5
bounds = np.linspace(counts.min(), counts.max(), num_colors + 1)
colors = [cmap(norm(value)) for value in bounds]
labels = [f"{bounds[i]:.0f} - {bounds[i+1]:.0f}" for i in range(num_colors)]
patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(num_colors)]
plt.legend(
    handles=patches,
    title='Number of licenses',
    loc='lower right',
    frameon=True,
)

# Add footnote
footnote = """\\textbf{Author}: Cannlytics and the Cannabis Data Science Team (2023).\n
\\textbf{Data source}: South African Health Products Regulatory Authority.
Retrieved from: https://www.sahpra.org.za/approved-licences/\n
\\textbf{Map data}: Hijmans, Robert J.. University of California, Berkeley. Museum of
Vertebrate Zoology. (2015). First-level Administrative Divisions,
South Africa, 2015. [Shapefile]. University of California, Berkeley.
Museum of Vertebrate Zoology.
Retrieved from https://earthworks.stanford.edu/catalog/stanford-js788dt6134"""
plt.annotate(
    footnote,
    xy=(0, 0),
    xycoords='axes fraction',
    fontsize=18,
    xytext=(0, -5),
    textcoords='offset points',
    ha='left',
    va='top'
)

# Draw notable features.
figure.drawcoastlines(linewidth=1)
figure.drawcountries(linewidth=0.5)

# Render the map.
plt.gcf().set_size_inches(30, 10)
plt.title(
    'South Africa Cannabis Cultivation\nLicenses by Province in 2023',
    fontsize=28,
    pad=10,
)

# Save the figure to a file.
plt.axis('off')
plt.savefig(
    'south-africa-cannabis-cultivations.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='#add8e6',
    edgecolor='none',
)
plt.show()
