"""

Data sources:

    -  Mississippi, Current County Subdivision State-based
    URL: <https://catalog.data.gov/dataset/tiger-line-shapefile-2016-state-mississippi-current-county-subdivision-state-based>

    - Mississippi FIPS codes
    URL: <https://www.colbertcounty.org/ema/weather/fips_miss.htm>

"""
import dotenv
from fredapi import Fred
import folium
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
import seaborn as sns


# === Setup ===

plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})



def get_county_population(fred, county_code: str):
    """Get the most recent population data for a specific county."""
    try:
        data = fred.get_series(county_code)
        most_recent_population = data.dropna().iloc[-1]
        return most_recent_population
    except:
        print('Failed to get data for:', county_code)
        return None


def get_all_county_populations(fred, county_codes: list):
    """Get the most recent population data for a list of counties."""
    populations = {}
    for code in county_codes:
        populations[code] = get_county_population(fred, code)
    return populations


# Read the license data.
licenses = pd.read_csv('./data/retailers-ms-2023-12-28.csv')

# Get the population for Mississippi counties.
config = dotenv.dotenv_values('../../.env')
fred_api_key = config['FRED_API_KEY']
fred = Fred(api_key=fred_api_key)
codes = {
    'Adams': 'MSADAM1POP',
    'Alcorn': 'MSALPOP',
    'Amite': 'MSAMIT5POP',
    'Attala': 'MSATPOP',
    'Benton': 'MSBEPOP',
    'Bolivar': 'MSBOPOP',
    'Calhoun': 'MSCAPOP',
    'Carroll': 'MSCRPOP',
    'Chickasaw': 'MSCHPOP',
    'Choctaw': 'MSCOPOP',
    'Claiborne': 'MSCLAI1POP',
    'Clarke': 'MSCLAR3POP',
    'Clay': 'MSCYPOP',
    'Coahoma': 'MSCMPOP',
    'Copiah': 'MSCOPI9POP',
    'Covington': 'MSCOVI1POP',
    'DeSoto': 'MSDEPOP',
    'Forrest': 'MSFORR0POP',
    'Franklin': 'MSFRAN7POP',
    'George': 'MSGEOR9POP',
    'Greene': 'MSGREE1POP',
    'Grenada': 'MSGNPOP',
    'Hancock': 'MSHANC5POP',
    'Harrison': 'MSHARR7POP',
    'Hinds': 'MSHIND9POP',
    'Holmes': 'MSHOPOP',
    'Humphreys': 'MSHUPOP',
    'Issaquena': 'MSISSA5POP',
    'Itawamba': 'MSITPOP',
    'Jackson': 'MSJACK9POP',
    'Jasper': 'MSJASP1POP',
    'Jefferson': 'MSJEFF3POP',
    'Jefferson Davis': 'MSJEFF5POP',
    'Jones': 'MSJONE0POP',
    'Kemper': 'MSKEMP9POP',
    'Lafayette': 'MSLAPOP',
    'Lamar': 'MSLAMA3POP',
    'Lauderdale': 'MSLAUD5POP',
    'Lawrence': 'MSLAWR7POP',
    'Leake': 'MSLEAK9POP',
    'Lee': 'MSLCPOP',
    'Leflore': 'MSLFPOP',
    'Lincoln': 'MSLINC5POP',
    'Lowndes': 'MSLOPOP',
    'Madison': 'MSMADI9POP',
    'Marion': 'MSMARI1POP',
    'Marshall': 'MSMSPOP',
    'Monroe': 'MSMOPOP',
    'Montgomery': 'MSMOPOP',
    'Neshoba': 'MSNESH9POP',
    'Newton': 'MSNEWT1POP',
    'Noxubee': 'MSNOPOP',
    'Oktibbeha': 'MSOKPOP',
    'Panola': 'MSPAPOP',
    'Pearl River': 'MSPEAR9POP',
    'Perry': 'MSPERR1POP',
    'Pike': 'MSPIKE3POP',
    'Pontotoc': 'MSPOPOP',
    'Prentiss': 'MSPNPOP',
    'Quitman': 'MSQUPOP',
    'Rankin': 'MSRANK5POP',
    'Scott': 'MSSCOT3POP',
    'Sharkey': 'MSSHAR5POP',
    'Simpson': 'MSSIMP7POP',
    'Smith': 'MSSMIT9POP',
    'Stone': 'MSSTON1POP',
    'Sunflower': 'MSSUPOP',
    'Tallahatchie': 'MSTAPOP',
    'Tate': 'MSTAPOP',
    'Tippah': 'MSTIPOP',
    'Tishomingo': 'MSTIPOP',
    'Tunica': 'MSTUPOP',
    'Union': 'MSUNPOP',
    'Walthall': 'MSWALT7POP',
    'Warren': 'MSWARR5POP',
    'Washington': 'MSWSPOP',
    'Wayne': 'MSWAYN3POP',
    'Webster': 'MSWEPOP',
    'Wilkinson': 'MSWILK7POP',
    'Winston': 'MSWNPOP',
    'Yalobusha': 'MSYAPOP',
    'Yazoo': 'MSYAZO3POP'
}
county_populations = get_all_county_populations(fred, list(codes.values()))
population_data = pd.DataFrame(list(county_populations.items()), columns=['county_code', 'population'])

# Calculate retailers per capita for each county.
retailer_counts = licenses['premise_county'].value_counts().reset_index()
retailer_counts.columns = ['county', 'count']
retailer_counts['county_code'] = retailer_counts['county'].str.replace(' County', '').map(codes)
counties = pd.merge(
    retailer_counts,
    population_data,
    on='county_code',
    how='left'
)
counties['retailers_per_capita'] = counties['count'] / (counties['population'] / 100)

# Visualize retailers per capita by county.
counties['county'] = counties['county'].str.replace(' County', '')
counties.sort_values(by='retailers_per_capita')['retailers_per_capita'].plot(kind='barh')
plt.figure(figsize=(15, 18))
counties['retailers_per_capita'].sort_values().plot(
    kind='barh',
    color='mediumpurple'
)
plt.xlabel('Dispensaries per 100,000 People')
plt.ylabel('County')
plt.title('Dispensaries Per Capita by County in Mississippi in 2023')
plt.yticks(ticks=range(len(counties)), labels=counties['county'])
plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.7)
sns.despine(trim=True)
plt.savefig('figures/ms-retailers-per-capita-2023-12-28.pdf', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


# === Visualize retailers per capita per county in Mississippi ===

def get_color(value, vmin, vmax):
    """Return color based on value from a colormap."""
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.Blues
    return cmap(norm(value))


# Get FIPS codes.
fips_codes = pd.read_excel('./data/ms-fips-codes.xlsx')
fips_codes.drop_duplicates(subset=['county'], inplace=True)

# Add fips codes to counties.
counties = pd.merge(
    counties,
    fips_codes,
    on='county',
    how='left'
)
counties['fips'] = counties['fips'].astype(str).str.replace('28', '')

# Create a choropleth map of licenses per capita.
fig, ax = plt.subplots(figsize=(10, 10))
shapefile = './data/ms-counties/tl_2016_28_cousub'
m = Basemap(
    llcrnrlat=30,
    urcrnrlat=35,
    llcrnrlon=-90.5,
    urcrnrlon=-88.1,
    lat_ts=20,
    resolution='i',
)
m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.readshapefile(shapefile, 'Mississippi')
vmin, vmax = counties['retailers_per_capita'].min(), counties['retailers_per_capita'].max()
for info, shape in zip(m.Mississippi_info, m.Mississippi):
    code = info['COUNTYFP']
    try:
        retailers_per_capita = counties.loc[counties['fips'] == code, 'retailers_per_capita'].values[0]
    except:
        retailers_per_capita = 0
    poly = Polygon(shape, facecolor=get_color(retailers_per_capita, vmin, vmax))
    ax.add_patch(poly)
    
# # Adjust borderlines to light grey
# for shape in m.Mississippi:
#     poly = Polygon(shape, edgecolor='lightgrey')
#     ax.add_patch(poly)

# # Annotate each county.
# plotted = []
# for info, shape in zip(m.Mississippi_info, m.Mississippi):
#     code = info['COUNTYFP']
#     plotted.append(code)
#     if code in plotted:
#         continue
#     try:
#         county = counties.loc[counties['fips'] == code, 'county'].values[0]
#     except:
#         continue
#     x, y = zip(*shape)  # get the shape points
#     ax.text(
#         np.mean(x),
#         np.mean(y),
#         county,
#         ha='center',
#         va='center',
#         fontsize=18,
#         fontweight='bold',
#     )

# FIXME: Set up the colorbar to match your color scheme
# norm = plt.Normalize(vmin=vmin, vmax=vmax)
# cmap = plt.cm.Blues
# sm = ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, orientation='horizontal')
# cbar.set_label('Retailers Per Capita')
plt.savefig('figures/ms-retailers-per-capita-map-2023-12-28.pdf', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


# === Create an interactive map of the licenses ===

def create_retailer_map(
        df,
        lat='premise_latitude',
        long='premise_longitude',
        color='crimson',
    ):
    """Create a map of licensed retailers."""
    m = folium.Map(
        location=[32.3547, -89.3985],
        zoom_start=7,
        control_scale=True
    )
    for _, row in df.iterrows():
        folium.Circle(
            radius=100,
            location=[row[lat], row[long]],
            color=color,
            fill=True
        ).add_to(m)
    return m


# Create and display the map
retailer_map = create_retailer_map(licenses)
retailer_map.save('figures/mississippi-retailers-map.html')
