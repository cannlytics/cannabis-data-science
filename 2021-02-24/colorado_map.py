#-----------------------------------------------------------------------------#
#                        Colorado Cannabis Cultivation and Dispensaries
#                                  February 2017
# Sources:
# https://www.colorado.gov/pacific/enforcement/med-licensed-facilities
# https://demography.dola.colorado.gov/gis/gis-data/#gis-data
# https://www.colorado.gov/pacific/revenue/colorado-marijuana-tax-data
#-----------------------------------------------------------------------------#
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
import pandas as pd
from colors import *
plt.rc('text', usetex=True)
#plt.rc('font',family='Computer Modern Roman')
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#-----------------------------------------------------------------------------#
#                                   DATA
#-----------------------------------------------------------------------------#
workbook = r'C:\Users\Lea-\Documents\Analytics\Research\Colorado-Market\Data\Colorado-cannabis-data.xlsx'
df = pd.read_excel(workbook, sheetname='Facilities-by-County',parse_cols =11,col=0)
                   
Total_Facilities_by_County = df['Total-Facilities']                   

#------------------------------ COLORADO BASEMAP -----------------------------#
# width and height of state in meters
# lat_0 and lon_0 are the center of the state
# lat_0=39.113014,lon_0=-105.358887,
# width=610000*1.05,height=450000*1.05,
map = Basemap(
              llcrnrlon = -109.060176 - .1,
              llcrnrlat = 36.992424 - .1,
              urcrnrlon = -102.041522 + .25,
              urcrnrlat = 41.003443999999995 + .005,
              lat_0=39.113014,lon_0=-105.358887,
              projection='lcc',
              resolution='c')
#             optional: rsphere=6370000.00
shp_info = map.readshapefile('ACS1014_county','states',drawbounds=True)
# Info: print(shp_info)  #print(map.states_info[0].keys())
colors={}
countynames=[]
#---------------------------------- COUNTY DATA -----------------------------#
map_data = {
'Adams'	:	74.0	,
'Alamosa'	:	2.0	,
'Arapahoe'	:	41.0	,
'Archuleta'	:	13.0	,
'Baca'	:	0.0	,
'Bent'	:	0.0	,
'Boulder'	:	187.0	,
'Broomfield':	0.0	,
'Chaffee'	:	11.0	,
'Cheyenne'	:	0.0	,
'Clear Creek':	36.0	,
'Conejos'	:	5.0	,
'Costilla'	:	16.0	,
'Crowley'	:	0.0	,
'Custer'	:	0.0	,
'Delta'	:	0.0	,
'Denver'	:	1227.0,
'Dolores'	:	0.0	,
'Douglas'	:	0.0	,
'Eagle'	:	33.0	,
'Elbert'	:	0.0	,
'El Paso'	:	376.0	,
'Fremont'	:	30.0	,
'Garfield'	:	60.0	,
'Gilpin'	:	10.0	,
'Grand'	:	13.0	,
'Gunnison'	:	25.0	,
'Hinsdale'	:	0.0	,
'Huerfano'	:	12.0	,
'Jackson'	:	0.0	,
'Jefferson'	:	70.0	,
'Kiowa'	:	0.0	,
'Kit Carson':	0.0	,
'Lake'	:	33.0	,
'La Plata'	:	10.0	,
'Larimer'	:	60.0	,
'Las Animas':	60.0	,
'Lincoln'	:	0.0	,
'Logan'	:	0.0	,
'Mesa'	:	7.0	,
'Mineral'	:	0.0	,
'Moffat'	:	1.0	,
'Montezuma'	:	18.0	,
'Montrose'	:	9.0	,
'Morgan'	:	17.0	,
'Otero'	:	3.0	,
'Ouray'	:	10.0	,
'Park'	:	32.0	,
'Phillips'	:	0.0	,
'Pitkin'	:	17.0	,
'Prowers'	:	0.0	,
'Pueblo'	:	237.0	,
'Rio Blanco':	0.0	,
'Rio Grande':	3.0	,
'Routt'	:	42.0	,
'Saguache'	:	20.0	,
'San Juan'	:	4.0	,
'San Miguel':	26.0	,
'Sedgwick'	:	4.0	,
'Summit'	:	23.0	,
'Teller'	:	5.0	,
'Washington':	0.0	,
'Weld'	:	20.0	,
'Yuma'	:	0.0	
}
area_names = []
area_colors={}
#------------------------------------PLOTTING---------------------------------#
"""HEATMAP"""
cmap = plt.cm.BuPu_r # use 'hot' colormap http://matplotlib.org/examples/color/colormaps_reference.html
vmin = 0.0; vmax = 1250.0 # set range.
for shapedict in map.states_info:
    area_name = shapedict['NAME']
    weight = map_data[area_name]
        # calling colormap with value between 0 and 1 returns
        # rgba value.  Invert color range (hot colors have high weight),
        # and then takes the sqrt root to spread out colors more.
    area_colors[area_name] = cmap(1.-np.sqrt(np.sqrt((weight-vmin)/(vmax-vmin))))[:3]
    area_names.append(area_name)
# cycle through state names, color each one.
ax = plt.gca() # get current axes instance
for nshape,seg in enumerate(map.states):
    area_color = rgb2hex(area_colors[area_names[nshape]]) 
    poly = Polygon(seg,facecolor=area_color,edgecolor=area_color)
    ax.add_patch(poly)
""" ANNOTATIONS """
#Adams
x, y  = map(-104.35, 39.87)
plt.annotate('1', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')
#
#Alamosa
x, y  = map(-105.75, 37.55)
plt.annotate('2', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Arapahoe
x, y = map(-104.33264, 39.65)
plt.annotate('3', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Archuleta
x, y = map(-107.00670, 37.175)
plt.annotate('4', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Baca
x, y = map(-102.52980, 37.3)
plt.annotate('5', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Bent
x, y = map(-103.08179, 37.95)
plt.annotate('6', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Boulder
x, y = map(-105.35, 40.1)
plt.annotate('7', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Broomfield
x, y = map(-105.04405, 39.94202)
plt.annotate('8', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Chaffee
x, y = map(-106.15, 38.70)
plt.annotate('9', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Cheyenne
x, y = map(-102.62162, 38.85)
plt.annotate('10', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Clear Creek
x, y = map(-105.64125, 39.69045)
plt.annotate('11', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Conejos
x, y = map(-106.20, 37.175)
plt.annotate('12', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Costilla
x, y = map(-105.45, 37.30)
plt.annotate('13', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Crowley
x, y = map(-103.775, 38.325)
plt.annotate('14', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Custer
x, y = map(-105.35, 38.075)
plt.annotate('15', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Delta
x, y = map(-107.80, 38.85314)
plt.annotate('16', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Dolores
x, y = map(-108.5, 37.75303)
plt.annotate('18', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Douglas
x, y = map(-104.93889, 39.35)
plt.annotate('19', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Eagle
x, y = map(-106.7, 39.6)
plt.annotate('20', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#El Paso
x, y = map(-104.50, 38.80)
plt.annotate('21', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Elbert
x, y = map(-104.15, 39.3)
plt.annotate('22', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Fremont
x, y = map(-105.45, 38.475)
plt.annotate('23', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Garfield
x, y = map(-107.65, 39.6)
plt.annotate('24', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Gilpin
x, y = map(-105.50, 39.8625)
plt.annotate('25', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Grand
x, y = map(-106.10, 40.10)
plt.annotate('26', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Gunnison
x, y = map(-107.00670, 38.70)
plt.annotate('27', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Hinsdale
x, y = map(-107.275, 37.80)
plt.annotate('28', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Huerfano
x, y = map(-104.93889, 37.65)
plt.annotate('29', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Jackson
x, y = map(-106.3, 40.69)
plt.annotate('30', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Jefferson
x, y = map(-105.225, 39.58003)
plt.annotate('31', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Kiowa
x, y = map(-102.62, 38.425)
plt.annotate('32', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Kit Carson
x, y = map(-102.6, 39.3)
plt.annotate('33', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#La Plata
x, y = map(-107.80, 37.325)
plt.annotate('34', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Lake
x, y = map(-106.35, 39.19412)
plt.annotate('35', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Larimer
x, y = map(-105.45, 40.69)
plt.annotate('36', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Las Animas
x, y = map(-104.10013, 37.30)
plt.annotate('37', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Lincoln
x, y = map(-103.40, 39.04575)
plt.annotate('38', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Logan
x, y = map(-103.1, 40.70562)
plt.annotate('39', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Mesa
x, y = map(-108.61756, 38.95854)
plt.annotate('40', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Mineral
x, y = map(-106.90, 37.65837)
plt.annotate('41', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Moffat
x, y = map(-108.23775, 40.61384)
plt.annotate('42', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Montezuma
x, y = map(-108.61756, 37.325)
plt.annotate('43', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Montrose
x, y = map(-108.14287, 38.46830)
plt.annotate('44', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Morgan
x, y = map(-103.8, 40.25)
plt.annotate('45', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Otero
x, y = map(-103.7, 37.9)
plt.annotate('46', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Ouray
x, y = map(-107.76362, 38.15)
plt.annotate('47', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Park
x, y = map(-105.7, 39.10)
plt.annotate('48', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Phillips
x, y = map(-102.34639, 40.6)
plt.annotate('49', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Pitkin
x, y = map(-106.9, 39.2)
plt.annotate('50', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Prowers
x, y = map(-102.35, 37.95)
plt.annotate('51', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Pueblo
x, y = map(-104.50, 38.15)
plt.annotate('52', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Rio Blanco
x, y = map(-108.23775, 39.98143)
plt.annotate('53', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Rio Grande
x, y = map(-106.35, 37.57495)
plt.annotate('54', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Routt
x, y = map(-107.0, 40.47716)
plt.annotate('55', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Saguache
x, y = map(-106.3, 38.1)
plt.annotate('56', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#San Juan
x, y = map(-107.65, 37.75737)
plt.annotate('57', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#San Miguel
x, y = map(-108.5, 38.025)
plt.annotate('58', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Sedgwick
x, y = map(-102.34639, 40.85)
plt.annotate('59', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Summit
x, y = map(-106.025, 39.575)
plt.annotate('60', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Teller
x, y = map(-105.17268, 38.86116)
plt.annotate('61', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Washington
x, y = map(-103.20, 40.0)
plt.annotate('62', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Weld
x, y = map(-104.45, 40.55)
plt.annotate('63', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')

#Yuma
x, y = map(-102.4, 40.0)
plt.annotate('64', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')


#Denver
x, y = map(-104.875, 39.9)
#plt.annotate('17', xy=(x,y), xycoords='data',fontsize=8,ha='center',va='center')
plt.annotate('17',
            xy=map(-105.1, 40.0),
            xytext=map(-104.95, 39.875),
            arrowprops=dict(arrowstyle="-"),
            )
plt.annotate('',
            xy=map(-105.05, 39.7),
            xytext=map(-104.875, 39.875),
            arrowprops=dict(arrowstyle="-"),
            )
#import Image
#im = Image.open('/Users/Lea-/Documents/Cannabis Analytics/Journal/Issue-1/Pictures/Cannabis_leaf_medium_black.png')
#height = im.size[1]
#im = np.array(im).astype(np.float) / 255
#plt.figimage(im, 210., 225. , zorder=99)
 
"""HEAT MAP LEGEND"""
legend_range = np.zeros((1,20))
for i in range(20):
    legend_range[0,i] = (i*5)/100.0
img = ax.imshow(legend_range, interpolation='nearest', vmin=vmin, vmax=vmax,
                cmap = plt.cm.BuPu)
color_bar = plt.colorbar(img,ticks=[0,250,500,750,1000,1250],
                         orientation='horizontal',
                         shrink=0.8,
                         pad= 0.05)
color_bar.ax.set_xticklabels(['0','250','500', '750', '1,000','1,250'])
    
#-----------------------------------------------------------------------------#               
plt.box(on=None)
plt.gcf().set_size_inches(6.5,6.5)
#plt.savefig('/Users/Lea-/Documents/Cannabis Analytics/Journal/Issue-1/Pictures/business_map.pdf',
#            bbox_inches='tight',
#            pad_inches = 0.05,
#            format='pdf',
#            dpi=300)
plt.show()













#-----------------------------------------------------------------------------# 
#                                  SCRAP
#-----------------------------------------------------------------------------# 
#for shapedict in map.states_info: #Assigns colors
#    countyname = shapedict['NAME']
#    law = counties[countyname]
#    if law==1:
#        colors[countyname] = Tgrey   
#    else:
#        colors[countyname] = Tgrey
#    countynames.append(countyname)  
#ax = plt.gca() #Gets current axes and cycle through to color each one.
#
#for nshape,seg in enumerate(map.states):    
#    if countynames[nshape] in ord_list:
#        color = rgb2hex(colors[countynames[nshape]]) 
#        poly = Polygon(seg,facecolor=color,edgecolor=almost_black,alpha=0.8,hatch='...')
#        ax.add_patch(poly)
#    else:
#        color = rgb2hex(colors[countynames[nshape]]) 
#        poly = Polygon(seg,facecolor=color,edgecolor=color,alpha=.2)
#        ax.add_patch(poly)
#-------------------------------------LEGEND----------------------------------#    
#laws = mpatches.Patch(facecolor=Tgrey, alpha=0.8,edgecolor=almost_black,
#                      hatch='...',label='Counties with MMJ Ordinances')
#no_laws = mpatches.Patch(facecolor=Tgrey, alpha=0.2,
#                         label='Counties without MMJ Ordinances')
#city_laws = mpatches.Circle((0.5, 0.5), 0.1, facecolor=almost_black,
#                            label='Cities with MMJ Ordinances')
##counties = mpatches.Patch(facecolor='white',edgecolor='white',
##                          label='A: Example County \
##                                \nB: Example County') 
##cities = mpatches.Patch(facecolor='white',edgecolor='white',
##                          label='1: Example City \
##                                \n2: Example City') 
##For bullet in legend
#from matplotlib.legend_handler import HandlerPatch
#class HandlerBullet(HandlerPatch):
#    def create_artists(self, legend, orig_handle,
#                       xdescent, ydescent, width, height, fontsize, trans):
#        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
#        p = mpatches.Circle(xy=center,radius=4.) #Adjust radius for size
#        self.update_prop(p, orig_handle, legend)
#        p.set_transform(trans)
#        return [p]
##Can add counties,cities
#plt.legend(handles=[no_laws,laws,city_laws],
#                loc='upper right',
#                bbox_to_anchor=(1.05,.95),frameon=True, shadow=False, 
#                ncol=1,fontsize=11,
#                handler_map={mpatches.Circle: HandlerBullet()})