"""
Bokeh Chloropleth | Cannabis Data Science

Author: Keegan Skeate
Created: Wed Feb 24 07:17:22 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    

Resources:
    https://docs.bokeh.org/en/latest/docs/gallery/texas.html

"""
from bokeh.io import show
from bokeh.models import LogColorMapper
from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure
# from bokeh.sampledata.unemployment import data as unemployment
from bokeh.sampledata.us_counties import data as counties

# Define color palette.
palette = tuple(reversed(palette))


def bokeh_chloropleth(state, values):
    """Creates a Bokeh chloropleth for a given state's counties
    given values.
    
    Args:
        state (str): The state to analyze.
        values (list): A list of county values.
    """    

    # Define county shapes.
    counties = {
        code: county for code, county in counties.items() if county["state"] == state
    }
    county_xs = [county["lons"] for county in counties.values()]
    county_ys = [county["lats"] for county in counties.values()]
    county_names = [county['name'] for county in counties.values()]
    
    # Define county values
    # county_rates = [unemployment[county_id] for county_id in counties]
    
    color_mapper = LogColorMapper(palette=palette)
    
    data=dict(
        x=county_xs,
        y=county_ys,
        name=county_names,
        value=values,
    )
    
    TOOLS = "pan,wheel_zoom,reset,hover,save"
    
    p = figure(
        title="Washington State Cannabis Cultivators",
        tools=TOOLS,
        x_axis_location=None,
        y_axis_location=None,
        tooltips=[
            ("Name", "@name"),
            ("Number", "@value%"),
            ("(Long, Lat)", "($x, $y)")
        ])
    p.grid.grid_line_color = None
    p.hover.point_policy = "follow_mouse"
    
    p.patches(
        'x',
        'y',
        source=data,
        fill_color={'field': 'value', 'transform': color_mapper},
        fill_alpha=0.7,
        line_color="white",
        line_width=0.5,
        plot_width=800,
        plot_height=500
    )
    show(p)
    return p