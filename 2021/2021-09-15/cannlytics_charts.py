"""
Crispy Charts | Cannlytics

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 9/16/2021
Updated: 9/16/2021
License: MIT License <https://opensource.org/licenses/MIT>

Resources:
    https://scentellegher.github.io/visualization/2018/10/10/beautiful-bar-plots-matplotlib.html
    https://towardsdatascience.com/a-simple-guide-to-beautiful-visualizations-in-python-f564e6b9d392
    https://www.python-graph-gallery.com/ridgeline-graph-seaborn
    http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
    https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def crispy_bar_chart(
        df,
        annotations=False,
        key=0,
        fig_size=(5, 3.5),
        font_family='serif',
        font_style='Times NEw Roman',
        text_color='#333F4B',
        notes='',
        notes_offset=.15,
        palette=None,
        percentage=False,
        title='',
        save='',
        x_label=None,
        y_label=None,
        y_ticks=None,
        zero_bound=False,
):
    """Create a beautiful bar chart given data.
    Args:
        
    Returns:
        (figure): The chart figure for any post-processing.
    """
    
    # Set the chart font.
    plt.rcParams['font.family'] = font_family
    if font_family == 'sans-serif':
        plt.rcParams['font.sans-serif'] = font_style
    else:
        plt.rcParams['font.sans-serif'] = font_style
    
    # Set the style of the axes and the text color.
    plt.style.use('fivethirtyeight')
    plt.rcParams['axes.edgecolor'] = text_color
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color
    plt.rcParams['text.color'] = text_color
    
    # we first need a numeric placeholder for the y axis
    y_range=  list(range(1, len(df.index) + 1))
    
    # Create a figure.
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot the data with the following method.
    # Create for each type a horizontal line
    # that starts at x = 0 with the length 
    # represented by the specific expense percentage value.
    plt.hlines(
        y=y_range,
        xmin=0,
        xmax=df[key],
        colors=palette,
        linewidth=5
    )
    
    # create for each expense type a dot at the level of the expense percentage value
    values = []
    for i in y_range:
        plt.plot(
            df[key][i - 1],
            y_range[i - 1],
            'o',
            markersize=8,
            color=palette[i - 1],
        )
        values.append(df[key][i - 1])
    
    # Add annotations to the chart.
    if annotations:
        for i, txt in enumerate(values):
            txt = str(round(txt, 2))
            if percentage:
                txt += '%'
            coords = (df[key][i] + 0.5, y_range[i] - 0.125)
            ax.annotate(txt, coords)

    # Add a title.
    if title:
        plt.title(title, fontsize=21, color=text_color, loc='left')    
    
    # Set the x and y axis labels.
    if x_label is None:
        x_label = key.title()
    if y_ticks is None:
        y_ticks = df.index
    ax.set_xlabel(x_label, fontsize=18, color=text_color)
    # ax.xaxis.set_label_coords(0.25, -0.05)
    ax.set_ylabel(y_label)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.yticks(y_range, y_ticks)
    
    if percentage:
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    # Hide unnecessary spines.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Restirct the x-axis to 0 and above.
    if zero_bound:
        plt.xlim(0)
    
    # Add figure notes.
    if notes:
        plt.figtext(0.0, -notes_offset, notes, ha='left', fontsize=11)
    
    # Optionally save the figure.
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig