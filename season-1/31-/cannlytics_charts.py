 """
Crispy Charts | Cannlytics

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 9/16/2021
Updated: 9/23/2021
License: MIT License <https://opensource.org/licenses/MIT>

Plotting Resources:
    https://scentellegher.github.io/visualization/2018/10/10/beautiful-bar-plots-matplotlib.html
    https://towardsdatascience.com/a-simple-guide-to-beautiful-visualizations-in-python-f564e6b9d392
    https://www.python-graph-gallery.com/ridgeline-graph-seaborn
    http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
    https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def crispy_scatter_plot(
        data,
        x,
        y,
        category_key,
        categories,
        colors,
        label_size=20,
        legend_loc='upper left',
        notes='',
        notes_offset=.15,
        note_size=14,
        percentage=False,
        save='',
        title='',
        title_size=24,
        font_size=18,
        fig_size=(15, 7.5),
        font_family='serif',
        font_style='Times New Roman',
        text_color='#333F4B',
):
    """Create a beautiful scatter plot given data.
    Args:
        TODO: Write docs!
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
    
    # Create a figure.
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot categorical data.
    for i, category in enumerate(categories):
        plt.scatter(
            x=x,
            y=y,
            data=data.loc[data[category_key]==category.upper(), :], 
            s=20,
            c=colors[i],
            label=str(category)
        )

    # Format the axes and axes labels.
    plt.gca().set(
        xlim=(0.0, 100),
        ylim=(0, 15),
    )
    
    # Format the X-axis.
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    
    # Add a title.
    if title:
        plt.title(title, fontsize=title_size, pad=20)
    
    # Add X and Y axis labels.
    plt.xlabel(x.replace('_', ' ').title(), fontsize=label_size, labelpad=10)
    plt.ylabel(y.replace('_', ' ').title(), fontsize=label_size, labelpad=10)
    
    # Format the legend.
    plt.legend(fontsize=font_size, loc=legend_loc)  

    # Hide unnecessary spines and ticks.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    
    # Format the axes as percentages.
    if percentage:
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    # Hide the first y-label to prevent overlap.
    plt.setp(ax.get_yticklabels()[0], visible=False)

    # Add figure notes.
    if notes:
        plt.figtext(0.0, -notes_offset, notes,
                    ha='left', fontsize=note_size)
    
    # Show and optionally save the figure.
    if save:
        plt.margins(1, 1)
        plt.savefig(
            save,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.75,
            transparent=False,
        )
    
    # Return the figure.
    plt.show()
    return fig


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
        TODO: Write docs!
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
