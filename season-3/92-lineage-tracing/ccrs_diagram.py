"""
CCRS Diagram
Copyright (c) 2022 Cannabis Data

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 12/26/2022
Updated: 12/27/2022
License: <https://github.com/cannabisdata/cannabisdata/blob/main/LICENSE>
"""
# Standard imports:
from urllib.request import urlretrieve

# External imports:
from cannlytics.data.ccrs.constants import CCRS_DATASETS
from diagrams import Cluster, Diagram, Edge, Node
from diagrams.custom import Custom


def download_ccrs_logo(dest='ccrs-logo.png'):
    """Download a logo for the WSLCB CCRS."""
    logo_url = 'https://pbs.twimg.com/profile_images/1149434490776768513/WtlHS5W2_400x400.png'
    urlretrieve(logo_url, dest)


def create_node_label(fields):
    """Format dataset fields into a label for a node."""
    string = ''
    for k, v in fields.items():
        string += f'\n{k}: {v}'
    return string.lstrip('\n')


def create_dataset_node(title, fields, cluster_attr={}, node_attr={}):
    """Create a dataset node."""
    with Cluster(title, graph_attr=cluster_attr):
        label = create_node_label(fields)
        cluster = Node(label, **node_attr)
    return cluster


def ccrs_diagram(
        filename=None,
        direction='LR',
        cluster_attr={},
        graph_attr={},
        node_attr={},
        logo_path='ccrs-logo.png',
    ):
    """Render a CCRS diagram."""
    download_ccrs_logo(logo_path)
    with Diagram(
        direction=direction,
        filename=filename, 
        graph_attr=graph_attr,
    ) as diagram:

        # Areas node.
        areas = create_dataset_node(
            'Areas',
            CCRS_DATASETS['areas']['fields'],
            cluster_attr,
            node_attr,
        )

        # Inventory node.
        inventory = create_dataset_node(
            'Inventory',
            CCRS_DATASETS['inventory']['fields'],
            cluster_attr,
            node_attr,
        )

        # Inventory Adjustments node.
        inventory_adjustments = create_dataset_node(
            'Inventory Adjustments',
            CCRS_DATASETS['inventory_adjustments']['fields'],
            cluster_attr,
            node_attr,
        )

        # Lab results node.
        lab_results = create_dataset_node(
            'Lab Results',
            CCRS_DATASETS['lab_results']['fields'],
            cluster_attr,
            node_attr,
        )

        # Licensees node.
        licensees = create_dataset_node(
            'Licensees',
            CCRS_DATASETS['licensees']['fields'],
            cluster_attr,
            node_attr,
        )

        # Plants node.
        plants = create_dataset_node(
            'Plants',
            CCRS_DATASETS['plants']['fields'],
            cluster_attr,
            node_attr,
        )

        # Plant Destructions node.
        plant_destructions = create_dataset_node(
            'Plant Destructions',
            CCRS_DATASETS['plant_destructions']['fields'],
            cluster_attr,
            node_attr,
        )

        # Products node.
        products = create_dataset_node(
            'Products',
            CCRS_DATASETS['products']['fields'],
            cluster_attr,
            node_attr,
        )

        # Sale details node.
        sale_details = create_dataset_node(
            'Sale Details',
            CCRS_DATASETS['sale_details']['fields'],
            cluster_attr,
            node_attr,
        )

        # Sale headers node.
        sale_headers = create_dataset_node(
            'Sale Headers',
            CCRS_DATASETS['sale_headers']['fields'],
            cluster_attr,
            node_attr,
        )

        # Strains node.
        strains = create_dataset_node(
            'Strains',
            CCRS_DATASETS['strains']['fields'],
            cluster_attr,
            node_attr,
        )

        # Define the relationships.
        areas >> licensees
        plants >> licensees
        plants >> strains
        plants >> areas
        inventory >> strains
        inventory >> products
        inventory >> licensees
        inventory >> areas
        inventory_adjustments >> inventory
        lab_results >> inventory
        lab_results >> licensees
        plant_destructions >> plants
        products >> licensees
        sale_details >> sale_headers
        sale_details >> inventory
        sale_headers >> licensees

        # Define a CCRS node.
        ccrs = Custom(
            'WSLCB CCRS',
            icon_path=logo_path,
            fontsize='21',
            fontname='Times-Roman bold',
            fontcolor='#24292e',
            margin='1.0',
        )
        edge = Edge(
            label='Licensees submit data for traceability.\nWSLCB provides all datasets on FOIA request.',
            color='red',
            style='dashed',
        )
        ccrs << edge >> licensees
    
    # Return the diagram.
    return diagram


# === Test ===
if __name__ == '__main__':

    # Render the diagram.
    diagram = ccrs_diagram(
        filename='ccrs_diagram',
        direction='LR',
        cluster_attr={
            'fontname': 'times bold',
            'fontsize': '18',
        },
        graph_attr={
            'center': 'true',
            'pad': '1.0',
            'fontsize': '25',
            'fontname': 'Times-Roman',
            'fontcolor': '#000',
            'nodesep': '1.0',
            
        },
        node_attr={
            'bgcolor': '#e8e8e8',
            'fixedsize': 'false',
            'labelloc': 't',
            'margin': '0.1',
            'color': '#e8e8e8',
        },
    )
    print('Diagram rendered:', diagram.filename)
