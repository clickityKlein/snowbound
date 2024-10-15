'''
ARM FUNCTIONS
'''

'''
IMPORTS
'''
# library imports
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import networkx as nx
from pyvis.network import Network
import matplotlib.colors as mcolors

# arm imports
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori as ml_apriori
from mlxtend.frequent_patterns import association_rules as ml_association

'''
FUNCTIONS
'''
# function to plot basic static networkx plot
def create_network(rules_df, antecedent_label=None, save_path=None):
    G = nx.DiGraph()
    consequent_count = dict()
    consequent_confidence = dict()
    consequent_lift = dict()
    for _, row in rules_df.iterrows():
        for antecedent in row['antecedents']:
            if antecedent_label and antecedent != antecedent_label:
                continue
            for consequent in row['consequents']:
                # G.add_edge(antecedent, consequent, weight=row['lift'])
                G.add_edge(antecedent, consequent)
                try:
                    consequent_count[consequent] += 1
                    consequent_confidence[consequent] += row['confidence']
                    consequent_lift[consequent] += row['lift']
                except KeyError:
                    consequent_count[consequent] = 1
                    consequent_confidence[consequent] = row['confidence']
                    consequent_lift[consequent] = row['lift']
    
    # confidence and lift information to nodes
    # note that support isn't applied here since it's over the entire google dataset
    for node in G.nodes():
        # consequents
        try:
            G.nodes[node]['count'] = consequent_count[node]
            G.nodes[node]['confidence'] = consequent_confidence[node] / consequent_count[node]
            G.nodes[node]['lift'] = consequent_lift[node] / consequent_count[node]
        # antecedents
        except KeyError:
                G.nodes[node]['count'] = sum(consequent_count.values())
                G.nodes[node]['confidence'] = sum(consequent_confidence.values()) / sum(consequent_count.values())
                G.nodes[node]['lift'] = sum(consequent_lift.values()) / sum(consequent_count.values())
    
    node_sizes = [G.nodes[node]['count'] * 1000 for node in G]
    
    # populate node edges
    for edge in G.edges():
        G.edges[edge]['distance'] = G.nodes[edge[1]]['lift']
    
    # layout
    pos = {}
    center_x, center_y = 0, 0  # Center of the layout
    angle_step = 2 * np.pi / (len(G.nodes) - 1)
    
    for node_count, node in enumerate(G.nodes()):
        # antecedent
        if node in [a for a, c in G.edges()]:
            pos[node] = (center_x, center_y)
        # consequent
        else:
            # radius -> G.nodes[node]['lift']
            angle = node_count * angle_step
            pos[node] = (center_x + G.nodes[node]['lift'] * np.cos(angle), center_y + G.nodes[node]['lift'] * np.sin(angle))
    
    # draw the network
    fig, ax = plt.subplots(figsize=(20, 20))
    nx.draw(G, pos, with_labels=False, node_size=node_sizes, node_color='lightblue', font_size=20, font_weight='bold', arrowsize=20)
    
    # custom node labels
    node_labels = {node: f"{node}\ncount: {G.nodes[node]['count']}\navg confidence: {G.nodes[node]['confidence']:.2f}\n avg lift: {G.nodes[node]['lift']:.2f}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=20, font_weight='bold')
    
    # custom edge labels
    edge_labels = nx.get_edge_attributes(G, 'distance')
    formatted_edge_labels = {label: f'Lift: {distance:.2f}' for label, distance in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels, font_size=20, font_weight='bold')
    
    # finish plot
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return G

# function to plot pyvis html plot
def create_network_pyvis(rules_df, save_path, antecedent_label=None, bins=5, color_palette=False, edge_multiplier=1):
    G = nx.DiGraph()
    
    consequent_count = dict()
    consequent_confidence = dict()
    consequent_lift = dict()
    ant_cons = {'ant_node': [], 'cons_node': []}
    
    # initialize graph with edges and get consequent metrics
    for _, row in rules_df.iterrows():
        for antecedent in row['antecedents']:
            if antecedent_label and antecedent != antecedent_label:
                continue
            for consequent in row['consequents']:
                G.add_edge(antecedent, consequent)
                try:
                    consequent_count[consequent] += 1
                    consequent_confidence[consequent] += row['confidence']
                    consequent_lift[consequent] += row['lift']
                    ant_cons['ant_node'].append(antecedent)
                    ant_cons['cons_node'].append(consequent)
                except KeyError:
                    consequent_count[consequent] = 1
                    consequent_confidence[consequent] = row['confidence']
                    consequent_lift[consequent] = row['lift']
                    ant_cons['ant_node'].append(antecedent)
                    ant_cons['cons_node'].append(consequent)
    
    # initialize nodes and get average metrics
    for node in G.nodes():
        if node in consequent_count:
            G.nodes[node]['count'] = consequent_count[node]
            G.nodes[node]['confidence'] = consequent_confidence[node] / consequent_count[node]
            G.nodes[node]['lift'] = consequent_lift[node] / consequent_count[node]
        else:
            antecedent_count = sum([1 for row in rules_df['antecedents'] if node in row])
            antecedent_confidence = sum([row['confidence'] for _, row in rules_df.iterrows() if node in row['antecedents']])
            antecedent_lift = sum([row['lift'] for _, row in rules_df.iterrows() if node in row['antecedents']])
            
            G.nodes[node]['count'] = antecedent_count
            G.nodes[node]['confidence'] = antecedent_confidence / antecedent_count if antecedent_count else 0
            G.nodes[node]['lift'] = antecedent_lift / antecedent_count if antecedent_count else 0

    
    # normalize counts
    total_count = sum([G.nodes[node]['count'] for node in G.nodes])
    for node in G.nodes():
        G.nodes[node]['normalized_count'] = (G.nodes[node]['count'] / total_count) * 100
        
    # binned counts
    # while loop with try and except to reduce bin by 1 if error
    while True:
        try:
            bin_df = pd.DataFrame({'node': [node for node in G.nodes], 'count': [G.nodes[node]['count'] for node in G.nodes]})
            bin_df['bin'] = pd.qcut([G.nodes[node]['count'] for node in G.nodes], q=bins, labels=[10 * bin for bin in range(1, bins + 1)])
            break
        except ValueError:
            bins -= 1
            print(f'Bins reduced to {bins}.')
            if bins < 1:
                raise ValueError('Binning process failed. Number of bins too low.')
            
            
    # add bin to node
    for node in G.nodes():
        G.nodes[node]['bin'] = int(bin_df[bin_df['node']==node]['bin'].values[0])
    
    # populate node edges
    for edge in G.edges():
        G.edges[edge]['distance'] = G.nodes[edge[1]]['normalized_count'] * edge_multiplier
    
    # define a color palette
    if color_palette:
        color_palette = plt.cm.get_cmap('tab20c', len(rules_df['antecedents'].unique()))
        color_palette_list = [color_palette(i) for i in range(len(rules_df['antecedents'].unique()))]
        # convert colors to hex
        color_palette_hex = [mcolors.to_hex(color) for color in color_palette_list]
        antecedents_unique = [list(ant)[0] for ant in rules_df['antecedents'].unique()]
        color_map = {antecedent: color_palette_hex[i] for i, antecedent in enumerate(antecedents_unique)}
        # assign consequents to most frequent antecedents
        ant_cons_df = pd.DataFrame(ant_cons)
        ant_cons_max = ant_cons_df.groupby('cons_node')['ant_node'].agg(lambda x: x.value_counts().idxmax()).reset_index()
        # assign per node
        for node in G.nodes():
            if node in consequent_count:
                ant_color = ant_cons_max[ant_cons_max['cons_node']==node]['ant_node'].values[0]
                G.nodes[node]['color_map'] = color_map[ant_color]
            else:
                G.nodes[node]['color_map'] = color_map[node]
    else:
        for node in G.nodes():
            G.nodes[node]['color_map'] = '#caeef0'
        
    # layout
    pos = {}
    center_x, center_y = 0, 0  # Center of the layout
    angle_step = 2 * np.pi / (len(G.nodes) - 1)
    
    for node_count, node in enumerate(G.nodes()):
        # antecedent
        if node in [a for a, c in G.edges()]:
            pos[node] = (center_x, center_y)
        # consequent
        else:
            # radius -> G.nodes[node]['lift']
            angle = node_count * angle_step
            pos[node] = (center_x + G.nodes[node]['lift'] * np.cos(angle), center_y + G.nodes[node]['lift'] * np.sin(angle))
    
    # create network object
    net = Network(height='750px', width='100%', directed=True)
    
    # populate net with nodes data
    for node in G.nodes(data=True):
        net.add_node(node[0], title=f"Node: {node[0]}\nCount: {node[1]['count']}\nAverage Confidence: {node[1]['confidence']:.2f}\nAverage Lift: {node[1]['lift']:.2f}", size=node[1]['bin'], color=node[1]['color_map'])
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], weight=edge[2]['distance'])
    
    return G

# function to subset antecedent of labels
def check_antecedent_in_list(antecedent, category_list):
    return any(category in antecedent for category in category_list)

# function to create single antecedent rules from labels
def create_label_rules(full_rules, category_list):
    # get rules with single antecedents
    rules_categories = full_rules[(full_rules['antecedents'].apply(lambda x: len(list(x))) == 1)]
    
    # subset to antecedent of labels
    rules_categories = rules_categories[rules_categories['antecedents'].apply(lambda x: check_antecedent_in_list(list(x), category_list))]
    
    # return
    return rules_categories

# function to get rules from label
def rules_from_label(apriori_ready_df, list_col, label_col, prefix='', apriori_min_support=0.01, rules_min_threshold=0.01, rules_min_metric='confidence'):
    # create fresh copy of df
    df = apriori_ready_df.copy()
    
    # create list for categories
    categories = [f'{prefix}{category.lower()}' for category in df[label_col].unique()]
    
    # label_col into list_col
    new_lists = df.apply(lambda row: [f"{prefix}{row[label_col].lower()}"] + row[list_col], axis=1)
    
    # initialization
    con = TransactionEncoder()
    con_arr = con.fit_transform(new_lists)
    df_transformed = pd.DataFrame(con_arr, columns=con.columns_)
    
    # run apriori
    model_apriori = ml_apriori(df_transformed, use_colnames=True, min_support=apriori_min_support)
    
    # create assocation rules
    model_rules = ml_association(model_apriori, metric=rules_min_metric, min_threshold=rules_min_threshold)
    
    # create label rules
    rules = create_label_rules(model_rules, categories)
    
    # return rules
    return rules
