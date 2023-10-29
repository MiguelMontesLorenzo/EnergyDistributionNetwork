import pandas as pd
import os
import math
import random
import ego

import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# SCRIPT PARAMETERS

# Declare whether a video (showing network optimal evolution) is wanted or not
MAKE_VIDEO = True

# Declare which optimizer to use (If you dont want to specify optimizer path just set it to None)
optimizer = 'glpk'
optimizer_path = None #'C:\\Program Files\\winglpk-4.65\\glpk-4.65\\w64\\glpsol.exe'



wind_production = lambda: random.randint(400,3000)


def load_and_prepare_data():
    data_directory = 'data'

    # Load datasets
    nodes = pd.read_csv(os.path.join('.', data_directory, 'nodes.csv'))
    connections = pd.read_csv(os.path.join('.', data_directory, 'connections.csv'))
    types = pd.read_csv(os.path.join('.', data_directory, 'types.csv'))
    distributions = pd.read_csv(os.path.join('.', data_directory, 'distributions.csv'))

    # Create items for sets
    Nodes = [node_id for node_id in nodes['node_id']]
    Time  = [i+1 for i,_ in enumerate(distributions['consumption'])]
    Types = [type_id for type_id in types['type_id']] + [len(types['type_id'])+1]

    nodes_amount = len(Nodes)
    time_units_amount = len(Time)
    types_amount = len(Types)


    # Node types
    columns = ['normal', 'hydraulic', 'solar', 'eolic']
    W = [[row[col] for col in columns] + [1 - sum(row[col] for col in columns)] for _, row in nodes.iterrows()]

    # Demand
    DEM = [(distributions['consumption']*factor).to_list() for factor in nodes['consumption'].tolist() ]

    # Fixed productions
    SOLAR = [(distributions['solar']*factor).to_list() for factor in nodes['solar'].tolist() ]
    WIND = [[wind_production()*factor for _ in Time] for factor in nodes['eolic'].tolist() ]

    # Production types
    # PrTy - Productions
    MAXPROD = [max_prod for max_prod in types['max_prod']] + [0]
    MINPROD = [min_prod for min_prod in types['min_prod']] + [0]
    THRESHOLD = [prod for prod in types['threshold_prod']] + [0]

    # PrTy - Costs
    UNITARYCOST = [cost for cost in types['unitary_cost']] + [0]
    FXCOST = [fx_prod for fx_prod in types['fixed_cost']] + [0]
    TRNONCOST = [on_cost for on_cost in types['on_cost']] + [0]
    TRNOFFCOST = [off_cost for off_cost in types['off_cost']] + [0]

    # Connections
    CONN = [[0 for _ in range(nodes_amount)] for _ in range(nodes_amount)]
    L = [[0 for _ in range(nodes_amount)] for _ in range(nodes_amount)]
    V = [[0 for _ in range(nodes_amount)] for _ in range(nodes_amount)]
    R = [[0 for _ in range(nodes_amount)] for _ in range(nodes_amount)]
    MAXPOWERFLOW = [[3000 for _ in range(nodes_amount)] for _ in range(nodes_amount)]

    for index, row in connections.iterrows():
        CONN[int(row['node_1']-1)][int(row['node_2']-1)] = row['existing_connection']
        L[int(row['node_1']-1)][int(row['node_2']-1)] = row['distance']
        V[int(row['node_1']-1)][int(row['node_2']-1)] = row['voltage']
        R[int(row['node_1']-1)][int(row['node_2']-1)] = row['resistivity']

        # MAXPOWERFLOW[row['node_1']-1][row['node_2']-1] = row['distance']


    # H, PH, y F están en types en lugar de en nodes. Esto implica que todas las presas tendrán la misma producción
    # Si quisiéramos poner dichas variaables en nodes lo haríamos de la siguiente manera:

    H  = [float(types.loc[types['type_id'] == 2]['H'].values[0]) for is_hidraulic in nodes['hydraulic']]
    PH = [float(types.loc[types['type_id'] == 2]['PH'].values[0]) for is_hidraulic in nodes['hydraulic']]


    node_positions = {row['node_id']: list(map(float, row['position'].replace('(','').replace(')','').split(','))) for _, row in nodes.iterrows()}
    voltage_types = { (i,j):connections.loc[(connections['node_1'] == i) & (connections['node_2'] == j), 'type'].iloc[0] for i in nodes['node_id'] for j in nodes['node_id']} 


    sets = {
        'Nodes': Nodes,
        'PrTy': Types,
        'Time': Time
    }

    params = {
        'DEM': DEM,
        'SOLAR': SOLAR,
        'WIND': WIND,
        'MAXPROD': MAXPROD,
        'MINPROD': MINPROD,
        'THRESHOLD': THRESHOLD,
        'UNITARYCOST': UNITARYCOST,
        'FXCOST': FXCOST,
        'TRNONCOST': TRNONCOST,
        'TRNOFFCOST': TRNOFFCOST,
        'CONN': CONN,
        'L': L,
        'V': V,
        'R': R,
        'MAXPOWERFLOW': MAXPOWERFLOW,
        'H': H,
        'PH': PH,
        'W': W
    }

    extra = {
        'NodePositons': node_positions,
        'VoltageTypes': voltage_types
    }

    return sets, params, extra



def load_and_prepare_dictionaries():


    def convert_to_dict_1d(data, set1):
        return {item1: data[item1-1] for item1 in set1}

    def convert_to_dict_2d(data, set1, set2):
        # set1: el que itera por filas
        # set2: el que itera por columnas
        return {(item1, item2): data[item1-1][item2-1] for item1 in set1 for item2 in set2}


    sets, params, extra = load_and_prepare_data()

    for key, data in params.items():

        if type(data[0]) == list:
            key_sets = [set_values for set_values in sets.values() for dim_lenght in [len(data), len(data[0])] if len(set_values) == dim_lenght]
            params[key] = convert_to_dict_2d(data, *key_sets)
        else:
            key_set = [set_values for set_values in sets.values() if len(set_values) == len(data)]
            params[key] = convert_to_dict_1d(data, *key_set)

            

    return sets, params, extra




def generate_graph(node_positions, connections, voltage_types, diffs, e_values, t, breakpoints, lims, save_as_image=False):

    # GENERATE GRAPH (NODES + EDGES)

    G = nx.DiGraph()  # Create a directed graph

    # Add nodes
    for node, position in node_positions.items():
        G.add_node(node, pos=position)

    # Add edges based on the connections and the direction of the power flow
    for pair_of_nodes, existing_connection in connections.items():
        i, j = pair_of_nodes
        if existing_connection == 1:
            if e_values[(i, j, t)] >= 0:
                G.add_edge(i, j)
            else:
                G.add_edge(j, i)


    pos = nx.get_node_attributes(G, 'pos')

    node_values = [diffs[(i,t)] for i in G.nodes()]
    edge_values = [[abs(e_values[(i, j, t)]), voltage_types[(i,j)]]  for i, j in G.edges()]


    # COLORS

    # Create a custom colormap using LinearSegmentedColormap
    colors = ['red', 'white', 'lime']
    node_cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(zip(breakpoints, colors)), N=256)

    # Custom colormap for edges: from gray (0) to yellow (max)
    # colors_edges = ['whitesmoke', 'yellow']
    colors_edges = [(0.0, 'whitesmoke'), (0.6, 'khaki'), (1.0, 'yellow')]
    edge_cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap_edges', colors_edges, N=256)

    # Normalize the values
    node_norm = mcolors.Normalize(vmin=lims['diffs'][0], vmax=lims['diffs'][1])
    edge_norm = mcolors.Normalize(vmin=0, vmax=lims['edges'][1])

    # Get colors for nodes and edges
    # node_colors = [node_cmap(node_norm(value)) for value in node_values]
    node_colors = [node_cmap(node_norm(value)) for value in node_values]
    edge_colors = [edge_cmap(edge_norm(value)) for value,_ in edge_values]


    # PLOTTING

    factor = 2

    node_size = int(200*factor)
    edge_width = int(1.5*factor)
    show_node_labels = True
    show_edge_labels = True
    label_fontsize = int(6*factor)

    # plt.figure(figsize=(16, 9))
    fig = plt.figure(figsize=(19.20, 10.88), dpi=100)
    # nx.draw(G, pos, with_labels=False, node_color=node_colors, edge_color=edge_colors, edgecolors='black', width=edge_width, node_size=node_size)
    nx.draw(G, pos, arrows=True, with_labels=False, node_color=node_colors, edge_color=edge_colors, edgecolors='black', width=edge_width, node_size=node_size)


    # Draw labels with modified positions and custom size
    displacement = node_size*0
    if show_node_labels:
        # Modify label position to put them outside the node
        label_pos = {k: [v[0] + displacement, v[1]] for k, v in pos.items()}  # Adjust the 0.05 value for desired offset
        nx.draw_networkx_labels(G, label_pos, font_size=label_fontsize)

    # Agregar etiquetas a las aristas
    if show_edge_labels:
        edge_labels = dict(zip(G.edges(), [f'{volt_typ}\n' + str(int(val)) for val, volt_typ in edge_values]))
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=label_fontsize)
    

    # COLORBARS

    # Add colorbar for nodes and edges
    sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    sm.set_array([])
    ax = plt.gca()
    node_cbar = plt.colorbar(sm, ax=ax, fraction=0.08, shrink=0.8)#, label="Node Net Production [kwh]")
    node_cbar.ax.set_xlabel("Node Net Production [kwh]", labelpad=15)

    sm_edge = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
    sm_edge.set_array([])
    ax = plt.gca()
    edge_cbar = plt.colorbar(sm_edge, ax=ax, fraction=0.08, shrink=0.8)#, label="")#, orientation='horizontal')
    edge_cbar.ax.set_xlabel("Edge PowerFlow [kwh]", labelpad=15)
    
    plt.title(f"Graph at t={t}")


    # LEGEND

    # Calculate the total demand and total production
    total_demand = sum(DEM_values[(i, t)] for i in node_positions.keys())
    total_production = sum(p_values[(i, t)] for i in node_positions.keys())
    
    # Calculate the percentage of production covered by each node with production different from 0
    production_percentages = {i: (p_values[(i, t)] / total_production) * 100 for i in node_positions.keys() if p_values[(i, t)] != 0}

    # Add text annotations
    textstr = f'Time: {t}\nTotal Demand: {total_demand:.2f} kWh\nTotal Production: {total_production:.2f} kWh\n'
    for node, percentage in production_percentages.items():
        textstr += f'Node {node}: {percentage:.2f}%\n'

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax = plt.gca()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)


    # SAVE PLOT AS IMAGE

    # Guardar como imagen o mostrar el gráfico directamente
    if save_as_image:
        if not os.path.exists('images'):
            os.makedirs('images')
        plt.savefig(f'images/graph_at_t_{t}.jpg')
        plt.close()
    else:
        plt.show()


def create_video(node_positions, connections, voltage_types, p_values, DEM_values, e_values, time_array):

    print('\nGenerating Video with results ...')


    diffs = dict()
    for i in node_positions.keys():
        for t in time_array:
            diffs[(i,t)] = p_values[(i,t)] - DEM_values[(i,t)]

    breakpoints = [min(diffs.values()), 0, max(diffs.values())]
    min_ = min(breakpoints)
    max_ = max(breakpoints)
    normalized_breakpoints = [(brp - min_)/(max_ - min_) for brp in breakpoints]

    lims = {'diffs':[min_, max_], 'edges':[min(e_values.values()), max(e_values.values())]}


    for t in time_array:
        generate_graph(node_positions, connections, voltage_types, diffs, e_values, t, normalized_breakpoints, lims, save_as_image=True)
    print()


    images = []
    filenames = [f'images/graph_at_t_{t}.jpg' for t in time_array]

    for filename in filenames:
        image = imageio.imread(filename)
        images.extend([image] * 15)  # Add the same image 15 times

    imageio.mimsave('graph_video.mp4', images, fps=15)





if __name__ == "__main__":

    # sets, params = load_and_prepare_data()
    sets, params, extra = load_and_prepare_dictionaries()
    
    # Instantiate model
    model = ego.ElectricGridOptimization(sets, params)
    
    # Select the optimizer
    model.define_solver_path(optimizer, optimizer_path)

    # Optimize the model
    solution_found, result = model.optimize_problem()

    # Display optimised results
    if solution_found:
        data = model.show_results()

        # Call the function to display the graph
        # display_graph(extra['NodePositons'], params['CONN'])

        if MAKE_VIDEO:

            p_values = data['p']
            e_values = data['e']
            DEM_values = params['DEM']
            time_array = sets['Time']

            create_video(extra['NodePositons'], params['CONN'], extra['VoltageTypes'], p_values, DEM_values, e_values, time_array)

