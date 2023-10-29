import pandas as pd
import random
import math
import os
import numpy as np

# Definición de las funciones para generar voltajes
high_voltage = lambda: random.randint(600, 800)
low_voltage = lambda: random.randint(300, 400)

# Resistividad constante
R = 0.5

# Función para calcular la distancia euclídea entre dos puntos
def euclidean_distance(p1, p2):
    return np.round(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2), 2)

# Función para parsear el archivo de conexiones
def parse_connections_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    connections = []
    for line in lines:
        if not line == '\n':
            nodes = line.strip().split(' - ')
            start_node = nodes[0]
            end_nodes = nodes[1].split(', ')

            # Create connections for each pair
            for end_node in end_nodes:
                connections.append((start_node, end_node))

    return connections

# Función para parsear el archivo de nodos
def parse_nodes_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    nodes = {}
    node_id_counter = 1
    for line in lines:
        if not line == '\n':
            parts = line.strip().split(', ')
            node_id = node_id_counter
            node_type = parts[1].split(':')[1]
            consumption = parts[2].split(':')[1]
            position = tuple(map(float, parts[3].split(':')[1].replace('(','').replace(')','').split(';') ))
            nodes[node_id] = {"node_id": node_id, "name": parts[0] ,"type": node_type, "consumption": consumption, "position": position}

            node_id_counter += 1
    return nodes

def parse_types_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    types = {}
    node_id_counter = 1
    for line in lines:
        if not line == '\n':
            parts = line.strip().split(', ')
            types[parts[0].split(':')[1]] = {part.split(':')[0]:part.split(':')[1] for i, part in enumerate(parts)}
    return types

def parse_distributions_file(filename):
    df = pd.read_csv(filename)
    return df

# Leer las conexiones y nodos
indications_directory = 'indications'
connections = parse_connections_file(os.path.join('.', indications_directory, 'Connections.txt'))
nodes = parse_nodes_file(os.path.join('.', indications_directory, 'Nodes.txt'))
types = parse_types_file(os.path.join('.', indications_directory, 'Types.txt'))
distributions = parse_distributions_file(os.path.join('.', indications_directory, 'distributions.csv'))


### --- connections

# Convertir las conexiones en un DataFrame
df = pd.DataFrame(connections, columns=['node_1', 'node_2'])
df['existing_connection'] = 1

# Mapear los nodos a enteros y añadir información de voltaje, resistividad y distancia
nodes_combined = pd.concat([df['node_1'], df['node_2']]).unique()
node_mapping = {node: idx+1 for idx, node in enumerate(nodes_combined)}
df.replace(node_mapping, inplace=True)


# Añadir voltaje, resistividad y distancia
voltages = [[] for _ in range(len(nodes))]
voltage_type = [[] for _ in range(len(nodes))]
distances = []

for idx, row in df.iterrows():
    node_1_name = nodes[row['node_1']]['name']
    node_2_name = nodes[row['node_2']]['name']

    if not "C" in node_1_name and not "C" in node_2_name:
        if row['node_1'] <= row['node_2']:
            voltage = high_voltage()
            voltages[row['node_1']-1].append(voltage)
            voltages[row['node_2']-1].append(voltage)
            voltage_type[row['node_1']-1].append('HV')
            voltage_type[row['node_2']-1].append('HV')

    else:
        if row['node_1'] <= row['node_2']:
            voltage = low_voltage()
            voltages[row['node_1']-1].append(voltage)
            voltages[row['node_2']-1].append(voltage)
            voltage_type[row['node_1']-1].append('LV')
            voltage_type[row['node_2']-1].append('LV')
    
    position_1 = nodes[row['node_1']]['position']
    position_2 = nodes[row['node_2']]['position']
    distances.append(euclidean_distance(position_1, position_2))

df['voltage'] = [item for sublist in voltages for item in sublist]
df['type'] = [item for sublist in voltage_type for item in sublist]
df['resistivity'] = R
df['distance'] = distances


# Determinar el número total de nodos y generar conexiones no existentes
total_nodes = max(df['node_1'].max(), df['node_2'].max())
rows_list = []

for i in range(1, total_nodes + 1):
    for j in range(1, total_nodes + 1):
        if not ((df["node_1"] == i) & (df["node_2"] == j)).any():
            row = {"node_1": i, "node_2": j, "existing_connection": 0, "voltage": 0, "type":'x', "resistivity": R, "distance": 0}
            rows_list.append(row)

no_connections = pd.DataFrame(rows_list)
connections_df = pd.concat([df, no_connections]).sort_values(by=["node_1", "node_2"])


### --- nodes

nodes_df = pd.DataFrame.from_dict(nodes).T.drop('name', axis=1)

production_types = nodes_df['type'].unique()
production_type_encoding = pd.DataFrame(columns=production_types).drop('none', axis=1)

for i, value in enumerate(nodes_df['type']):
    row = [int(value == column) for column in list(production_type_encoding.columns)]
    row_df = pd.DataFrame([row], columns=production_type_encoding.columns)
    production_type_encoding = pd.concat([production_type_encoding, row_df], axis=0, ignore_index=True)

nodes_df = pd.concat([nodes_df.reset_index(drop=True), production_type_encoding.reset_index(drop=True)], axis=1)


### --- types

types_df = pd.DataFrame.from_dict(types).T.drop('type', axis=1)



# Guardar los resultados en un fichero CSV
connections_df.to_csv("connections.csv", index=False)
nodes_df.to_csv("nodes.csv", index=False)
types_df.to_csv("types.csv", index=False)
distributions.to_csv('distributions.csv', index=False)

