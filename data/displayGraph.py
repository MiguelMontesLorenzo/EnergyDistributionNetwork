import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Leer el fichero CSV
df = pd.read_csv("connections.csv")

# Crear un grafo dirigido
G = nx.DiGraph()

# Añadir nodos al grafo
nodes = set(df["node_1"]) | set(df["node_2"])
G.add_nodes_from(nodes)

# Añadir aristas al grafo
for index, row in df.iterrows():
    if row["existing_connection"] == 1:
        G.add_edge(row["node_1"], row["node_2"])

# Dibujar el grafo
pos = nx.spring_layout(G)  # Usar la disposición spring para la visualización
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=15)

plt.title("Grafo de conexiones")
plt.show()
