import pandas as pd
import networkx as nx
import numpy as np

airports_path = "./data/airports.dat"
routes_path = "./data/routes.dat"

airport_cols = ['Airport_ID', 'Name', 'City', 'Country', 'IATA', 'ICAO',
                'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz', 'Type', 'Source']
airports = pd.read_csv(airports_path, header=None, names=airport_cols, na_values='\\N')

route_cols = ['Airline', 'Airline_ID', 'Source_airport', 'Source_ID', 
              'Dest_airport', 'Dest_ID', 'Codeshare', 'Stops', 'Equipment']
routes = pd.read_csv(routes_path, header=None, names=route_cols, na_values='\\N')

print(f"Loaded {len(airports)} airports and {len(routes)} routes.")

routes = routes.dropna(subset=['Source_ID', 'Dest_ID'])
routes = routes[routes['Source_ID'] != routes['Dest_ID']]

routes['Source_ID'] = routes['Source_ID'].astype(int)
routes['Dest_ID'] = routes['Dest_ID'].astype(int)

valid_ids = set(airports['Airport_ID'].dropna().astype(int))
routes = routes[routes['Source_ID'].isin(valid_ids) & routes['Dest_ID'].isin(valid_ids)]

print(f"Cleaned routes: {len(routes)} remaining after filtering invalid airports.")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

edges = routes.merge(airports[['Airport_ID','Latitude','Longitude']],
                     left_on='Source_ID', right_on='Airport_ID', how='left')
edges = edges.merge(airports[['Airport_ID','Latitude','Longitude']],
                     left_on='Dest_ID', right_on='Airport_ID', how='left', suffixes=('_src','_dst'))

edges['Distance_km'] = edges.apply(lambda r: haversine(r.Latitude_src, r.Longitude_src,
                                                       r.Latitude_dst, r.Longitude_dst), axis=1)

print("Sample distances:\n", edges['Distance_km'].head())

G = nx.Graph()

for _, row in airports.iterrows():
    if pd.notnull(row['Airport_ID']):
        G.add_node(int(row['Airport_ID']),
                   name=row['Name'],
                   country=row['Country'],
                   lat=row['Latitude'],
                   lon=row['Longitude'])

for _, row in edges.iterrows():
    G.add_edge(int(row['Source_ID']), int(row['Dest_ID']),
               distance=row['Distance_km'])

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("\nNetwork summary:")
print(f"Graph summary:")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Is directed: {G.is_directed()}")

largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()
print(f"Largest connected component has {G_lcc.number_of_nodes()} nodes "
      f"and {G_lcc.number_of_edges()} edges.")

avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
print(f"Average degree: {avg_degree:.2f}")

avg_path_length = nx.average_shortest_path_length(G_lcc)
print(f"Average shortest path length (LCC): {avg_path_length:.2f}")

avg_clustering = nx.average_clustering(G)
print(f"Average clustering coefficient: {avg_clustering:.3f}")

nx.write_gml(G, "./output/global_airline_network.gml")
print("Graph saved to ./output/global_airline_network.gml")
