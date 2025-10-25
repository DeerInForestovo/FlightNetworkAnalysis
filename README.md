# FlightNetworkAnalysis
CMU 18755 Project 2025 Fall

Files:

+ ```network_build.py```: build the network
    + INPUT: ```data/airpors.data,data/routes.dat```, ref: https://openflights.org/data
    + OUTPUT: ```output/global_airline_network.gml```
+ ```visualization.py```: visualize the network with Plotly
    + OUTPUT: ```output/global_airline_network.html```
+ ```analysis.py```: analysis and clustering
    + OUTPUT: ```output/top10_hubs.csv,output/centrality_metrics.csv```
+ ```clustering_visual.py```: visulize the clustering
    + OUTPUT: ```global_airline_centrality_map.html```