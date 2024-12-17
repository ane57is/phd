import networkx as nx
import random
import matplotlib.pyplot as plt

# Parameters
NUM_PEOPLE = 1000  # Reduced size for testing
AVG_CONNECTIONS = 10
INITIAL_INFECTED = 100
INFECTION_RATE = 0.2
RECOVERY_RATE_PAGANS = 0.2
RECOVERY_RATE_CHRISTIANS = 0.6
NUM_ITERATIONS = 10  # Number of iterations to simulate infection spread


# Step 1: Create a simple city network
def create_city_network(num_people, avg_connections):
    p = avg_connections / (num_people - 1)
    G = nx.erdos_renyi_graph(num_people, p)

    # Initialize nodes: All start as susceptible
    for node in G.nodes():
        G.nodes[node]['status'] = 'S'  # S for Susceptible
        G.nodes[node]['religion'] = 'pagan' if random.random() > 0.004 else 'christian'  # 0.4% Christians

    return G


# Step 2: Introduce infection into the network
def introduce_infection(G, initial_infected):
    infected_nodes = random.sample(list(G.nodes()), initial_infected)
    for node in infected_nodes:
        G.nodes[node]['status'] = 'I'  # I for Infected


# Step 3: Spread infection in the network
def spread_infection(G):
    new_infections = []
    recoveries = []

    for node in G.nodes():
        if G.nodes[node]['status'] == 'I':  # Infected nodes can infect others
            # Attempt to infect neighbors
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['status'] == 'S':  # Only susceptible neighbors can be infected
                    if random.random() < INFECTION_RATE:
                        new_infections.append(neighbor)

            # Recovery attempt for the infected node
            recovery_rate = RECOVERY_RATE_CHRISTIANS if G.nodes[node][
                                                            'religion'] == 'christian' else RECOVERY_RATE_PAGANS
            if random.random() < recovery_rate:
                recoveries.append(node)

    # Apply new infections
    for node in new_infections:
        G.nodes[node]['status'] = 'I'

    # Apply recoveries
    for node in recoveries:
        G.nodes[node]['status'] = 'R'  # R for Recovered


# Step 4: Run the simulation
def run_simulation(G, initial_infected, num_iterations):
    introduce_infection(G, initial_infected)

    for _ in range(num_iterations):
        spread_infection(G)

# Step 5: Calculate average number of connections with Christians and pagans
def average_connections(G, religion_type):
    total_christian_connections = 0
    total_pagan_connections = 0
    total_nodes_of_type = 0

    for node in G.nodes():
        if G.nodes[node]['religion'] == religion_type:
            total_nodes_of_type += 1
            christian_neighbors = sum(
                1 for neighbor in G.neighbors(node) if G.nodes[neighbor]['religion'] == 'christian')
            pagan_neighbors = sum(1 for neighbor in G.neighbors(node) if G.nodes[neighbor]['religion'] == 'pagan')
            total_christian_connections += christian_neighbors
            total_pagan_connections += pagan_neighbors

    avg_christian_connections = total_christian_connections / total_nodes_of_type if total_nodes_of_type > 0 else 0
    avg_pagan_connections = total_pagan_connections / total_nodes_of_type if total_nodes_of_type > 0 else 0

    return avg_christian_connections, avg_pagan_connections

# Create the network
city_network = create_city_network(NUM_PEOPLE, AVG_CONNECTIONS)

# Run the simulation
run_simulation(city_network, INITIAL_INFECTED, NUM_ITERATIONS)

# Count final statuses
status_counts = {'S': 0, 'I': 0, 'R': 0}
for node in city_network.nodes():
    status = city_network.nodes[node]['status']
    status_counts[status] += 1

# Print results
print(f"Final counts after simulation: {status_counts}")

# Optional: Visualization of the network (color-coded by status)
color_map = {'S': 'blue', 'I': 'red', 'R': 'green'}
node_colors = [color_map[city_network.nodes[node]['status']] for node in city_network.nodes()]

plt.figure(figsize=(8, 8))
nx.draw(city_network, node_size=10, node_color=node_colors)
plt.show()

# Get average connections for Christians
avg_christian_to_christians, avg_christian_to_pagans = average_connections(city_network, 'christian')
print(f"Average connections for Christians - To Christians: {avg_christian_to_christians}, To Pagans: {avg_christian_to_pagans}")

# Get average connections for Pagans
avg_pagan_to_christians, avg_pagan_to_pagans = average_connections(city_network, 'pagan')
print(f"Average connections for Pagans - To Christians: {avg_pagan_to_christians}, To Pagans: {avg_pagan_to_pagans}")
