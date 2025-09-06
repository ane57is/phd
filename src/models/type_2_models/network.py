from src.parameters.params import (
    smallpox_param_sets,
    measles_param_sets,
    cchf_param_sets,
    evd_param_sets,
    lassa_param_sets
)
import networkx as nx
import random

from copy import deepcopy
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from tabulate import tabulate
import os


def create_community_network(num_pagans, num_christians):
    """Creates a social network with pagan and Christian nodes."""
    G = nx.Graph()

    # Add pagan nodes
    for i in range(num_pagans):
        G.add_node(f'Pagan_{i}', group='pagan', status='susceptible')

    # Add Christian nodes
    for i in range(num_christians):
        G.add_node(f'Christian_{i}', group='christian', status='susceptible')

    # Create random social connections
    nodes = list(G.nodes())
    for _ in range(int(len(nodes) * 5)):  # Create an average of 2 connections per node
        u, v = random.sample(nodes, 2)
        G.add_edge(u, v)

    return G


def simulate_epidemic(graph, christian_mortality, pagan_mortality, initial_infected, reproduction_number=15):
    """Simulates an epidemic with differential mortality rates."""

    # Infect initial individuals
    infected_nodes = random.sample(list(graph.nodes()), initial_infected)
    for node in infected_nodes:
        graph.nodes[node]['status'] = 'infected'

    newly_infected = set(infected_nodes)
    while newly_infected:
        current_infected = list(newly_infected)
        newly_infected = set()
        if reproduction_number is not None:
            for node in current_infected:
                susceptible_neighbors = [n for n in graph.neighbors(node) if graph.nodes[n]['status'] == 'susceptible']
                num_to_infect = min(int(reproduction_number), len(susceptible_neighbors))
                to_infect = random.sample(susceptible_neighbors, num_to_infect)
                for n in to_infect:
                    graph.nodes[n]['status'] = 'infected'
                    newly_infected.add(n)
        else:
            # Infect a random set of susceptible nodes (like initial infection)
            susceptible_nodes = [n for n in graph.nodes if graph.nodes[n]['status'] == 'susceptible']
            if not susceptible_nodes:
                break
            # Infect the same number as current infected, or all if fewer left
            num_to_infect = min(len(current_infected), len(susceptible_nodes))
            to_infect = random.sample(susceptible_nodes, num_to_infect)
            for n in to_infect:
                graph.nodes[n]['status'] = 'infected'
                newly_infected.add(n)

    # Apply mortality
    nodes_to_remove = []
    for node, data in graph.nodes(data=True):
        if data['status'] == 'infected':
            mortality_rate = christian_mortality if data['group'] == 'christian' else pagan_mortality
            if random.random() < mortality_rate:
                nodes_to_remove.append(node)

    graph.remove_nodes_from(nodes_to_remove)

    return graph


def analyze_network_ties(graph):
    """Analyzes the composition of social ties in the network."""
    christian_to_christian = 0
    pagan_to_pagan = 0
    christian_to_pagan = 0

    for u, v in graph.edges():
        group_u = graph.nodes[u]['group']
        group_v = graph.nodes[v]['group']

        if group_u == 'christian' and group_v == 'christian':
            christian_to_christian += 1
        elif group_u == 'pagan' and group_v == 'pagan':
            pagan_to_pagan += 1
        else:
            christian_to_pagan += 1

    return {
        "C-C ties": christian_to_christian,
        "P-P ties": pagan_to_pagan,
        "C-P ties": christian_to_pagan
    }

def run_epidemic_scenarios(diseases, output_path="epidemic_tie_results.csv"):
    """
    Runs simulate_epidemic for each disease, collects tie stats, prints a table, and saves as CSV.
    """
    # Prepare initial network for reference
    num_pagans = 500000
    num_christians = int(num_pagans * 0.004)
    initial_network = create_community_network(num_pagans, num_christians)
    initial_ties = analyze_network_ties(initial_network)

    results = []
    for disease, params in diseases.items():
        # Fresh copy of the network for each run
        network = initial_network.copy()
        surviving_network = simulate_epidemic(
            network,
            params["christian_mortality"],
            params["pagan_mortality"],
            params["initial_infected"],
            params["reproduction_number"]
        )
        surviving_ties = analyze_network_ties(surviving_network)
        total_surviving_ties = sum(surviving_ties.values())

        row = {
            "Disease": disease,
            "C-C ties": surviving_ties["C-C ties"],
            "P-P ties": surviving_ties["P-P ties"],
            "C-P ties": surviving_ties["C-P ties"],
            "Total surviving ties": total_surviving_ties,
        }

        for tie_type in initial_ties:
            initial = initial_ties[tie_type]
            surviving = surviving_ties[tie_type]
            percent = (surviving / initial * 100) if initial > 0 else 0
            row[f"{tie_type} (% of surviving)"] = f"{percent:.2f}%"
        results.append(row)




    df = pd.DataFrame(results)
    print("\n" + tabulate(df, headers="keys", tablefmt="github", showindex=False) + "\n")
    df.to_csv(output_path, index=False)
    print(f"[Saved] CSV output written to: {os.path.abspath(output_path)}")


def run_epidemic_scenarios_n_times(diseases,total_population=10_000, percentage_christians=0.004, n=10, to_csv=False, output_path="epidemic_tie_ranges.csv"):
    """
    Runs run_epidemic_scenarios n times for each disease, collects min/max of tie counts and
    ranges of percentage of surviving ties for each type.
    """
    total_population = total_population
    percentage_christians = percentage_christians  # 0.4% Christians
    # percentage_christians = 0.2  # 20% Christians
    # percentage_christians = 0.25  # 25% Christians
    num_pagans = int(total_population - total_population * percentage_christians)
    num_christians = int(total_population * percentage_christians)

    results = []
    for disease, params in diseases.items():
        tie_counts = {"C-C ties": [], "P-P ties": [], "C-P ties": []}
        percent_surviving = {"C-C ties": [], "P-P ties": [], "C-P ties": []}

        for i in range(n):
            initial_network = create_community_network(num_pagans, num_christians)
            G_initial = deepcopy(initial_network)
            initial_ties = analyze_network_ties(initial_network)

            network = initial_network.copy()
            surviving_network = simulate_epidemic(
                graph=network,
                christian_mortality=params["christian_mortality"],
                pagan_mortality=params["pagan_mortality"],
                initial_infected=int(0.01 * total_population),  # 1% of total population infected
                reproduction_number=params["reproduction_number"]
            )
            G_final = deepcopy(surviving_network)
            surviving_ties = analyze_network_ties(surviving_network)
            for tie_type in initial_ties:
                tie_counts[tie_type].append(surviving_ties[tie_type])
                initial = initial_ties[tie_type]
                percent = (surviving_ties[tie_type] / initial * 100) if initial > 0 else 0
                percent_surviving[tie_type].append(percent)

            if i == 0:
                print(list(network.nodes(data=True))[:5])

                plot_pre_post_clean(
                    G_initial,
                    G_final,
                    disease=disease,
                    total_population=total_population,
                    percentage_christians=percentage_christians,
                    # save_path="figure_2_network_pre_post.png"
                    save_path=f"figure_2_network_pre_post_{disease.replace(' ', '_').lower()}_stark_{str(percentage_christians).replace('.', '')}.png"
                )

        labels = {
            "C-C ties": "Christian-Christian ties",
            "P-P ties": "Pagan-Pagan ties",
            "C-P ties": "Christian-Pagan ties"

        }
        row = {"Disease": disease}
        for tie_type in initial_ties:
            min_count = min(tie_counts[tie_type])
            max_count = max(tie_counts[tie_type])
            avg_percent = sum(percent_surviving[tie_type]) / n
            min_percent = min(percent_surviving[tie_type])
            max_percent = max(percent_surviving[tie_type])
            row[f"{labels[tie_type]} count [min, max]"] = f"[{min_count}, {max_count}]"
            row[f"{labels[tie_type]} percent [min, max]"] = f"{avg_percent:.2f} [{min_percent:.2f}, {max_percent:.2f}]"
        # Add the check column
        cc_avg = sum(percent_surviving["C-C ties"]) / n
        cp_avg = sum(percent_surviving["C-P ties"]) / n
        pp_avg = sum(percent_surviving["P-P ties"]) / n
        row["CC>PP avg"] = "✅" if cc_avg > pp_avg else "❌"
        row["CP>PP avg"] = "✅" if cp_avg > pp_avg else "❌"

        results.append(row)

    df = pd.DataFrame(results)

    print("\n" + tabulate(df, headers="keys", tablefmt="github", showindex=False) + "\n")
    if to_csv:
        df.to_csv(output_path, index=False)
        print(f"[Saved] CSV output written to: {os.path.abspath(output_path)}")


def plot_pre_post_clean(G_before, G_after, disease, total_population, percentage_christians, save_path=None):
    """
    Plot side-by-side networks (pre/post pandemic) with a clean, white style.
    Nodes use:
      - 'religion': 'Christian' or 'Pagan'
      - 'status': 'alive' or 'dead'
    """
    def get_color(attrs):
        group = attrs.get('group', '').lower()
        status = attrs.get('status', '').lower()

        if status == 'dead':
            return '#888888'  # gray
        elif group == 'christian':
            return '#d62728'  # red
        elif group == 'pagan':
            return '#2ca02c'  # green
        return '#1f77b4'  # fallback

    def draw_graph(ax, G, pos, title):
        node_colors = [get_color(G.nodes[n]) for n in G]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, edgecolors='black',
                               linewidths=0.3, node_size=15, alpha=0.9)
        # nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', width=0.4, alpha=0.6)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=1.2, alpha=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()

    # Shared layout to highlight structure change
    # pos = nx.spring_layout(G_before, seed=42)
    pos = nx.spring_layout(G_before, seed=42, scale=1.9, k=0.08)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    draw_graph(axes[0], G_before, pos, "Before pandemic")
    draw_graph(axes[1], G_after, pos, "After pandemic")

    # Legend
    legend = [
        mpatches.Patch(color='#d62728', label='Christian (alive)'),
        mpatches.Patch(color='#2ca02c', label='Pagan (alive)'),
        mpatches.Patch(color='#888888', label='Dead (any group)')
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(
        f"Pandemic of {disease} with CFRs per Stark 1996 applied on initial\npopulation of {total_population:,}".replace(',', ' ') +
        f" individuals ({percentage_christians*100}% of the population being Christians)",
        fontsize=14
    )
    fig.subplots_adjust(top=0.90)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    def create_tables_5678(total_population=10_000, n=1, print_csv=False):
        diseases_literature_cfr = {
            "Stark's example": {
                "christian_mortality": 0.1,
                "pagan_mortality": 0.3,
                "initial_infected": 100,
                "reproduction_number": None
            },
            "Smallpox": {
                "christian_mortality": smallpox_param_sets["without_conversion_literature_cfr"].fatality_rate_c,
                "pagan_mortality": smallpox_param_sets["without_conversion_literature_cfr"].fatality_rate_p,
                "initial_infected": 100,
                "reproduction_number": smallpox_param_sets["without_conversion_literature_cfr"].beta / smallpox_param_sets["without_conversion_literature_cfr"].gamma
            },
            "Measles": {
                "christian_mortality": measles_param_sets["without_conversion_literature_cfr"].fatality_rate_c,
                "pagan_mortality": measles_param_sets["without_conversion_literature_cfr"].fatality_rate_p,
                "initial_infected": 100,
                "reproduction_number": measles_param_sets["without_conversion_literature_cfr"].beta / measles_param_sets["without_conversion_literature_cfr"].gamma
            },
            "CCHF": {
                "christian_mortality": cchf_param_sets["without_conversion_literature_cfr"].fatality_rate_c,
                "pagan_mortality": cchf_param_sets["without_conversion_literature_cfr"].fatality_rate_p,
                "initial_infected": 100,
                "reproduction_number": cchf_param_sets["without_conversion_literature_cfr"].beta / cchf_param_sets["without_conversion_literature_cfr"].gamma
            },
            "EVD": {
                "christian_mortality": evd_param_sets["without_conversion_literature_cfr"].fatality_rate_c,
                "pagan_mortality": evd_param_sets["without_conversion_literature_cfr"].fatality_rate_p,
                "initial_infected": 100,
                "reproduction_number": evd_param_sets["without_conversion_literature_cfr"].beta / evd_param_sets["without_conversion_literature_cfr"].gamma
            },
            "Lassa": {
                "christian_mortality": lassa_param_sets["without_conversion_literature_cfr"].fatality_rate_c,
                "pagan_mortality": lassa_param_sets["without_conversion_literature_cfr"].fatality_rate_p,
                "initial_infected": 100,
                "reproduction_number": lassa_param_sets["without_conversion_literature_cfr"].beta / lassa_param_sets["without_conversion_literature_cfr"].gamma
            }
        }
        diseases_hardcoded_cfr = {
            "Stark": {
                "christian_mortality": 0.1,
                "pagan_mortality": 0.3,
                "initial_infected": 100,
                "reproduction_number": None
            },
            "Smallpox": {
                "christian_mortality": smallpox_param_sets["without_conversion_hardcoded_cfr"].fatality_rate_c,
                "pagan_mortality": smallpox_param_sets["without_conversion_hardcoded_cfr"].fatality_rate_p,
                "initial_infected": 100,
                "reproduction_number": smallpox_param_sets["without_conversion_hardcoded_cfr"].beta / smallpox_param_sets[
                    "without_conversion_hardcoded_cfr"].gamma
            },
            "Measles": {
                "christian_mortality": measles_param_sets["without_conversion_hardcoded_cfr"].fatality_rate_c,
                "pagan_mortality": measles_param_sets["without_conversion_hardcoded_cfr"].fatality_rate_p,
                "initial_infected": 100,
                "reproduction_number": measles_param_sets["without_conversion_hardcoded_cfr"].beta / measles_param_sets[
                    "without_conversion_hardcoded_cfr"].gamma
            },
            "CCHF": {
                "christian_mortality": cchf_param_sets["without_conversion_hardcoded_cfr"].fatality_rate_c,
                "pagan_mortality": cchf_param_sets["without_conversion_hardcoded_cfr"].fatality_rate_p,
                "initial_infected": 100,
                "reproduction_number": cchf_param_sets["without_conversion_hardcoded_cfr"].beta / cchf_param_sets[
                    "without_conversion_hardcoded_cfr"].gamma
            },
            "EVD": {
                "christian_mortality": evd_param_sets["without_conversion_hardcoded_cfr"].fatality_rate_c,
                "pagan_mortality": evd_param_sets["without_conversion_hardcoded_cfr"].fatality_rate_p,
                "initial_infected": 100,
                "reproduction_number": evd_param_sets["without_conversion_hardcoded_cfr"].beta / evd_param_sets[
                    "without_conversion_hardcoded_cfr"].gamma
            },
            "Lassa": {
                "christian_mortality": lassa_param_sets["without_conversion_hardcoded_cfr"].fatality_rate_c,
                "pagan_mortality": lassa_param_sets["without_conversion_hardcoded_cfr"].fatality_rate_p,
                "initial_infected": 100,
                "reproduction_number": lassa_param_sets["without_conversion_hardcoded_cfr"].beta / lassa_param_sets[
                    "without_conversion_hardcoded_cfr"].gamma
            }
        }

        print("\n\nTable 8 Tie survival rates with fatality rate based on literature and 0.4% of population being Christian (2a)")
        run_epidemic_scenarios_n_times(
            # diseases_literature_cfr,
            diseases_hardcoded_cfr,
            total_population=total_population,
            # percentage_christians=0.004,
            percentage_christians=0.2,
            n=n,
            to_csv=print_csv,
            output_path="table_11_2b_02.csv"
        )

        # print("\n\nTable 6 Tie survival rates with fatality rate based on literature and 20% of population being Christian (2a)")
        # run_epidemic_scenarios_n_times(
        #     diseases_literature_cfr,
        #     # diseases_hardcoded_cfr,
        #     total_population=total_population,
        #     # percentage_christians=0.004,
        #     percentage_christians=0.2,
        #     n=n,
        #     to_csv=print_csv,
        #     output_path="table_6_2a_02.csv"
        # )
        #
        # print("\n\nTable 7 Tie survival rates with hardcoded lower fatality rate and 0.4% of population being Christian (2b).")
        # run_epidemic_scenarios_n_times(
        #     # diseases_literature_cfr,
        #     diseases_hardcoded_cfr,
        #     total_population=total_population,
        #     percentage_christians=0.004,
        #     # percentage_christians=0.2,
        #     n=n,
        #     to_csv=print_csv,
        #     output_path="table_7_2b_0004.csv"
        # )
        #
        # print("\n\nTable 8 Tie survival rates with hardcoded lower fatality rate and 20% of population being Christian (2b).")
        # run_epidemic_scenarios_n_times(
        #     # diseases_literature_cfr,
        #     diseases_hardcoded_cfr,
        #     total_population=total_population,
        #     # percentage_christians=0.004,
        #     percentage_christians=0.2,
        #     n=n,
        #     to_csv=print_csv,
        #     output_path="table_8_2b_02.csv"
        # )


    create_tables_5678(total_population=10_000, n=100, print_csv=True)
