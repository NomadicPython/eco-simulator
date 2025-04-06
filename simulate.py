#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from community import Community
import scienceplots
import pandas as pd
import networkx as nx

plt.style.use(["science"])


def save_figure(fig: plt.Figure, filename: str):
    """
    Save the plot to a file in multiple formats.

    :param fig: Matplotlib figure to save
    :param filename: base filename for the plot (with path)
    """
    fig.savefig(filename + ".svg", dpi=600, bbox_inches="tight", transparent=False)
    fig.savefig(
        filename + ".pdf",
        dpi=600,
        bbox_inches="tight",
        transparent=False,
    )
    fig.savefig(
        filename + "_transparent.svg", dpi=600, bbox_inches="tight", transparent=True
    )
    plt.close(fig)


def plot_species_and_resources(
    t: np.ndarray,
    y: np.ndarray,
    split_index: int,
    species_names: list[str],
    resource_names: list[str],
) -> plt.Figure:
    """
    Plot species and resources over time.

    :param t: Time points
    :param y: State variables (species and resources)
    :param split_index: Index separating species and resources in the state variables
    :param species_names: List of species names
    :param resource_names: List of resource names
    :return: Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 2))

    # Plot species
    for i in range(split_index):
        axes[0].plot(t, y[i], label=species_names[i])
    axes[0].set_title("Species")
    axes[0].set_xlabel("Time")
    axes[0].legend()

    # Plot resources
    for i in range(split_index, y.shape[0]):
        axes[1].plot(t, y[i], label=resource_names[i - split_index])
    axes[1].set_title("Resources")
    axes[1].set_xlabel("Time")
    axes[1].legend()

    return fig


def plot_consumer_preference_and_leakage(
    consumer_matrix: np.ndarray,
    leakage_matrix: np.ndarray,
    species_names: list[str],
    resource_names: list[str],
) -> plt.Figure:
    """
    Plot consumer preference and leakage matrices.

    :param consumer_matrix: Consumer preference matrix
    :param leakage_matrix: Leakage coefficient matrix
    :param species_names: List of species names
    :param resource_names: List of resource names
    :return: Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 2))

    # Plot consumer preference
    im1 = axes[0].imshow(consumer_matrix, cmap="Greys", aspect="equal")
    axes[0].set_title("Consumer Preference")
    axes[0].set_xlabel("Resources")
    axes[0].set_ylabel("Species")
    axes[0].set_xticks(ticks=np.arange(len(resource_names)), labels=resource_names)
    axes[0].set_yticks(ticks=np.arange(len(species_names)), labels=species_names)
    fig.colorbar(im1, ax=axes[0])

    # Plot leakage
    im2 = axes[1].imshow(leakage_matrix, cmap="Greys", aspect="equal")
    axes[1].set_title("Leakage")
    axes[1].set_xlabel("Resources")
    axes[1].set_ylabel("Species")
    axes[1].set_xticks(ticks=np.arange(len(resource_names)), labels=resource_names)
    axes[1].set_yticks(ticks=np.arange(len(species_names)), labels=species_names)
    fig.colorbar(im2, ax=axes[1])

    return fig


def plot_production_consumption_matrices(
    production_matrices: list[np.ndarray],
    resource_names: list[str],
) -> plt.Figure:
    """
    Plot production and consumption matrices for each species.

    :param production_matrices: List of production matrices for each species
    :param resource_names: List of resource names
    :return: Matplotlib figure object
    """
    num_species = len(production_matrices)
    fig, axes = plt.subplots(1, num_species, figsize=(3 * num_species, 2))

    for i in range(num_species):
        ax = axes[i] if num_species > 1 else axes
        im = ax.imshow(production_matrices[i], cmap="Greys", aspect="equal")
        ax.set_title(f"Species {i}")
        ax.set_xlabel("Produced")
        ax.set_ylabel("Consumed")
        ax.set_xticks(ticks=np.arange(len(resource_names)), labels=resource_names)
        ax.set_yticks(ticks=np.arange(len(resource_names)), labels=resource_names)
        fig.colorbar(im, ax=ax, orientation="vertical")

    return fig


def plot_network_graph(
    species_names: np.ndarray,
    resource_names: np.ndarray,
    production: np.ndarray,
    resource_intake: np.ndarray,
) -> plt.Figure:
    """
    Plot network graph with fluxes annotated on the edges.

    :param species_names: List of names of species.
    :param resource_names: List of names of resources.
    :param production: Numpy array-like object of shape (N, R, R) where
        - N is the number of species.
        - R is the number of resources.
    :return: Matplotlib plot of the network graph.
    """

    def get_network_edges():
        """
        Create a dictionary of edges with flux and species information.

        :return: Dictionary where keys are tuples of (node0, node1) and values are dictionaries with keys 'species' and 'flux'.
        """
        edges = {}
        for species, matrix in zip(species_names, production):
            for b in range(len(resource_names)):
                for a in range(len(resource_names)):
                    if matrix[b][a] > 0:
                        edge_nodes = (resource_names[b], resource_names[a])
                        if edge_nodes not in edges.keys():
                            edges[edge_nodes] = {
                                "species": [species],
                                "flux": matrix[b][a],
                            }
                        elif edge_nodes in edges.keys():
                            edges[edge_nodes]["species"].append(species)
                            edges[edge_nodes]["flux"] += matrix[b][a]
        return edges

    # Annotate edges with labels
    network_edges = get_network_edges()
    edge_labels = {
        (key[0], key[1]): f"{','.join(value['species'])}: {value['flux']}"
        for key, value in network_edges.items()
    }

    # Draw the graph and return
    g = nx.DiGraph()
    g.add_edges_from(network_edges)

    # Add edges for intake
    color_map = []
    for node in g.nodes:
        if node.startswith("Hidden_"):
            color_map.append("None")
        else:
            color_map.append("skyblue")  # Default color for visible nodes

    for index, value in enumerate(resource_intake):
        if value > 0:
            hidden_node = f"Hidden_{resource_names[index]}"
            g.add_node(hidden_node)
            g.add_edge(hidden_node, resource_names[index])
            edge_labels[(hidden_node, resource_names[index])] = f"In: {value}"
            color_map.append("None")  # Ensure hidden nodes are invisible

    pos = nx.planar_layout(g)

    # Filter out hidden nodes from labels
    visible_labels = {node: node for node in g.nodes if not node.startswith("Hidden_")}

    nx.draw(g, pos, with_labels=False, arrows=True, node_color=color_map)
    nx.draw_networkx_labels(
        g, pos, labels=visible_labels
    )  # Add labels only for visible nodes
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

    fig = plt.gcf()  # Get the current figure
    return fig


def main(experiment_name, time, update_data, network_graph):
    # Load the community
    pond = Community(experiment_name)
    pond.load_data()
    # Set results paths
    pond.results_path = os.path.join(os.path.dirname(pond.data_path), "results")
    os.makedirs(pond.results_path, exist_ok=True)
    fig_path = os.path.join(pond.results_path, "figures")
    os.makedirs(fig_path, exist_ok=True)

    # Integrate the system
    sol = pond.integrate(time)

    # Print final dynamics
    dynamics_at_end = pond.dynamics(
        0, sol.y[:, -1], pond.C, pond.D, pond.l, pond.params
    )
    resources_at_end = pond.resource_rates(sol.y[:, -1])
    resources_at_end["Concentration"] = sol.y[len(pond.species_names) :, -1]
    resources_at_end["Rates"] = dynamics_at_end[len(pond.species_names) :]
    print("Species at end of simulation")
    print(
        pd.DataFrame(
            {
                "Species": pond.species_names,
                "Concentration": sol.y[: len(pond.species_names), -1],
                "Rates": dynamics_at_end[: len(pond.species_names)],
            }
        )
    )
    print(resources_at_end)

    # Plot species and resources over full time span
    split_index = len(pond.C)
    save_figure(
        plot_species_and_resources(
            sol.t, sol.y, split_index, pond.species_names, pond.resource_names
        ),
        os.path.join(fig_path, "species_resources"),
    )

    # Plot species and resources over first 10 time units
    save_figure(
        plot_species_and_resources(
            sol.t[: len(sol.t) // 10],
            sol.y[:, : len(sol.y[0]) // 10],
            split_index,
            pond.species_names,
            pond.resource_names,
        ),
        os.path.join(fig_path, "species_resources_first_10_percent"),
    )

    if update_data:
        # Plot consumer preference and leakage
        save_figure(
            plot_consumer_preference_and_leakage(
                pond.C, pond.l, pond.species_names, pond.resource_names
            ),
            os.path.join(pond.data_path, "consumer_preference_leakage"),
        )

        # Plot production and consumption matrices
        save_figure(
            plot_production_consumption_matrices(pond.D, pond.resource_names),
            os.path.join(pond.data_path, "production_consumption_matrices"),
        )

    if network_graph:
        save_figure(
            plot_network_graph(
                species_names=pond.species_names,
                resource_names=pond.resource_names,
                production=np.round(pond.resource_production(sol.y[:, -1]), 2),
                resource_intake=pond.params["R_intake"],
            ),
            os.path.join(fig_path, "network_fluxes_at_steady_state"),
        )


if __name__ == "__main__":

    def parse_arguments():
        parser = argparse.ArgumentParser(
            description="Analyze ecological simulation results."
        )
        parser.add_argument(
            "experiment_name", type=str, help="Name of the experiment to analyze."
        )
        parser.add_argument(
            "-t",
            "--time",
            type=int,
            default=100,
            help="Simulation time (default: 100).",
        )
        parser.add_argument(
            "-d",
            "--update-data",
            action="store_true",
            default=False,
            help="Save figures in the data_path directory instead of results_path.",
        )
        parser.add_argument(
            "-g",
            "--network-graph",
            action="store_true",
            default=False,
            help="Draw network fluxes at steady state.",
        )
        return parser.parse_args()

    args = parse_arguments()
    main(args.experiment_name, args.time, args.update_data, args.network_graph)
