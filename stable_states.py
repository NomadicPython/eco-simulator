import os
import sys
import numpy as np
import pandas as pd
from community import Community
from utilities import next_experiment_path
from multiprocessing import Pool, cpu_count
from itertools import product  # Import product for generating grid combinations


def simulate(
    community: Community,
    n0_value: np.ndarray,
    R0: np.ndarray,
    simulation_index: int,
    time: int,
) -> dict:
    """
    Simulate the community dynamics for a single set of initial conditions.

    :param community: Community object representing the ecosystem.
    :param n0_value: Array of initial abundances for species.
    :param R0: Array of initial abundances for resources.
    :param simulation_index: Index of the current simulation.
    :param time: Duration of the simulation in time units.
    :return: Dictionary containing:
        - Initial values of species and resources.
        - Final values of species and resources.
        - Absolute sum of final rates of change.
    """
    y0 = np.concatenate((n0_value, R0))
    result = community.optimized_integrate(time, y0)
    final_values = result.y[:, -1]  # Get the last column of y
    final_rates = community.dynamics(
        0, result.y[:, -1], community.C, community.D, community.l, community.params
    )

    row: dict = {
        "Simulation": simulation_index + 1,
        **{
            f"Initial_{name}": val
            for name, val in zip(community.species_names + community.resource_names, y0)
        },
        **{
            f"{name}": val
            for name, val in zip(
                community.species_names + community.resource_names, final_values
            )
        },
        "abs_sum_final_rates": np.sum(np.abs(final_rates)),
    }
    print(f"Simulation {simulation_index + 1} completed.")
    return row


def run_simulations_diff_y0(experiment_name: str, time: int, cv: float = None):
    """
    Run multiple numerical simulation experiments on a community with different initial species and resource values.

    :param experiment_name: Name of the experiment, used to initialize the community.
    :param time: Duration of each simulation in time units.
    :param cv: Coefficient of variation for the initial species values.
    :return: None
    """
    # Initialize the community
    community = Community(experiment_name)
    community.load_data()

    # Set cv if provided
    if cv is not None:
        community.params["cv"] = cv

    # Create a directory to store simulation results
    batch_results_dir = os.path.join(
        os.path.dirname(community.data_path), "batch_results"
    )
    os.makedirs(batch_results_dir, exist_ok=True)

    results_dir = next_experiment_path(batch_results_dir)
    os.makedirs(results_dir)

    # Define grid values for N0 and R0
    N0_values = [0.1, 1.0, 10.0]
    R0 = np.ones(len(community.resource_names))
    grid = list(
        product(*[N0_values for _ in range(len(community.species_names))])
    )  # Generate all combinations of N0

    # Run simulations in parallel
    num_workers = min(
        16, cpu_count()
    )  # Use up to 16 workers or the number of available CPUs
    with Pool(num_workers) as pool:
        results = pool.starmap(
            simulate,
            [(community, n0_value, R0, i, time) for i, n0_value in enumerate(grid)],
        )

    # Save all results to a single CSV file
    csv_file_path = os.path.join(results_dir, "simulation_results.csv")
    print(f"Saving results to: {csv_file_path}")  # Debug: Print the file path
    pd.DataFrame(results).to_csv(csv_file_path, index=False)
    print("All simulations completed and results saved.")

    # Save community data and simulation variables
    community.save_data(results_dir)

    # Save the simulation parameters
    with open(os.path.join(results_dir, "simulation_parameters.txt"), "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Time: {time}\n")
        f.write(f"Species Names: {', '.join(community.species_names)}\n")
        f.write(f"Initial Species Values: {', '.join(map(str, N0_values))}\n")
        f.write(f"Resource Names: {', '.join(community.resource_names)}\n")
        f.write(f"Initial Resource Values: {', '.join(map(str, R0))}\n")


def cv_sim_counts(
    experiment_name: str, cv_values: list, batch_simulations: dict
) -> dict:
    """
    Count the number of simulations for each cv value in the batch simulations.

    :param experiment_name: Name of the experiment.
    :param cv_values: List of cv values to check.
    :param batch_simulations: Dictionary containing batch simulation run paths.
    :return: Dictionary with cv values as keys and counts as values.
    """
    counts = {cv: 0 for cv in cv_values}
    for run_path in batch_simulations[experiment_name]:
        exp = Community("temp")
        exp.load_data(run_path, False)
        if exp.params["cv"] in counts:
            counts[exp.params["cv"]] += 1
    return counts


def cv_simulations(batch_simulations: dict, cv_sim_minimum: int = 10):
    """
    Run simulations with different coefficients of variation (cv) for the initial species values.

    :param batch_simulations: Dictionary containing batch simulation parameters.
    :param cv_sim_minimum: Minimum number of simulations to run for each cv value.
    :return: None
    """
    # cv values
    cv_values = [0.01, 0.1]

    for experiment_name in batch_simulations:
        cv_counts = cv_sim_counts(experiment_name, cv_values, batch_simulations)
        for cv in cv_values:
            if cv_counts[cv] < cv_sim_minimum:
                print(
                    f"Running {cv_sim_minimum - cv_counts[cv]} simulations for {experiment_name} with cv={cv}"
                )
                for i in range(cv_counts[cv], cv_sim_minimum):
                    run_simulations_diff_y0(experiment_name, 10000, cv=cv)


if __name__ == "__main__":
    experiment_name = sys.argv[1]
    time = 10000
    if len(sys.argv) == 3:
        time = int(sys.argv[2])

    run_simulations_diff_y0(experiment_name, time)
