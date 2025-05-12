#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Date: Saturday, March 1st 2025, 7:59:59 pm
Author: NomadicPython

Copyright (c) 2025  
"""

import os
import numpy as np
import pandas as pd
from utilities import *
import json
import scipy.integrate


class Community:
    """TODO
    Each community has the following attributes to be functional

    consumer_matrix:
    TODO what happens if species gets deleted, its easier to deal with dataframes
    dictionary of metabolic matrices for each species
    dictionary of leakage coefficients for each species
    active species: moment a species goes extinct it can not revive.
    """

    def __init__(self, experiment_name: str):
        "Intialize the community with the data that is same for an experiment"
        self.exp = experiment_name

    def load_data(self, data_path: str | None = None, randomize: bool = True) -> None:
        """
        Load experimental data from the specified path.

        :param data_path: Path to the data folder (optional, defaults to the experiment's data folder if None).
        :param randomize: If true, consumer preference and metabolic matrices use random sampling from existing data.
        """
        if data_path == None:
            data_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "experiments", self.exp, "data")
            )

        # load parameters
        with open(os.path.join(data_path, "parameters.json"), "r") as f:
            param_dict = json.load(f)

        self.params = param_dict
        for key, value in param_dict.items():
            if isinstance(value, list) and key != "cv":
                self.params[key] = np.array(value)

        # load consumer preference
        if randomize:
            cv = self.params["cv"]
        else:
            cv = 0
        self.C, self.species_names, self.resource_names = create_c_matrix(
            pd.read_csv(
                os.path.join(data_path, "consumer_preference.csv"),
                index_col=0,
                header=0,
            ),
            cv=cv,
        )

        # load metabolic matrices
        self.D = extract_d_matrices(
            os.path.join(data_path, "metabolic_matrices.csv"), use_dirichlet=randomize
        )

        # load leakage coefficients
        self.l = pd.read_csv(
            os.path.join(data_path, "leakage_coefficients.csv"), index_col=0, header=0
        ).to_numpy()
        self.data_path = data_path

    def create_data(self, num_species: int, num_resources: int) -> None:
        """
        Generate experimental setup for a new experiment.
        TODO: Refactor using save_data, BUG: metabolic matrices

        :param num_species: Number of species in the system.
        :param num_resources: Number of resources in the system.
        :raise ValueError: If the directory for the experiment name already exists.
        """
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "experiments", self.exp, "data")
        )

        # raise error if data already exists
        if os.path.exists(data_path):
            raise ValueError(
                f"The data folder for experiment: {self.exp} already exists at {data_path}"
            )

        os.makedirs(data_path)

        species_list = [f"S{i}" for i in range(num_species)]
        resource_list = [f"M{i}" for i in range(num_resources)]

        # create the metabolic csv
        metabolic_matrices = pd.DataFrame(
            np.zeros([num_species * num_resources, num_resources]),
            index=np.repeat(species_list, num_resources),
            columns=resource_list,
        )
        metabolic_matrices["resource"] = resource_list * num_species
        metabolic_matrices.to_csv(os.path.join(data_path, "metabolic_matrices.csv"))

        # create the consumer_preference csv
        consumer_preference = pd.DataFrame(
            np.zeros([num_species, num_resources]),
            index=species_list,
            columns=resource_list,
        )
        consumer_preference.to_csv(os.path.join(data_path, "consumer_preference.csv"))

        # create the leakage coefficients csv
        leakage_coefficients = pd.DataFrame(
            np.zeros([num_species, num_resources]),
            index=species_list,
            columns=resource_list,
        )
        leakage_coefficients.to_csv(os.path.join(data_path, "leakage_coefficients.csv"))

        # create parameter dict to store in json
        param_dict = {
            "N0": [1] * num_species,
            "R0": [1] * num_resources,
            "growth_factor": [1] * num_species,
            "maintenance_energy": [1] * num_species,
            "w": [1] * num_resources,
            "cv": 0.1,
            "R_intake": [1] + [0] * (num_resources - 1),
        }
        with open(os.path.join(data_path, "parameters.json"), "w") as f:
            json.dump(param_dict, f, indent=4)

    def save_data(self, path):
        """
        Save the community's data to the specified path.

        :param path: Path to the directory where data will be saved.
        """
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)

        # Convert numpy arrays in params to lists for JSON serialization
        serializable_params = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in self.params.items()
        }

        # Store parameter dict in JSON
        with open(os.path.join(path, "parameters.json"), "w") as f:
            json.dump(serializable_params, f, indent=4)

        # Create the metabolic csv
        metabolic_matrices = pd.DataFrame(
            np.concatenate(self.D),
            index=np.repeat(self.species_names, len(self.resource_names)),
            columns=self.resource_names,
        )
        metabolic_matrices["resource"] = self.resource_names * len(self.species_names)
        metabolic_matrices.to_csv(os.path.join(path, "metabolic_matrices.csv"))

        # Create the consumer_preference csv
        consumer_preference = pd.DataFrame(
            self.C,
            index=self.species_names,
            columns=self.resource_names,
        )
        consumer_preference.to_csv(os.path.join(path, "consumer_preference.csv"))

        # Create the leakage coefficients csv
        leakage_coefficients = pd.DataFrame(
            self.l,
            index=self.species_names,
            columns=self.resource_names,
        )
        leakage_coefficients.to_csv(os.path.join(path, "leakage_coefficients.csv"))

    def __str__(self) -> str:
        """Prints the community object with the data loaded"""
        return (
            f"Community: {self.exp}\n"
            f"Species: {self.C.shape[0]}\n"
            f"Resources: {self.C.shape[1]}\n"
        )

    def create_dynamics(self) -> None:
        """
        Create the dynamics of the system, including consumer and resource dynamics.

        This method defines and assigns the following dynamic functions as attributes of the class:
        - `dynamics`: Calculates the rate of change of species and resources.
        - `resource_rates`: Calculates the rate of change of resources.
        - `resource_consumption`: Calculates resource consumption rates for each species.
        - `resource_production`: Calculates resource production rates for each species and resource.
        """

        def consumer_dynamics(
            N: np.ndarray,
            R: np.ndarray,
            C: np.ndarray,
            g: np.ndarray,
            l: np.ndarray,
            w: np.ndarray,
            m: np.ndarray,
        ) -> np.ndarray:
            """
            Calculate the growth rate of the species.

            :param N: Abundance of species.
            :param R: Abundance of resources.
            :param C: Consumer preference matrix.
            :param g: Growth factor for each species.
            :param l: Leakage coefficient matrix (species x resources).
            :param w: Energy factor for each resource.
            :param m: Maintenance energy for each species.
            :return: Array of growth rates of each species.
            """
            return g * N * (np.einsum("j,ij,ij,j->i", w, 1 - l, C, R) - m)

        def resource_dynamics(
            N: np.ndarray,
            R: np.ndarray,
            C: np.ndarray,
            D: np.ndarray,
            h: np.ndarray,
            l: np.ndarray,
            w: np.ndarray,
        ) -> np.ndarray:
            """
            Calculate the rate of change of resources.

            :param N: Abundance of species.
            :param R: Abundance of resources.
            :param C: Consumer preference matrix.
            :param D: Array of metabolic matrices for each species.
            :param h: Uptake rate for resources.
            :param l: Leakage coefficient matrix (species x resources).
            :param w: Energy factor for each resource.
            :return: Array of production rates of each resource.
            """
            return (
                h  # intake
                - (R * np.einsum("ja,j->a", C, N))  # consumption
                + (np.einsum("b,b,j,jb,jb,jbi->i", R, w, N, l, C, D) / w)  # production
            )

        def dynamics(
            t: float,
            y: np.ndarray,
            C: np.ndarray,
            D: np.ndarray,
            l: np.ndarray,
            params: dict,
        ) -> np.ndarray:
            """
            Calculate the rate of change of species and resources.

            :param t: Time.
            :param y: State vector.
            :param C: Consumer preference matrix.
            :param D: Array of metabolic matrices for each species.
            :param l: Leakage coefficient matrix (species x resources).
            :param params: Parameters.
            :return: Rate of change of species and resources.
            """
            # unpack state vector
            N = y[: len(C)]
            R = y[len(C) :]

            return np.concatenate(
                (
                    consumer_dynamics(
                        N,
                        R,
                        C,
                        params["growth_factor"],
                        l,
                        params["w"],
                        params["maintenance_energy"],
                    ),
                    resource_dynamics(N, R, C, D, params["R_intake"], l, params["w"]),
                )
            )

        def resource_rates(y: np.ndarray) -> pd.DataFrame:
            """
            Calculate detailed resource dynamics including intake, consumption, and production.

            :param y: State vector containing species and resource abundances
            :param C: Consumer preference matrix
            :param D: Array of metabolic matrices for each species
            :param l: Leakage coefficient matrix (species x resources)
            :param params: Dictionary of parameters including R_intake and w
            :return: DataFrame with detailed resource dynamics (intake, consumption, production)
            """
            return pd.DataFrame(
                {
                    "Resource": self.resource_names,
                    "Intake": self.params["R_intake"],
                    "Consumption": y[len(self.C) :]
                    * np.einsum("ji,j", self.C, y[: len(self.C)]),
                    "Production": np.einsum(
                        "b,b,j,jb,jb,jba,a->a",
                        y[len(self.C) :],
                        self.params["w"],
                        y[: len(self.C)],
                        self.l,
                        self.C,
                        self.D,
                        1 / self.params["w"],
                    ),
                }
            ).set_index("Resource")

        def resource_consumption(y: np.ndarray) -> np.ndarray:
            """
            Calculate the resource consumption rates for each species.

            :param y: Abundance of species and resources in a concatenated array.
            :return: A 2D array where each element [j, a] represents the consumption rate
                     of resource a by species j.
            """
            return np.einsum(
                "a,j,ja->ja",
                y[len(self.species_names) :],
                y[: len(self.species_names)],
                self.C,
            )

        def resource_production(y: np.ndarray) -> np.ndarray:
            """
            Calculate the resource production rates for each species and resource.

            :param y: Abundance of species and resources in a concatenated array.
            :return: A 3D array where each element [j, b, a] represents the production rate
                     of resource 'a' by species 'j' using resource 'b'.
            """
            return np.einsum(
                "b,b,j,jb,jb,jba,a->jba",
                y[len(self.C) :],
                self.params["w"],
                y[: len(self.C)],
                self.l,
                self.C,
                self.D,
                1 / self.params["w"],
            )

        self.dynamics = dynamics
        self.resource_rates = resource_rates
        self.resource_consumption = resource_consumption
        self.resource_production = resource_production

    def integrate(
        self, time: int | float, y0: np.ndarray | None = None, **kwargs
    ) -> scipy.integrate._ivp.ivp.OdeResult:
        """
        Numerically integrate the community over the provided timespan using consumer-resource dynamics.

        :param time: Time duration for integration.
        :param y0: Initial state vector of species and resources (optional).
        :param kwargs: Additional arguments passed to scipy.integrate.solve_ivp.
        :return: Integration result as a scipy.integrate._ivp.ivp.OdeResult object.
        """
        # set initial concentrations of species and resources if not provided
        if y0 is None:
            y0 = np.concatenate((self.params["N0"], self.params["R0"]))
        t_span = (0, time)  # time span for solution
        # create dynamics
        if not hasattr(self, "dynamics"):
            self.create_dynamics()
        return scipy.integrate.solve_ivp(
            self.dynamics,
            t_span,
            y0,
            args=(self.C, self.D, self.l, self.params),
            **kwargs,
        )

    def optimized_integrate(
        self,
        time: int | float,
        y0: np.ndarray | None = None,
        time_step: int = 1000,
        threshold: float = 10 ** (-4),
        **kwargs,
    ) -> scipy.integrate._ivp.ivp.OdeResult:
        """
        Numerically integrate the community over the provided timespan using consumer-resource dynamics.
        Sets species with really low presence to 0.

        :param time: Time duration for integration.
        :param y0: Initial state vector of species and resources (optional).
        :param kwargs: Additional arguments passed to scipy.integrate.solve_ivp.
        :return: Integration result as a scipy.integrate._ivp.ivp.OdeResult object.
        """
        # set initial concentrations of species and resources if not provided
        if y0 is None:
            y0 = np.concatenate((self.params["N0"], self.params["R0"]))
        num_steps, leftover_time = time // time_step, time % time_step
        # simulate in batches of time_step duration
        for i in range(num_steps):
            prev_y0 = y0
            sol = self.integrate(time_step, y0, **kwargs)
            y0 = sol.y[:, -1]
            # set species with low presence to 0
            y0[: len(self.species_names)] = y0[: len(self.species_names)] * (
                y0[: len(self.species_names)] > threshold
            )
            # if the solution has not changed, break the loop
            if (np.round(prev_y0, 4) == np.round(y0, 4)).all():
                break
        # simulate for any remaining time
        sol = self.integrate(leftover_time, y0, **kwargs)
        return sol
