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

    def load_data(self, data_path: str | None = None):
        """
        Assumes existing data in the folder
        N0 = param_dict["N0"]
        R0 = param_dict["R0"]
        growth_factor = param_dict["growth_factor"]
        maintenance_energy = param_dict["maintenance_energy"]
        w = np.array(param_dict["w"])
        cv = param_dict["cv"]
        R_intake = param_dict["R_intake"]

        :param data_path: (Optional)
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
        self.C, self.species_names, self.resource_names = create_c_matrix(
            pd.read_csv(
                os.path.join(data_path, "consumer_preference.csv"),
                index_col=0,
                header=0,
            ),
            cv=self.params["cv"],
        )

        # load metabolic matrices
        self.D = extract_d_matrices(os.path.join(data_path, "metabolic_matrices.csv"))

        # load leakage coefficients
        self.l = pd.read_csv(
            os.path.join(data_path, "leakage_coefficients.csv"), index_col=0, header=0
        ).to_numpy()
        self.data_path = data_path

    def create_data(self, num_species: int, num_resources: int):
        """
        Generate experimental setup for a new one

        :param num_species: Number of species in the system
        :param num_resources: Number of resources in the system
        :raise ValueError: If directory for experiment name already exists
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

    def __str__(self) -> str:
        """Prints the community object with the data loaded"""
        return (
            f"Community: {self.exp}\n"
            f"Species: {self.C.shape[0]}\n"
            f"Resources: {self.C.shape[1]}\n"
        )

    def create_dynamics(self):
        """
        Create the dynamics of the system, vectorized
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
            Calculate the growth rate of the species
            :param N: abundance of species
            :param R: abundance of resources
            :param C: consumer preference matrix
            :param g: growth factor for each species
            :param l: leakage coefficient matrix (species x resources)
            :param w: energy factor for each resource
            :param m: maintenance energy for each species
            :return: array of growth rates of each species
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
            Calculate the rate of change of resources
            :param N: abundance of species
            :param R: abundance of resources
            :param C: consumer preference matrix
            :param D: array of metabolic matrices for each species
            :param h: uptake rate for resources
            :param l: leakage coefficient matrix (species x resources)
            :param w: energy factor for each resource
            :return: array of production rates of each resource
            """
            return (
                h  # intake
                - (R * np.einsum("ja,j->a", C, N))  # consumption
                + (np.einsum("b,b,j,jb,jb,jbi->i", R, w, N, l, C, D) / w)  # production
            )

        def dynamics(t, y, C, D, l, params):
            """
            :param t: time
            :param y: state vector
            :param C: consumer preference matrix
            :param D: array of metabolic matrices for each species
            :param l: leakage coefficient matrix (species x resources)
            :param params: parameters
            :return: rate of change of species and resources
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

        self.dynamics = dynamics

        def detailed_resource_dynamics(y, C, D, l, params) -> dict:
            """
            Calculate the rate of change of resources

            :param N: abundance of species
            :param R: abundance of resources
            :param C: consumer preference matrix
            :param D: array of metabolic matrices for each species
            :param h: uptake rate for resources
            :param l: leakage coefficient matrix (species x resources)
            :param w: energy factor for each resource
            :return: array of production rates of each resource
            """
            return {
                "intake": params["R_intake"],
                "consumption": y[len(C) :] * np.einsum("ji,j", C, y[: len(C)]),
                "production": np.einsum(
                    "b,b,j,jb,jb,jbi->i", y[len(C) :], params["w"], y[: len(C)], l, C, D
                )
                / params["w"],
            }

        self.detailed_resource_dynamics = detailed_resource_dynamics

    def integrate(self, time: int | float, y0: np.ndarray | None = None, **kwargs) -> scipy.integrate._ivp.ivp.OdeResult:
        """
        Numerically integrates community over provided timespan using consumer-resource dynamics. \
        Uses scipy.integrate.solve_ivp function and takes all keyword arguments that solve_ivp takes.
        
        :param time: Time duration for integration
        :param y0: Initial state vector of species and resources (optional)
        :param kwargs: Additional arguments passed to scipy.integrate.solve_ivp
        :return: Integration result as a scipy.integrate.OdeResult object
        """
        # set initial concentrations of species and resources if not provided
        if y0 is None:
            y0 = np.concatenate((self.params["N0"], self.params["R0"]))
        t_span = (0, time)  # time span for solution
        # create dynamics
        if not hasattr(self, "dynamics"):
            self.create_dynamics()
        return scipy.integrate.solve_ivp(self.dynamics, t_span, y0, args=(self.C, self.D, self.l, self.params), **kwargs)
