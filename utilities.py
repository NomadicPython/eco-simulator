#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Date: Saturday, March 1st 2025, 7:59:59 pm
Author: NomadicPython

Copyright (c) 2025  
"""
import numpy as np
import pandas as pd


def load_data(csv_file: str) -> pd.DataFrame:
    """Loads values from a csv file to a pandas dataframe,
    expects column names in header and first row is used as index

    :param csv_file: path to csv_file with data
    :return: pandas dataframe with data
    """
    return pd.read_csv(csv_file, header=0, index_col=0)


def replace_zero(A: np.ndarray, replacement_type: str = "perturbation") -> np.ndarray:
    """Replaces 0 values with a small baseline positive value = p

        where p = minimum nonzero value in A divided by sum of all values in A
    :param A: a numpy array like object
    :param replacement_type: 'perturbation' (default) | 'minimum'
    :return: a numpy array like object without 0 values
    """
    if replacement_type == "perturbation":
        p = A.min(where=(A != 0), initial=A.max()) / A.sum()
    elif replacement_type == "minimum":
        p = A.min(where=(A != 0), initial=A.max())
    else:
        raise ValueError("replacement_type should be perturbation or minimum")
    return (A == 0).astype(int) * p + A


def create_c_matrix(
    consumer_pref: pd.DataFrame, cv: float, replacement_type: str = "perturbation"
) -> np.ndarray:
    """Generate a stochastic consumer preference matrix from template

    :param consumer_df: pandas dataframe with integer values for consumer preferences
                        serve as the mean for the randomly sampled value.
    :param cv: A single coefficient of variation used for random sampling
    :return c_matrix: numpy matrix
    """
    species_names = consumer_pref.index.to_list()
    resource_names = consumer_pref.columns.to_list()

    consumer_pref = consumer_pref.to_numpy()
    consumer_pref = np.abs(
        np.random.default_rng().normal(
            consumer_pref, cv * consumer_pref  # mean values  # std values
        )
    )
    return consumer_pref, species_names, resource_names


def create_d_matrix(metabolic_df: pd.DataFrame) -> np.ndarray:
    """Generates dirichlet distribution for conversion of resources to another resource

    :param metabolic_df: concentration values for dirichlet distribution for metabolic matrix
    :return: sampled values for dirichlet distribution, rowsums = 1
    """
    return np.apply_along_axis(
        np.random.default_rng().dirichlet, 1, metabolic_df.to_numpy()
    )


def extract_d_matrices(combined_metabolic_csv: str) -> np.ndarray:
    """Extracts species specific DF from a single csv file

    :param combined_metabolic_csv: path to the csv file
    :return D_dict: dictionary with species as key and metabolic_df as value
    """
    data = pd.read_csv(combined_metabolic_csv, header=0, index_col=0)
    species_list = data.index.unique()
    D = [data.loc[species].set_index("resource") for species in species_list]
    return np.array(D)
