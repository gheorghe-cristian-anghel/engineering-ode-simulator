"""Reusable CSV export helpers for simulation results."""

import csv
from pathlib import Path

import numpy as np


def ensure_output_directory(directory="outputs"):
    """Create and return an output directory Path."""
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory


def export_simulation_to_csv(file_path, columns):
    """Export simulation data to a CSV file.

    Parameters
    ----------
    file_path:
        Path to the CSV file to create.
    columns:
        Dictionary mapping column names to one-dimensional arrays or lists.

    Returns
    -------
    pathlib.Path
        Path to the created CSV file.
    """
    if not columns:
        raise ValueError("columns must not be empty")

    column_arrays = {}
    expected_length = None

    for column_name, values in columns.items():
        if not isinstance(column_name, str):
            raise ValueError("column names must be strings")

        if column_name == "":
            raise ValueError("column names must not be empty")

        array = np.asarray(values)

        if array.ndim != 1:
            raise ValueError("all columns must be one-dimensional")

        if expected_length is None:
            expected_length = len(array)
        elif len(array) != expected_length:
            raise ValueError("all columns must have the same length")

        column_arrays[column_name] = array

    csv_path = Path(file_path)
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("file_path must have a .csv extension")

    if csv_path.parent != Path("."):
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        column_names = list(column_arrays.keys())
        writer.writerow(column_names)

        for row_index in range(expected_length):
            writer.writerow([column_arrays[name][row_index] for name in column_names])

    return csv_path
