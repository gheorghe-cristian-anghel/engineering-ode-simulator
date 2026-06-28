import csv

import numpy as np
import pytest

from analysis.export_utils import export_simulation_to_csv


def _read_csv_rows(file_path):
    """Read CSV rows from a file path."""
    with file_path.open(newline="") as csv_file:
        return list(csv.reader(csv_file))


def test_export_creates_csv_file(tmp_path):
    """CSV export should create the requested file."""
    file_path = tmp_path / "simulation.csv"

    export_simulation_to_csv(file_path, {"time_s": [0, 1], "speed_rad_s": [0, 10]})

    assert file_path.exists()


def test_csv_file_contains_correct_header(tmp_path):
    """CSV export should write column names as the header row."""
    file_path = tmp_path / "simulation.csv"

    export_simulation_to_csv(file_path, {"time_s": [0, 1], "speed_rad_s": [0, 10]})
    rows = _read_csv_rows(file_path)

    assert rows[0] == ["time_s", "speed_rad_s"]


def test_csv_file_contains_correct_number_of_data_rows(tmp_path):
    """CSV export should write one data row per sample."""
    file_path = tmp_path / "simulation.csv"

    export_simulation_to_csv(file_path, {"time_s": [0, 1, 2], "speed_rad_s": [0, 10, 20]})
    rows = _read_csv_rows(file_path)

    assert len(rows[1:]) == 3


def test_exported_values_match_input_arrays_and_lists(tmp_path):
    """CSV export should preserve input values."""
    file_path = tmp_path / "simulation.csv"

    export_simulation_to_csv(
        file_path,
        {
            "time_s": np.array([0.0, 0.5]),
            "speed_rad_s": [0.0, 12.5],
        },
    )
    rows = _read_csv_rows(file_path)

    assert rows[1] == ["0.0", "0.0"]
    assert rows[2] == ["0.5", "12.5"]


def test_parent_directory_is_created_automatically(tmp_path):
    """CSV export should create missing parent directories."""
    file_path = tmp_path / "nested" / "simulation.csv"

    returned_path = export_simulation_to_csv(file_path, {"time_s": [0], "speed_rad_s": [0]})

    assert returned_path == file_path
    assert file_path.exists()


def test_empty_columns_raise_value_error(tmp_path):
    """CSV export needs at least one column."""
    with pytest.raises(ValueError):
        export_simulation_to_csv(tmp_path / "simulation.csv", {})


def test_mismatched_column_lengths_raise_value_error(tmp_path):
    """All columns must have the same number of samples."""
    with pytest.raises(ValueError):
        export_simulation_to_csv(
            tmp_path / "simulation.csv",
            {"time_s": [0, 1], "speed_rad_s": [0]},
        )


def test_non_string_column_name_raises_value_error(tmp_path):
    """Column names must be strings."""
    with pytest.raises(ValueError):
        export_simulation_to_csv(tmp_path / "simulation.csv", {1: [0, 1]})


def test_empty_string_column_name_raises_value_error(tmp_path):
    """Column names must not be empty."""
    with pytest.raises(ValueError):
        export_simulation_to_csv(tmp_path / "simulation.csv", {"": [0, 1]})


def test_multidimensional_array_raises_value_error(tmp_path):
    """Columns must be one-dimensional."""
    with pytest.raises(ValueError):
        export_simulation_to_csv(
            tmp_path / "simulation.csv",
            {"matrix": np.array([[1, 2], [3, 4]])},
        )
