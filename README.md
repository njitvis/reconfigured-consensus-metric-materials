# Simulation Study

This folder contains two working files:

- `residual_consensus.py`: a standalone Python implementation of the residual-consensus metric.
- `simulation_study_v2.ipynb`: a notebook for exploratory simulation and visualization built on top of the Python module.

## Files

### `residual_consensus.py`

This script provides the reusable metric logic. It includes:

- `calculate_cofA(...)`: the consensus calculation.
- `residual_from_counts(...)`: computes the residual-consensus metrics for one count vector.
- `compute_units(...)`: computes residual-consensus metrics for grouped survey-response data in a pandas DataFrame.
- a built-in self-test and demo when run directly.

Use this file if you want a scriptable, importable implementation.

### `simulation_study_v2.ipynb`

This notebook imports `residual_consensus.py` and adds:

- simulation helpers
- baseline-curve exploration
- residual distribution analysis
- plotting with matplotlib and seaborn

Use this file if you want to inspect or extend the simulation workflow interactively.

## Requirements

Install the Python packages listed in `requirement.txt`:

```bash
pip install -r requirement.txt
```

The requirement file includes notebook dependencies as well as the plotting and scientific packages used by `simulation_study_v2.ipynb`.

## How To Use

### Option 1: Run the Python module

From this folder:

```bash
python residual_consensus.py
```

That runs:

- the built-in self-test
- a small demo using an example count vector

You can also import the module from another script:

```python
import pandas as pd
from residual_consensus import compute_units, residual_from_counts

record = residual_from_counts([4, 5, 6, 26, 42])
print(record)

df = pd.read_csv("student_responses.csv")
units = compute_units(
    df,
    response_col="Overallteachingeffectiv",
    group_cols=["cname"],
    scale_min=0,
    scale_max=4,
)
print(units.head())
```

Expected input for `compute_units(...)`:

- one row per response
- a numeric response column for a single question
- one or more grouping columns such as section or department

### Option 2: Open the notebook

Launch Jupyter from this folder:

```bash
jupyter lab
```

Then open `simulation_study_v2.ipynb`.

The notebook expects `residual_consensus.py` to remain in the same folder, because it imports it directly as:

```python
import residual_consensus as rc
```

## Notes

- The notebook uses the script in the current folder rather than a packaged install.
- If you only need the Python module, the critical runtime dependencies are `numpy` and `pandas`.
- If you want to rerun the notebook end to end, keep the scientific and notebook packages from `requirement.txt` installed.
