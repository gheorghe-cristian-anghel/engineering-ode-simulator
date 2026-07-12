# Optional Numba 2D heat acceleration

The 2D heat solver can use an isolated Numba kernel for its explicit time-step
loop. The default `acceleration="auto"` uses it when installed and otherwise
keeps the original NumPy implementation. Use `acceleration="python"` to force
the fallback or `acceleration="numba"` to request it with graceful fallback.

Install the optional Windows-compatible wheel:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[acceleration]"
```

Run the comparison benchmark after the first compilation:

```powershell
.\.venv\Scripts\python.exe benchmarks\heat_2d_numba.py
```

The kernel preserves the explicit `float64` scheme and both supported boundary
semantics. Numerical equivalence is tested against `acceleration="python"`.
