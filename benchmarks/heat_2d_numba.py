"""Compare the optional Numba heat kernel with the preserved Python path."""

from time import perf_counter

from models.heat_equation_2d import simulate_heat_equation_2d


def measure(acceleration):
    start = perf_counter()
    simulate_heat_equation_2d(
        nx=401, ny=401, alpha=0.01, t_final=0.0125, dt=0.00000625,
        store_every=2_000, acceleration=acceleration,
    )
    return perf_counter() - start


if __name__ == "__main__":
    measure("numba")  # compile before the measured native run
    python_seconds = measure("python")
    numba_seconds = measure("numba")
    print(f"python={python_seconds:.3f}s numba={numba_seconds:.3f}s speedup={python_seconds / numba_seconds:.2f}x")
