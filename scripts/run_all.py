import argparse
import subprocess
import sys
import time

STEPS = [
    ("Schema audit", ["-m", "house_prices.load"]),
    ("Clean", ["-m", "house_prices.clean"]),
    ("Split", ["-m", "house_prices.split"]),
    ("Train", ["-m", "house_prices.train"]),
    ("Evaluate", ["-m", "house_prices.evaluate"]),
    ("Intervals", ["-m", "house_prices.intervals"]),
    ("Explain", ["-m", "house_prices.explain"]),
    ("README tables", ["-m", "scripts.update_readme"]),
]

NOTEBOOKS = [
    "notebooks/01_data_audit.ipynb",
    "notebooks/02_eda.ipynb",
    "notebooks/03_modelling.ipynb",
    "notebooks/04_error_analysis.ipynb",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the whole pipeline in order.")
    parser.add_argument("--notebooks", action="store_true", help="also execute the four notebooks in place")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    steps = list(STEPS)
    if args.notebooks:
        steps.append(("Build notebooks", ["-m", "scripts.build_notebooks"]))
        steps.append(
            (
                "Execute notebooks",
                ["-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", *NOTEBOOKS],
            )
        )
    for title, command in steps:
        print(f"\n=== {title} ===", flush=True)
        started = time.time()
        result = subprocess.run([sys.executable, *command])
        print(f"--- {title}: exit {result.returncode} in {time.time() - started:.0f}s", flush=True)
        if result.returncode != 0:
            print("Stopping: fix the error above and rerun.", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
