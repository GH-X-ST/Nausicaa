import aerosandbox as asb
import aerosandbox.numpy as np
from pathlib import Path

def main():
    # Output directory
    cache_dir = Path("cache_test")
    cache_dir.mkdir(exist_ok=True)

    # Create NACA 0002 airfoil
    af = asb.Airfoil(name="naca0002")

    # Angle of attack range (degrees)
    alphas = np.linspace(-10, 10, 41)

    # Generate and save polars
    af.generate_polars(
        alphas=alphas,
        cache_filename=str(cache_dir / "naca0002.json"),
    )

    print("Done.")
    print("File exists:", (cache_dir / "naca0002.json").exists())
    print("Location:", (cache_dir / "naca0002.json").resolve())

if __name__ == "__main__":
    main()
