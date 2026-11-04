import numpy as np
import xarray as xr

# Load the weather dataset used by the test simulation
ds = xr.open_dataset("tests/data/tests_weather_data.nc")

print("--- Weather Dataset Dimensions & Variables ---")
print(ds)

# Calculate statistics for each available data variable
print("\n--- Variable Statistics ---")
for var in ds.data_vars:
    values = ds[var].values.flatten()
    valid = values[~np.isnan(values)]

    if len(valid) == 0:
        continue

    print(f"\nVariable: {var}")
    print(f"  Mean:               {np.mean(valid):.4f}")
    print(f"  Median:             {np.median(valid):.4f}")
    print(f"  Standard Deviation: {np.std(valid):.4f}")
    print(f"  Min / Max:          {np.min(valid):.4f} / {np.max(valid):.4f}")