from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Load weather dataset
ds = xr.open_dataset("tests/data/tests_weather_data.nc")
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# Extract Wave Height (VHM0) and Mean Wave Direction (VMDR)
vhm0 = ds["VHM0"].values.flatten()
vhm0 = vhm0[~np.isnan(vhm0)]

vmdr = ds["VMDR"].values.flatten()
vmdr = vmdr[~np.isnan(vmdr)]

time_series_vhm0 = ds["VHM0"].mean(dim=["latitude", "longitude"])

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. Time series of spatial mean wave height
axs[0, 0].plot(ds["time"].values, time_series_vhm0.values, color="navy", marker="o", lw=1.5)
axs[0, 0].set_title("Mean Significant Wave Height Over Time (Mediterranean)", fontsize=11, fontweight="bold")
axs[0, 0].set_ylabel("Wave Height (m)")
axs[0, 0].grid(True, linestyle="--", alpha=0.6)
axs[0, 0].tick_params(axis='x', rotation=30)

# 2. Histogram / Distribution of Wave Height (VHM0)
axs[0, 1].hist(vhm0, bins=35, color="steelblue", edgecolor="black", alpha=0.8)
axs[0, 1].axvline(np.mean(vhm0), color="red", linestyle="--", label=f"Mean: {np.mean(vhm0):.2f}m")
axs[0, 1].axvline(np.median(vhm0), color="green", linestyle=":", label=f"Median: {np.median(vhm0):.2f}m")
axs[0, 1].set_title("Wave Height Distribution (VHM0)", fontsize=11, fontweight="bold")
axs[0, 1].set_xlabel("Wave Height (m)")
axs[0, 1].set_ylabel("Count")
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle="--", alpha=0.6)

# 3. Wave Direction Distribution (VMDR)
axs[1, 0].hist(vmdr, bins=36, color="coral", edgecolor="black", alpha=0.8)
axs[1, 0].axvline(np.mean(vmdr), color="red", linestyle="--", label=f"Mean: {np.mean(vmdr):.1f}°")
axs[1, 0].axvline(np.median(vmdr), color="green", linestyle=":", label=f"Median: {np.median(vmdr):.1f}°")
axs[1, 0].set_title("Mean Wave Direction Distribution (VMDR)", fontsize=11, fontweight="bold")
axs[1, 0].set_xlabel("Degrees (°)")
axs[1, 0].set_ylabel("Count")
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle="--", alpha=0.6)

# 4. Summary Statistics Box
axs[1, 1].axis("off")
stats_text = (
    "Weather Dataset Summary (CMEMS Mediterranean):\n\n"
    f"Time Steps: {len(ds['time'])}\n"
    f"Spatial Grid: {len(ds['latitude'])} lats x {len(ds['longitude'])} lons\n\n"
    f"Significant Wave Height (VHM0):\n"
    f"  • Mean:   {np.mean(vhm0):.4f} m\n"
    f"  • Median: {np.median(vhm0):.4f} m\n"
    f"  • StdDev: {np.std(vhm0):.4f} m\n\n"
    f"Wave Direction (VMDR):\n"
    f"  • Mean:   {np.mean(vmdr):.4f}°\n"
    f"  • Median: {np.median(vmdr):.4f}°\n"
    f"  • StdDev: {np.std(vmdr):.4f}°"
)
axs[1, 1].text(0.1, 0.25, stats_text, fontsize=12, family="monospace",
               bbox=dict(boxstyle="round,pad=1", facecolor="whitesmoke", edgecolor="gray"))

plt.tight_layout()
plot_file = output_dir / "weather_analysis.png"
plt.savefig(plot_file, dpi=200)
print(f"Weather visualization saved successfully at: {plot_file.resolve()}")