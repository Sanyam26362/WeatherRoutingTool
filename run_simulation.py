import cProfile
import logging
import pstats
from pathlib import Path

from WeatherRoutingTool.config import Config, set_up_logging
from WeatherRoutingTool.execute_routing import execute_routing
from WeatherRoutingTool.ship.ship_config import ShipConfig

CONFIG_PATH = Path("tests/config.regression.json")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the regression configuration
config = Config.assign_config(str(CONFIG_PATH))
config.ROUTE_PATH = str(OUTPUT_DIR)
ship_config = ShipConfig.assign_config(str(CONFIG_PATH))

# 2. Configure logging to file and console
log_path = OUTPUT_DIR / "simulation.log"
set_up_logging(info_log_file=str(log_path))
logging.getLogger().setLevel(logging.INFO)

print("Starting route optimization simulation...")

# 3. Profile and execute the genetic algorithm
profiler = cProfile.Profile()
profiler.enable()

execute_routing(config, ship_config)

profiler.disable()

# 4. Dump performance statistics
prof_file = OUTPUT_DIR / "routing_performance.prof"
profiler.dump_stats(str(prof_file))

print(f"\nSimulation complete. Output saved in: {OUTPUT_DIR.resolve()}")
print(f"Performance profile written to: {prof_file}\n")

# 5. Print the top 15 most time-consuming functions
stats = pstats.Stats(profiler)
stats.strip_dirs().sort_stats("cumulative").print_stats(15)