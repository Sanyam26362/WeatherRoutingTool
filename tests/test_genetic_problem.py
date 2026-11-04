import numpy as np
import pytest

from WeatherRoutingTool.algorithms.genetic.problem import RoutingProblem
from WeatherRoutingTool.config import Config
from WeatherRoutingTool.constraints.constraints import ConstraintsListFactory
from WeatherRoutingTool.ship.ship_config import ShipConfig
from WeatherRoutingTool.ship.ship_factory import ShipFactory
from WeatherRoutingTool.utils.maps import Map
from WeatherRoutingTool.weather_factory import WeatherFactory


@pytest.fixture
def problem_fixture():
    """Initializes a deterministic RoutingProblem instance from config."""
    config_path = "tests/config.regression.json"
    config = Config.assign_config(config_path)
    ship_config = ShipConfig.assign_config(config_path)

    lat1, lon1, lat2, lon2 = config.DEFAULT_MAP
    default_map = Map(lat1, lon1, lat2, lon2)

    wt = WeatherFactory.get_weather(
        config._DATA_MODE_WEATHER,
        config.WEATHER_DATA,
        config.DEPARTURE_TIME,
        config.TIME_FORECAST,
        config.DELTA_TIME_FORECAST,
        default_map,
    )
    boat = ShipFactory.get_ship(config.BOAT_TYPE, ship_config)
    constraint_list = ConstraintsListFactory.get_constraints_list(
        constraints_string_list=config.CONSTRAINTS_LIST,
        data_mode=config._DATA_MODE_DEPTH,
        min_depth=boat.get_required_water_depth(),
        map_size=default_map,
        depthfile=config.DEPTH_DATA,
        waypoints=config.INTERMEDIATE_WAYPOINTS,
        courses_path=config.COURSES_FILE,
        time_frame=(wt.time_start, wt.time_end),
    )

    problem = RoutingProblem(
        departure_time=config.DEPARTURE_TIME,
        arrival_time=config.ARRIVAL_TIME,
        boat=boat,
        boat_speed=config.BOAT_SPEED,
        constraint_list=constraint_list,
        objectives=config.GENETIC_OBJECTIVES,
        symmetric_time_objective=config.GENETIC_SYMMETRIC_TIME_OBJECTIVE,
    )
    return problem


def test_routing_problem_get_power(problem_fixture):
    """
    Unit test for RoutingProblem.get_power assuming current calculations
    are correct. Verifies output dictionary structure, positive fuel
    aggregation, and physical ship parameter quantities.
    """
    problem = problem_fixture

    # 3-point test route across Mediterranean bounds [lat, lon, speed]
    route = np.array([
        [38.1884, 2.3880, 7.0],
        [38.8711, 6.5033, 7.0],
        [39.5539, 10.6186, 7.0],
    ])

    result = problem.get_power(route)

    # 1. Dictionary structure verification
    assert isinstance(result, dict)
    assert "fuel_sum" in result
    assert "shipparams" in result
    assert "time_obj" in result
    assert "start_times" in result

    # 2. Fuel consumption objective verification
    assert result["fuel_sum"] is not None
    fuel_val = (
        result["fuel_sum"].value
        if hasattr(result["fuel_sum"], "value")
        else result["fuel_sum"]
    )
    assert fuel_val > 0.0

    # 3. Ship parameters object verification
    shipparams = result["shipparams"]
    assert shipparams is not None

    power = shipparams.get_power()
    assert power is not None
    assert np.all(power.value > 0.0)
    assert str(power.unit) in ["kW", "W", "MW", "kg m2 / s3"]

    fuel_rate = shipparams.get_fuel_rate()
    assert fuel_rate is not None
    assert np.all(fuel_rate.value > 0.0)
