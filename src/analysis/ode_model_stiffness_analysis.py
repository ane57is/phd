import numpy as np
from scipy.integrate import solve_ivp
import time
from src.models.type_1_models.seir import (
    direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model
)
from src.parameters.params import (
    initial_populations_in_zones,
    zeleners_initial_delta_1,
    zeleners_initial_delta_2,
    zeleners_initial_delta_3,
    zeleners_initial_delta_4,
)
from src.parameters.params import smallpox_seir_params_with_starks_conversion
import logging
import json

logging.basicConfig(level=logging.INFO)
# Set up initial conditions and parameters
# The Antonine Plague part:
y0_ap = [
    initial_populations_in_zones["initial_christian_population_zone_1"] - 1,  # s1_c
    0,  # e1_c
    1,  # i1_c
    0,  # r1_c
    0,  # d1_c
    0,  # a1_c
    initial_populations_in_zones["initial_pagan_population_zone_1"] - 1,  # s1_p
    0,  # e1_p
    1,  # i1_p
    0,  # r1_p
    0,  # d1_p
    0,  # a1_p
    initial_populations_in_zones["initial_christian_population_zone_2"] - 1,  # s2_c
    0,  # e2_c
    1,  # i2_c
    0,  # r2_c
    0,  # d2_c
    0,  # a2_c
    initial_populations_in_zones["initial_pagan_population_zone_2"] - 1,  # s2_p
    0,  # e2_p
    1,  # i2_p
    0,  # r2_p
    0,  # d2_p
    0,  # a2_p
    initial_populations_in_zones["initial_christian_population_zone_3"] - 1,  # s3_c
    0,  # e3_c
    1,  # i3_c
    0,  # r3_c
    0,  # d3_c
    0,  # a3_c
    initial_populations_in_zones["initial_pagan_population_zone_3"] - 1,  # s3_p
    0,  # e3_p
    1,  # i3_p
    0,  # r3_p
    0,  # d3_p
    0,  # a3_p
    initial_populations_in_zones["initial_christian_population_zone_4"] - 1,  # s4_c
    0,  # e4_c
    1,  # i4_c
    0,  # r4_c
    0,  # d4_c
    0,  # a4_c
    initial_populations_in_zones["initial_pagan_population_zone_4"] - 1,  # s4_p
    0,  # e4_p
    1,  # i4_p
    0,  # r4_p
    0,  # d4_p
    0,  # a4_p
    zeleners_initial_delta_1,  # delta_1_c
    zeleners_initial_delta_1,  # delta_1_p
    zeleners_initial_delta_2,  # delta_2_c
    zeleners_initial_delta_2,  # delta_2_p
    zeleners_initial_delta_3,  # delta_3_c
    zeleners_initial_delta_3,  # delta_3_p
    zeleners_initial_delta_4,  # delta_4_c
    zeleners_initial_delta_4,  # delta_4_p
]

# Compare explicit vs implicit solvers
methods = ['RK45', 'BDF', 'Radau']
results = {}

for method in methods:
    start = time.time()
    sol = solve_ivp(
        lambda t, y: direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model(
            y, t, smallpox_seir_params_with_starks_conversion
        ),
        t_span=[0, 365*10],  # 10 years simulation
        y0=y0_ap,
        method=method,
        rtol=1e-6,
        atol=1e-9
    )
    end = time.time()

    results[method] = {
        'time': end - start,
        'nfev': sol.nfev,  # Number of function evaluations
        'njev': getattr(sol, 'njev', 0),  # Number of Jacobian evaluations
        'success': sol.success,
        'message': sol.message
    }


logging.info("Numerical experiment approach")
logging.info("Stiffness analysis results:")
# print(results)

# logging.basicConfig(level=logging.INFO)

logging.info(json.dumps(results, indent=2))


def compute_jacobian_at_point(func, y, t, parameters, h=1e-8):
    n = len(y)
    jacobian = np.zeros((n, n))

    for i in range(n):
        y_plus = y.copy()
        y_plus[i] += h
        f_plus = func(y_plus, t, parameters)

        y_minus = y.copy()
        y_minus[i] -= h
        f_minus = func(y_minus, t, parameters)

        jacobian[:, i] = (np.array(f_plus) - np.array(f_minus)) / (2 * h)

    return jacobian

# Compute at initial conditions
jac = compute_jacobian_at_point(
    direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model,
    y0_ap,
    0,
    smallpox_seir_params_with_starks_conversion
)

# Calculate eigenvalues
eigenvalues = np.linalg.eigvals(jac)

# Stiffness ratio
stiffness_ratio = max(abs(eigenvalues)) / min(abs(np.where(abs(eigenvalues) > 1e-10, eigenvalues, 1e10)))
logging.info("Jacobian analysis")
logging.info(f"Stiffness ratio: {stiffness_ratio}")