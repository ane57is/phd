import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Demographic parameters and initial populations shared across models
demographic_params = {
    "natural_birth_rate": 1 / (25 * 365),
    "natural_death_rate": 1 / (25 * 365),
}

# Total initial population
initial_christian_population = 40_000
initial_pagan_population = 60_000_000

# Initial populations for each zone:
# - "[Z]one 1 consists of the largest urban centers and their immediate hinterlands, the army and other highly mobile
#   portions of the population. The population of Zone 1 begins with 10 million, an initial interaction rate of 12 and
#   the 100 infected individuals" (Zelener 2003: 96).
# - "[Z]one 2 consists of smaller cities and the most densely populated regions, such as Egypt, Italy and coastal Asia
#   Minor. The initial population in Zone 2 is 20 million and the initial interaction rate is 9" (Zelener 2003: 96).
# - "[Z]one 3 consists of less dense regions, such as the Western provinces and the Danube region, with an initial
#   population of 20 million and an interaction rate of 8" (Zelener 2003: 96).
# - "[Z]one 4 has the remaining population of 10 million, an interaction rate of 7 and represents the most remote
#   regions of the Roman world, for example, Britain and Mauretania" (Zelener 2003: 96).
initial_populations_in_zones = {
    "initial_christian_population_zone_1": 6_667,
    "initial_pagan_population_zone_1": 10_000_000,
    "initial_christian_population_zone_2": 13_333,
    "initial_pagan_population_zone_2": 20_000_000,
    "initial_christian_population_zone_3": 13_333,
    "initial_pagan_population_zone_3": 20_000_000,
    "initial_christian_population_zone_4": 6_667,
    "initial_pagan_population_zone_4": 10_000_000,
}


# Initial interaction rates in the 4-zone model
zeleners_initial_delta_1 = 12
zeleners_initial_delta_2 = 9
zeleners_initial_delta_3 = 8
zeleners_initial_delta_4 = 7


def adjust_delta(delta_prev, delta_0, s_n, i_n, r_n, increment=0.1, threshold=10, max_delta=12):
    """
    Adjusts the interaction rate delta dynamically based on Zelener's equations.

    Parameters:
    - delta_prev: Previous interaction rate delta (delta_{n-1}).
    - delta_0: Initial maximum delta value.
    - s_n: Susceptible population in the zone.
    - i_n: Currently infected population.
    - r_n: Recovered/immune population.
    - increment: Increment step for delta adjustment.
    - threshold: Threshold value for S_n / D_n.

    Returns:
    - Updated interaction rate delta_n.
    """
    epsilon = 1e-4  # small value to prevent division by zero

    if i_n <= 0 and r_n <= 0:
        return delta_prev
    if i_n > 0 and s_n / i_n > threshold:  # Zelener's S_n / D_n > 10
        delta_n = min(delta_0, delta_prev + increment)
    elif r_n > 0 and s_n / r_n < threshold:  # S_n / I_n < 10
        # delta_n = s_n / r_n
        delta_n = s_n / max(r_n, epsilon)
    elif i_n > 0:  # Zelener's default to S_n / D_n
        # delta_n = s_n / i_n
        delta_n = s_n / max(i_n, epsilon)
    else:  # No infected individuals; use previous delta
        delta_n = delta_prev

    return max(0.01, min(delta_n, max_delta))
    # return max(delta_0, min(delta_n, max_delta))


class BaseModelParameters:
    def __init__(
        self,
        beta,  # transmission rate
        sigma,  # incubation period
        gamma,  # infectious period
        natural_birth_rate,
        natural_death_rate,
        initial_delta_1=None,  # interaction rate in zone 1
        initial_delta_2=None,  # interaction rate in zone 2
        initial_delta_3=None,  # interaction rate in zone 3
        initial_delta_4=None,  # interaction rate in zone 4
        unified_deltas=None,  # use unified deltas for both Christian and Pagan compartments
        fatality_rate=None,
        fatality_rate_p=None,
        fatality_rate_c=None,
        conversion_rate_decennial=None,
        max_delta=None,
    ):
        # Disease-specific parameters
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

        # Interaction rates
        self.initial_delta_1 = initial_delta_1
        self.initial_delta_2 = initial_delta_2
        self.initial_delta_3 = initial_delta_3
        self.initial_delta_4 = initial_delta_4
        self.unified_deltas = unified_deltas
        self.max_delta = max_delta

        # TODO: check if it makes sense to always have 3 variables as such:
        # - fatality_rate
        # - fatality_rate_treated or fatality_rate_nursing
        # - fatality_rate_stark_christian
        #
        # This way I could have only one BaseModelParameters instance per disease, and based on the model
        # I would use only some of the parameters:
        # - in the simple one just fatality_rate for both Pagans and Christians
        # - in the 1.1 models (nursing/treatment for Christians) fatality_rate_treated or _nursing as
        #   fatality_rate_c and the basic fatality_rate as fatality_rate_p
        # - in the 1.2 models fatality_rate_stark_christian as fatality_rate_c and fatality_rate as fatality_rate_p

        if fatality_rate is not None:
            self.fatality_rate = fatality_rate
            self.fatality_rate_p = fatality_rate
            self.fatality_rate_c = fatality_rate
        else:
            self.fatality_rate_p = fatality_rate_p if fatality_rate_p is not None else 0
            self.fatality_rate_c = (
                fatality_rate_c
                if fatality_rate_c is not None
                else (self.fatality_rate_p / 3)
            )

        # Demographic parameters of humans
        self.natural_birth_rate = natural_birth_rate
        self.natural_death_rate = natural_death_rate

        # Conversion rate parameters
        self.conversion_rate_decennial = conversion_rate_decennial

        if conversion_rate_decennial is not None:
            self.conversion_rate_annual = self.calculate_annual_rate(
                conversion_rate_decennial
            )
            self.conversion_rate_daily = self.calculate_daily_rate(
                self.conversion_rate_annual
            )
        else:
            self.conversion_rate_annual = None
            self.conversion_rate_daily = None

    @staticmethod
    def calculate_annual_rate(decennial_rate):
        return (1 + decennial_rate) ** (1 / 10) - 1

    @staticmethod
    def calculate_daily_rate(annual_rate):
        return (1 + annual_rate) ** (1 / 365) - 1


# Instantiation of the parameters (using smallpox values for default)
default_seir_params = BaseModelParameters(
    beta=0.584, sigma=1 / 12, gamma=1 / 9.5, fatality_rate=0.9, **demographic_params
)
default_two_cfrs_params = BaseModelParameters(
    beta=0.584, sigma=1 / 12, gamma=1 / 9.5, fatality_rate_p=0.9, **demographic_params
)
# smallpox_seir_params = SmallpoxSEIRParams(beta=0.4, sigma=0.1, gamma=0.05, fatality_rate=0.3, **demographic_params)
measles_seir_params = BaseModelParameters(
    beta=1.175, sigma=1 / 10, gamma=1 / 13.5, fatality_rate=0.3, **demographic_params
)
measles_seir_params_with_lower_cfr_for_c_and_starks_conversion = BaseModelParameters(
    beta=1.175,
    sigma=1 / 10,
    gamma=1 / 13.5,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.3,
    conversion_rate_decennial=0.4,
    **demographic_params
)
smallpox_seir_params_with_starks_conversion = BaseModelParameters(
    beta=0.584,
    sigma=1 / 12,
    gamma=1 / 9.5,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.9,
    conversion_rate_decennial=0.4,
    **demographic_params
)


def direct_transmission_over_two_connected_subpopulations_seird_model(y, t, parameters):
    s_c, e_c, i_c, r_c, d_c, a_c, s_p, e_p, i_p, r_p, d_p, a_p = y
    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    fatality_rate = parameters.fatality_rate
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate

    # Totals of subpopulations and whole population
    n_c = s_c + e_c + i_c + r_c
    n_p = s_p + e_p + i_p + r_p
    n = n_c + n_p

    # Christian compartments:
    ds_c = b * n_c - beta * s_c * (i_c + i_p) / n - d * s_c
    de_c = beta * s_c * (i_c + i_p) / n - sigma * e_c - d * e_c
    di_c = sigma * e_c - gamma * i_c - d * i_c
    dr_c = (1 - fatality_rate) * gamma * i_c - d * r_c
    dd_c = fatality_rate * gamma * i_c
    da_c = d * n_c

    # Pagan compartments:
    ds_p = b * n_p - beta * s_p * (i_c + i_p) / n - d * s_p
    de_p = beta * s_p * (i_c + i_p) / n - sigma * e_p - d * e_p
    di_p = sigma * e_p - gamma * i_p - d * i_p
    dr_p = (1 - fatality_rate) * gamma * i_p - d * r_p
    dd_p = fatality_rate * gamma * i_p
    da_p = d * n_p

    # Return the derivatives in the same order
    return [ds_c, de_c, di_c, dr_c, dd_c, da_c, ds_p, de_p, di_p, dr_p, dd_p, da_p]


def direct_transmission_over_one_population_as_in_plos_paper(y, t, parameters):
    s_c, e_c, i_c, r_c, d_c, a_c, s_p, e_p, i_p, r_p, d_p, a_p = y
    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    fatality_rate = parameters.fatality_rate
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate

    # Totals of subpopulations and whole population
    n_p = s_p + e_p + i_p + r_p

    # Christian compartments:
    ds_c = 0
    de_c = 0
    di_c = 0
    dr_c = 0
    dd_c = 0
    da_c = 0

    # Pagan compartments:
    ds_p = b * n_p - beta * s_p * i_p / n_p - d * s_p
    de_p = beta * s_p * i_p / n_p - sigma * e_p - d * e_p
    di_p = sigma * e_p - gamma * i_p - d * i_p
    dr_p = (1 - fatality_rate) * gamma * i_p - d * r_p
    dd_p = fatality_rate * gamma * i_p
    da_p = d * n_p

    # Return the derivatives in the same order
    return [ds_c, de_c, di_c, dr_c, dd_c, da_c, ds_p, de_p, di_p, dr_p, dd_p, da_p]


def simple_demographic_model(y, t, parameters):
    s_c, e_c, i_c, r_c, d_c, a_c, s_p, e_p, i_p, r_p, d_p, a_p = y
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate

    # Totals of subpopulations
    n_c = s_c + e_c + i_c + r_c
    n_p = s_p + e_p + i_p + r_p

    # Christian compartments:
    ds_c = b * n_c - d * s_c
    de_c = -d * e_c
    di_c = -d * i_c
    dr_c = -d * r_c
    dd_c = 0
    da_c = d * n_c

    # Pagan compartments:
    ds_p = b * n_p - d * s_p
    de_p = -d * e_p
    di_p = -d * i_p
    dr_p = -d * r_p
    dd_p = 0
    da_p = d * n_p

    # Return the derivatives in the same order
    return [ds_c, de_c, di_c, dr_c, dd_c, da_c, ds_p, de_p, di_p, dr_p, dd_p, da_p]


def direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model(
    y, t, parameters
):
    """
    Extension of direct_transmission_over_two_connected_subpopulations_seird_model with two CFRs.
    """

    s_c, e_c, i_c, r_c, d_c, a_c, s_p, e_p, i_p, r_p, d_p, a_p = y
    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    fatality_rate_p = parameters.fatality_rate_p
    fatality_rate_c = parameters.fatality_rate_c
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate

    # Totals of subpopulations and whole population
    n_c = s_c + e_c + i_c + r_c
    n_p = s_p + e_p + i_p + r_p
    n = n_c + n_p

    # Christian compartments:
    ds_c = b * n_c - beta * s_c * (i_c + i_p) / n - d * s_c
    de_c = beta * s_c * (i_c + i_p) / n - sigma * e_c - d * e_c
    di_c = sigma * e_c - gamma * i_c - d * i_c
    dr_c = (1 - fatality_rate_c) * gamma * i_c - d * r_c
    dd_c = fatality_rate_c * gamma * i_c
    da_c = d * n_c

    # Pagan compartments:
    ds_p = b * n_p - beta * s_p * (i_c + i_p) / n - d * s_p
    de_p = beta * s_p * (i_c + i_p) / n - sigma * e_p - d * e_p
    di_p = sigma * e_p - gamma * i_p - d * i_p
    dr_p = (1 - fatality_rate_p) * gamma * i_p - d * r_p
    dd_p = fatality_rate_p * gamma * i_p
    da_p = d * n_p

    # Return the derivatives in the same order
    return [ds_c, de_c, di_c, dr_c, dd_c, da_c, ds_p, de_p, di_p, dr_p, dd_p, da_p]


def simple_demographic_model_with_conversion(y, t, parameters):
    s_c, e_c, i_c, r_c, d_c, a_c, s_p, e_p, i_p, r_p, d_p, a_p = y
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily

    # Totals of subpopulations
    n_c = s_c + e_c + i_c + r_c
    n_p = s_p + e_p + i_p + r_p

    # Pagans converted to Christianity for all living compartments:
    converted_s = conversion_rate * s_c
    converted_e = conversion_rate * e_c
    converted_i = conversion_rate * i_c
    converted_r = conversion_rate * r_c

    # Christian compartments:
    ds_c = b * n_c - d * s_c + converted_s
    de_c = -d * e_c + converted_e
    di_c = -d * i_c + converted_i
    dr_c = -d * r_c + converted_r
    dd_c = 0
    da_c = d * n_c

    # Pagan compartments:
    ds_p = b * n_p - d * s_p - converted_s
    de_p = -d * e_p - converted_e
    di_p = -d * i_p - converted_i
    dr_p = -d * r_p - converted_r
    dd_p = 0
    da_p = d * n_p

    # Return the derivatives in the same order
    return [ds_c, de_c, di_c, dr_c, dd_c, da_c, ds_p, de_p, di_p, dr_p, dd_p, da_p]


def direct_transmission_over_two_connected_subpopulations_with_two_cfrs_and_conversion_seird_model(
    y, t, parameters
):
    """
    Extension of direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model with conversion.
    """

    s_c, e_c, i_c, r_c, d_c, a_c, s_p, e_p, i_p, r_p, d_p, a_p = y
    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    fatality_rate_p = parameters.fatality_rate_p
    fatality_rate_c = parameters.fatality_rate_c
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily

    # Totals of subpopulations and whole population
    n_c = s_c + e_c + i_c + r_c
    n_p = s_p + e_p + i_p + r_p
    n = n_c + n_p

    # Pagans converted to Christianity for all living compartments:
    converted_s = conversion_rate * s_c
    converted_e = conversion_rate * e_c
    converted_i = conversion_rate * i_c
    converted_r = conversion_rate * r_c

    # Christian compartments:
    ds_c = b * n_c - beta * s_c * (i_c + i_p) / n - d * s_c + converted_s
    de_c = beta * s_c * (i_c + i_p) / n - sigma * e_c - d * e_c + converted_e
    di_c = sigma * e_c - gamma * i_c - d * i_c + converted_i
    dr_c = (1 - fatality_rate_c) * gamma * i_c - d * r_c + converted_r
    dd_c = fatality_rate_c * gamma * i_c
    da_c = d * n_c

    # Pagan compartments:
    ds_p = b * n_p - beta * s_p * (i_c + i_p) / n - d * s_p - converted_s
    de_p = beta * s_p * (i_c + i_p) / n - sigma * e_p - d * e_p - converted_e
    di_p = sigma * e_p - gamma * i_p - d * i_p - converted_i
    dr_p = (1 - fatality_rate_p) * gamma * i_p - d * r_p - converted_r
    dd_p = fatality_rate_p * gamma * i_p
    da_p = d * n_p

    # Return the derivatives in the same order
    return [ds_c, de_c, di_c, dr_c, dd_c, da_c, ds_p, de_p, di_p, dr_p, dd_p, da_p]


def direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_conversion_in_pairs_seird_model(
    y, t, parameters
):
    """
    Extension of direct_transmission_over_two_connected_subpopulations_with_two_cfrs_and_conversion_seird_model by
    four zones. Each zone has a pair of Christian and Pagan subpopulations. Each pair of subpopulations has a conversion
    rate mechanism. Within pairs of connected subpopulations (inside of four separate zones) a dynamic interaction rate
    modifies the impact of beta (the transmission rate) by the sum of these subpopulations (the total population of
    a zone).
    """

    (
        s1_c, e1_c, i1_c, r1_c, d1_c, a1_c, s1_p, e1_p, i1_p, r1_p, d1_p, a1_p,
        s2_c, e2_c, i2_c, r2_c, d2_c, a2_c, s2_p, e2_p, i2_p, r2_p, d2_p, a2_p,
        s3_c, e3_c, i3_c, r3_c, d3_c, a3_c, s3_p, e3_p, i3_p, r3_p, d3_p, a3_p,
        s4_c, e4_c, i4_c, r4_c, d4_c, a4_c, s4_p, e4_p, i4_p, r4_p, d4_p, a4_p
    ) = y

    beta = parameters.beta
    delta = None  # TODO: implement calculation of delta
    sigma = parameters.sigma
    gamma = parameters.gamma
    fatality_rate_p = parameters.fatality_rate_p
    fatality_rate_c = parameters.fatality_rate_c
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily

    # Totals of subpopulations in each zone and whole population
    n1_c = s1_c + e1_c + i1_c + r1_c
    n1_p = s1_p + e1_p + i1_p + r1_p
    n1 = n1_c + n1_p

    n2_c = s2_c + e2_c + i2_c + r2_c
    n2_p = s2_p + e2_p + i2_p + r2_p
    n2 = n2_c + n2_p

    n3_c = s3_c + e3_c + i3_c + r3_c
    n3_p = s3_p + e3_p + i3_p + r3_p
    n3 = n3_c + n3_p

    n4_c = s4_c + e4_c + i4_c + r4_c
    n4_p = s4_p + e4_p + i4_p + r4_p
    n4 = n4_c + n4_p

    # TODO: not sure if there is a need for this n variable as long as there is no interaction between zones 🤷🤔
    n = n1 + n2 + n3 + n4

    # Pagans converted to Christianity for all living compartments in each zone
    converted_s1 = conversion_rate * s1_p
    converted_e1 = conversion_rate * e1_p
    converted_i1 = conversion_rate * i1_p
    converted_r1 = conversion_rate * r1_p

    converted_s2 = conversion_rate * s2_p
    converted_e2 = conversion_rate * e2_p
    converted_i2 = conversion_rate * i2_p
    converted_r2 = conversion_rate * r2_p

    converted_s3 = conversion_rate * s3_p
    converted_e3 = conversion_rate * e3_p
    converted_i3 = conversion_rate * i3_p
    converted_r3 = conversion_rate * r3_p

    converted_s4 = conversion_rate * s4_p
    converted_e4 = conversion_rate * e4_p
    converted_i4 = conversion_rate * i4_p
    converted_r4 = conversion_rate * r4_p

    # Zone 1
    # Christian compartments in Zone 1
    ds1_c = b * n1_c - beta * s1_c * (i1_c + i1_p) / n1 - d * s1_c + converted_s1
    de1_c = beta * s1_c * (i1_c + i1_p) / n1 - sigma * e1_c - d * e1_c + converted_e1
    di1_c = sigma * e1_c - gamma * i1_c - d * i1_c + converted_i1
    dr1_c = (1 - fatality_rate_c) * gamma * i1_c - d * r1_c + converted_r1
    dd1_c = fatality_rate_c * gamma * i1_c
    da1_c = d * n1_c

    # Pagan compartments in Zone 1
    ds1_p = b * n1_p - beta * s1_p * (i1_c + i1_p) / n1 - d * s1_p - converted_s1
    de1_p = beta * s1_p * (i1_c + i1_p) / n1 - sigma * e1_p - d * e1_p - converted_e1
    di1_p = sigma * e1_p - gamma * i1_p - d * i1_p - converted_i1
    dr1_p = (1 - fatality_rate_p) * gamma * i1_p - d * r1_p - converted_r1
    dd1_p = fatality_rate_p * gamma * i1_p
    da1_p = d * n1_p

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = b * n2_c - beta * s2_c * (i2_c + i2_p) / n2 - d * s2_c + converted_s2
    de2_c = beta * s2_c * (i2_c + i2_p) / n2 - sigma * e2_c - d * e2_c + converted_e2
    di2_c = sigma * e2_c - gamma * i2_c - d * i2_c + converted_i2
    dr2_c = (1 - fatality_rate_c) * gamma * i2_c - d * r2_c + converted_r2
    dd2_c = fatality_rate_c * gamma * i2_c
    da2_c = d * n2_c

    # Pagan compartments in Zone 2
    ds2_p = b * n2_p - beta * s2_p * (i2_c + i2_p) / n2 - d * s2_p - converted_s2
    de2_p = beta * s2_p * (i2_c + i2_p) / n2 - sigma * e2_p - d * e2_p - converted_e2
    di2_p = sigma * e2_p - gamma * i2_p - d * i2_p - converted_i2
    dr2_p = (1 - fatality_rate_p) * gamma * i2_p - d * r2_p - converted_r2
    dd2_p = fatality_rate_p * gamma * i2_p
    da2_p = d * n2_p

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = b * n3_c - beta * s3_c * (i3_c + i3_p) / n3 - d * s3_c + converted_s3
    de3_c = beta * s3_c * (i3_c + i3_p) / n3 - sigma * e3_c - d * e3_c + converted_e3
    di3_c = sigma * e3_c - gamma * i3_c - d * i3_c + converted_i3
    dr3_c = (1 - fatality_rate_c) * gamma * i3_c - d * r3_c + converted_r3
    dd3_c = fatality_rate_c * gamma * i3_c
    da3_c = d * n3_c

    # Pagan compartments in Zone 3
    ds3_p = b * n3_p - beta * s3_p * (i3_c + i3_p) / n3 - d * s3_p - converted_s3
    de3_p = beta * s3_p * (i3_c + i3_p) / n3 - sigma * e3_p - d * e3_p - converted_e3
    di3_p = sigma * e3_p - gamma * i3_p - d * i3_p - converted_i3
    dr3_p = (1 - fatality_rate_p) * gamma * i3_p - d * r3_p - converted_r3
    dd3_p = fatality_rate_p * gamma * i3_p
    da3_p = d * n3_p

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = b * n4_c - beta * s4_c * (i4_c + i4_p) / n4 - d * s4_c + converted_s4
    de4_c = beta * s4_c * (i4_c + i4_p) / n4 - sigma * e4_c - d * e4_c + converted_e4
    di4_c = sigma * e4_c - gamma * i4_c - d * i4_c + converted_i4
    dr4_c = (1 - fatality_rate_c) * gamma * i4_c - d * r4_c + converted_r4
    dd4_c = fatality_rate_c * gamma * i4_c
    da4_c = d * n4_c

    # Pagan compartments in Zone 4
    ds4_p = b * n4_p - beta * s4_p * (i4_c + i4_p) / n4 - d * s4_p - converted_s4
    de4_p = beta * s4_p * (i4_c + i4_p) / n4 - sigma * e4_p - d * e4_p - converted_e4
    di4_p = sigma * e4_p - gamma * i4_p - d * i4_p - converted_i4
    dr4_p = (1 - fatality_rate_p) * gamma * i4_p - d * r4_p - converted_r4
    dd4_p = fatality_rate_p * gamma * i4_p
    da4_p = d * n4_p

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p
    ]


def simple_demographic_model_subpopulation_pairs_with_conversion_in_four_zones(y, t, parameters):
    """
    Extension of def simple_demographic_model_with_conversion by four zones.
    """

    (
        s1_c, e1_c, i1_c, r1_c, d1_c, a1_c, s1_p, e1_p, i1_p, r1_p, d1_p, a1_p,
        s2_c, e2_c, i2_c, r2_c, d2_c, a2_c, s2_p, e2_p, i2_p, r2_p, d2_p, a2_p,
        s3_c, e3_c, i3_c, r3_c, d3_c, a3_c, s3_p, e3_p, i3_p, r3_p, d3_p, a3_p,
        s4_c, e4_c, i4_c, r4_c, d4_c, a4_c, s4_p, e4_p, i4_p, r4_p, d4_p, a4_p
    ) = y
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily

    # Totals of subpopulations in each zone
    n1_c = s1_c + e1_c + i1_c + r1_c
    n1_p = s1_p + e1_p + i1_p + r1_p

    n2_c = s2_c + e2_c + i2_c + r2_c
    n2_p = s2_p + e2_p + i2_p + r2_p

    n3_c = s3_c + e3_c + i3_c + r3_c
    n3_p = s3_p + e3_p + i3_p + r3_p

    n4_c = s4_c + e4_c + i4_c + r4_c
    n4_p = s4_p + e4_p + i4_p + r4_p

    # Pagans converted to Christianity for all living compartments in each zone
    converted_s1 = min(conversion_rate * s1_p, s1_p)
    converted_e1 = min(conversion_rate * e1_p, e1_p)
    converted_i1 = min(conversion_rate * i1_p, i1_p)
    converted_r1 = min(conversion_rate * r1_p, r1_p)

    converted_s2 = min(conversion_rate * s2_p, s2_p)
    converted_e2 = min(conversion_rate * e2_p, e2_p)
    converted_i2 = min(conversion_rate * i2_p, i2_p)
    converted_r2 = min(conversion_rate * r2_p, r2_p)

    converted_s3 = min(conversion_rate * s3_p, s3_p)
    converted_e3 = min(conversion_rate * e3_p, e3_p)
    converted_i3 = min(conversion_rate * i3_p, i3_p)
    converted_r3 = min(conversion_rate * r3_p, r3_p)

    converted_s4 = min(conversion_rate * s4_p, s4_p)
    converted_e4 = min(conversion_rate * e4_p, e4_p)
    converted_i4 = min(conversion_rate * i4_p, i4_p)
    converted_r4 = min(conversion_rate * r4_p, r4_p)

    def non_negative(value):
        # return max(0, value)
        return value

    # Zone 1
    # Christian compartments in Zone 1
    ds1_c = non_negative(b * n1_c - d * s1_c + converted_s1)
    de1_c = non_negative(-d * e1_c + converted_e1)
    di1_c = non_negative(-d * i1_c + converted_i1)
    dr1_c = non_negative(-d * r1_c + converted_r1)
    dd1_c = 0
    da1_c = non_negative(d * n1_c)

    # Pagan compartments in Zone 1
    ds1_p = non_negative(b * n1_p - d * s1_p - converted_s1)
    de1_p = non_negative(-d * e1_p - converted_e1)
    di1_p = non_negative(-d * i1_p - converted_i1)
    dr1_p = non_negative(-d * r1_p - converted_r1)
    dd1_p = 0
    da1_p = non_negative(d * n1_p)

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = non_negative(b * n2_c - d * s2_c + converted_s2)
    de2_c = non_negative(-d * e2_c + converted_e2)
    di2_c = non_negative(-d * i2_c + converted_i2)
    dr2_c = non_negative(-d * r2_c + converted_r2)
    dd2_c = 0
    da2_c = non_negative(d * n2_c)

    # Pagan compartments in Zone 2
    ds2_p = non_negative(b * n2_p - d * s2_p - converted_s2)
    de2_p = non_negative(-d * e2_p - converted_e2)
    di2_p = non_negative(-d * i2_p - converted_i2)
    dr2_p = non_negative(-d * r2_p - converted_r2)
    dd2_p = 0
    da2_p = non_negative(d * n2_p)

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = non_negative(b * n3_c - d * s3_c + converted_s3)
    de3_c = non_negative(-d * e3_c + converted_e3)
    di3_c = non_negative(-d * i3_c + converted_i3)
    dr3_c = non_negative(-d * r3_c + converted_r3)
    dd3_c = 0
    da3_c = non_negative(d * n3_c)

    # Pagan compartments in Zone 3
    ds3_p = non_negative(b * n3_p - d * s3_p - converted_s3)
    de3_p = non_negative(-d * e3_p - converted_e3)
    di3_p = non_negative(-d * i3_p - converted_i3)
    dr3_p = non_negative(-d * r3_p - converted_r3)
    dd3_p = 0
    da3_p = non_negative(d * n3_p)

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = non_negative(b * n4_c - d * s4_c + converted_s4)
    de4_c = non_negative(-d * e4_c + converted_e4)
    di4_c = non_negative(-d * i4_c + converted_i4)
    dr4_c = non_negative(-d * r4_c + converted_r4)
    dd4_c = 0
    da4_c = non_negative(d * n4_c)

    # Pagan compartments in Zone 4
    ds4_p = non_negative(b * n4_p - d * s4_p - converted_s4)
    de4_p = non_negative(-d * e4_p - converted_e4)
    di4_p = non_negative(-d * i4_p - converted_i4)
    dr4_p = non_negative(-d * r4_p - converted_r4)
    dd4_p = 0
    da4_p = non_negative(d * n4_p)

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p
    ]


def direct_transmission_with_four_dynamic_deltas_two_cfrs_and_conversion_in_pairs_seird_model(
    y, t, parameters
):
    """
    Extension of direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_conversion_in_pairs_seird_model
    by a dynamic delta (interaction rate). The current delta values are stored in the returned list of derivatives.

    """

    (
        s1_c, e1_c, i1_c, r1_c, d1_c, a1_c, s1_p, e1_p, i1_p, r1_p, d1_p, a1_p,
        s2_c, e2_c, i2_c, r2_c, d2_c, a2_c, s2_p, e2_p, i2_p, r2_p, d2_p, a2_p,
        s3_c, e3_c, i3_c, r3_c, d3_c, a3_c, s3_p, e3_p, i3_p, r3_p, d3_p, a3_p,
        s4_c, e4_c, i4_c, r4_c, d4_c, a4_c, s4_p, e4_p, i4_p, r4_p, d4_p, a4_p,
        delta_1_c, delta_1_p, delta_2_c, delta_2_p, delta_3_c, delta_3_p, delta_4_c, delta_4_p
    ) = y
    if t % 365 == 0:  # Example: print yearly updates
        print(f"Time {t}: delta_1_c={delta_1_c}, delta_1_p={delta_1_p}")

    # if t % (365 * 10) == 0:  # Log every 10 years
    #     print(f"Time {t}: delta_1_c={delta_1_c}, delta_1_p={delta_1_p}")

    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    # unified_deltas = parameters.unified_deltas
    unified_deltas = False
    fatality_rate_p = parameters.fatality_rate_p
    fatality_rate_c = parameters.fatality_rate_c
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily
    max_delta = parameters.max_delta

    if unified_deltas:
        updated_delta_1_c = adjust_delta(
            delta_prev=delta_1_c, delta_0=zeleners_initial_delta_1, s_n=s1_c + s1_p, i_n=i1_c + i1_p, r_n=r1_c + r1_p
        )
        updated_delta_1_p = updated_delta_1_c

        updated_delta_2_c = adjust_delta(
            delta_prev=delta_2_c, delta_0=zeleners_initial_delta_2, s_n=s2_c + s2_p, i_n=i2_c + i2_p, r_n=r2_c + r2_p
        )
        updated_delta_2_p = updated_delta_2_c

        updated_delta_3_c = adjust_delta(
            delta_prev=delta_3_c, delta_0=zeleners_initial_delta_3, s_n=s3_c + s3_p, i_n=i3_c + i3_p, r_n=r3_c + r3_p
        )
        updated_delta_3_p = updated_delta_3_c

        updated_delta_4_c = adjust_delta(
            delta_prev=delta_4_c, delta_0=zeleners_initial_delta_4, s_n=s4_c + s4_p, i_n=i4_c + i4_p, r_n=r4_c + r4_p
        )
        updated_delta_4_p = updated_delta_4_c

    else:
        updated_delta_1_c = adjust_delta(
            delta_prev=delta_1_c, delta_0=zeleners_initial_delta_1, s_n=s1_c, i_n=i1_c, r_n=r1_c
        )
        updated_delta_1_p = adjust_delta(
            delta_prev=delta_1_p, delta_0=zeleners_initial_delta_1, s_n=s1_p, i_n=i1_p, r_n=r1_p
        )
        updated_delta_2_c = adjust_delta(
            delta_prev=delta_2_c, delta_0=zeleners_initial_delta_2, s_n=s2_c, i_n=i2_c, r_n=r2_c
        )
        updated_delta_2_p = adjust_delta(
            delta_prev=delta_2_p, delta_0=zeleners_initial_delta_2, s_n=s2_p, i_n=i2_p, r_n=r2_p
        )
        updated_delta_3_c = adjust_delta(
            delta_prev=delta_3_c, delta_0=zeleners_initial_delta_3, s_n=s3_c, i_n=i3_c, r_n=r3_c
        )
        updated_delta_3_p = adjust_delta(
            delta_prev=delta_3_p, delta_0=zeleners_initial_delta_3, s_n=s3_p, i_n=i3_p, r_n=r3_p
        )
        updated_delta_4_c = adjust_delta(
            delta_prev=delta_4_c, delta_0=zeleners_initial_delta_4, s_n=s4_c, i_n=i4_c, r_n=r4_c
        )
        updated_delta_4_p = adjust_delta(
            delta_prev=delta_4_p, delta_0=zeleners_initial_delta_4, s_n=s4_p, i_n=i4_p, r_n=r4_p
        )

    # Totals of subpopulations in each zone and whole population
    n1_c = s1_c + e1_c + i1_c + r1_c
    n1_p = s1_p + e1_p + i1_p + r1_p
    # n1 = n1_c + n1_p

    n2_c = s2_c + e2_c + i2_c + r2_c
    n2_p = s2_p + e2_p + i2_p + r2_p
    # n2 = n2_c + n2_p

    n3_c = s3_c + e3_c + i3_c + r3_c
    n3_p = s3_p + e3_p + i3_p + r3_p
    # n3 = n3_c + n3_p

    n4_c = s4_c + e4_c + i4_c + r4_c
    n4_p = s4_p + e4_p + i4_p + r4_p
    # n4 = n4_c + n4_p

    # TODO: not sure if there is a need for this n variable as long as there is no interaction between zones 🤷🤔
    # n = n1 + n2 + n3 + n4

    # Pagans converted to Christianity for all living compartments in each zone
    converted_s1 = min(conversion_rate * s1_p, s1_p)
    converted_e1 = min(conversion_rate * e1_p, e1_p)
    converted_i1 = min(conversion_rate * i1_p, i1_p)
    converted_r1 = min(conversion_rate * r1_p, r1_p)

    converted_s2 = min(conversion_rate * s2_p, s2_p)
    converted_e2 = min(conversion_rate * e2_p, e2_p)
    converted_i2 = min(conversion_rate * i2_p, i2_p)
    converted_r2 = min(conversion_rate * r2_p, r2_p)

    converted_s3 = min(conversion_rate * s3_p, s3_p)
    converted_e3 = min(conversion_rate * e3_p, e3_p)
    converted_i3 = min(conversion_rate * i3_p, i3_p)
    converted_r3 = min(conversion_rate * r3_p, r3_p)

    converted_s4 = min(conversion_rate * s4_p, s4_p)
    converted_e4 = min(conversion_rate * e4_p, e4_p)
    converted_i4 = min(conversion_rate * i4_p, i4_p)
    converted_r4 = min(conversion_rate * r4_p, r4_p)

    def non_negative(value):
        return max(0, value)
        # return value
        # return max(0, min(value, 1e10))  # Cap at 10 billion
        # return max(0, min(value, 10_000_000))  # Cap at 10 billion

    # Zone 1
    # Christian compartments in Zone 1
    ds1_c = non_negative(b * n1_c - beta * updated_delta_1_c * (i1_c + i1_p) / max_delta - d * s1_c + converted_s1)
    de1_c = non_negative(beta * updated_delta_1_c * (i1_c + i1_p) / max_delta - sigma * e1_c - d * e1_c + converted_e1)
    di1_c = non_negative(sigma * e1_c - gamma * i1_c - d * i1_c + converted_i1)
    dr1_c = non_negative((1 - fatality_rate_c) * gamma * i1_c - d * r1_c + converted_r1)
    dd1_c = non_negative(fatality_rate_c * gamma * i1_c)
    da1_c = non_negative(d * n1_c)

    # Pagan compartments in Zone 1
    ds1_p = non_negative(b * n1_p - beta * updated_delta_1_p * (i1_c + i1_p) / max_delta - d * s1_p - converted_s1)
    de1_p = non_negative(beta * updated_delta_1_p * (i1_c + i1_p) / max_delta - sigma * e1_p - d * e1_p - converted_e1)
    di1_p = non_negative(sigma * e1_p - gamma * i1_p - d * i1_p - converted_i1)
    dr1_p = non_negative((1 - fatality_rate_p) * gamma * i1_p - d * r1_p - converted_r1)
    dd1_p = non_negative(fatality_rate_p * gamma * i1_p)
    da1_p = non_negative(d * n1_p)

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = non_negative(b * n2_c - beta * updated_delta_2_c * (i2_c + i2_p) / max_delta - d * s2_c + converted_s2)
    de2_c = non_negative(beta * updated_delta_2_c * (i2_c + i2_p) / max_delta - sigma * e2_c - d * e2_c + converted_e2)
    di2_c = non_negative(sigma * e2_c - gamma * i2_c - d * i2_c + converted_i2)
    dr2_c = non_negative((1 - fatality_rate_c) * gamma * i2_c - d * r2_c + converted_r2)
    dd2_c = non_negative(fatality_rate_c * gamma * i2_c)
    da2_c = non_negative(d * n2_c)

    # Pagan compartments in Zone 2
    ds2_p = non_negative(b * n2_p - beta * updated_delta_2_p * (i2_c + i2_p) / max_delta - d * s2_p - converted_s2)
    de2_p = non_negative(beta * updated_delta_2_p * (i2_c + i2_p) / max_delta - sigma * e2_p - d * e2_p - converted_e2)
    di2_p = non_negative(sigma * e2_p - gamma * i2_p - d * i2_p - converted_i2)
    dr2_p = non_negative((1 - fatality_rate_p) * gamma * i2_p - d * r2_p - converted_r2)
    dd2_p = non_negative(fatality_rate_p * gamma * i2_p)
    da2_p = non_negative(d * n2_p)

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = non_negative(b * n3_c - beta * updated_delta_3_c * (i3_c + i3_p) / max_delta - d * s3_c + converted_s3)
    de3_c = non_negative(beta * updated_delta_3_c * (i3_c + i3_p) / max_delta - sigma * e3_c - d * e3_c + converted_e3)
    di3_c = non_negative(sigma * e3_c - gamma * i3_c - d * i3_c + converted_i3)
    dr3_c = non_negative((1 - fatality_rate_c) * gamma * i3_c - d * r3_c + converted_r3)
    dd3_c = non_negative(fatality_rate_c * gamma * i3_c)
    da3_c = non_negative(d * n3_c)

    # Pagan compartments in Zone 3
    ds3_p = non_negative(b * n3_p - beta * updated_delta_3_p * (i3_c + i3_p) / max_delta - d * s3_p - converted_s3)
    de3_p = non_negative(beta * updated_delta_3_p * (i3_c + i3_p) / max_delta - sigma * e3_p - d * e3_p - converted_e3)
    di3_p = non_negative(sigma * e3_p - gamma * i3_p - d * i3_p - converted_i3)
    dr3_p = non_negative((1 - fatality_rate_p) * gamma * i3_p - d * r3_p - converted_r3)
    dd3_p = non_negative(fatality_rate_p * gamma * i3_p)
    da3_p = non_negative(d * n3_p)

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = non_negative(b * n4_c - beta * updated_delta_4_c * (i4_c + i4_p) / max_delta - d * s4_c + converted_s4)
    de4_c = non_negative(beta * updated_delta_4_c * (i4_c + i4_p) / max_delta - sigma * e4_c - d * e4_c + converted_e4)
    di4_c = non_negative(sigma * e4_c - gamma * i4_c - d * i4_c + converted_i4)
    dr4_c = non_negative((1 - fatality_rate_c) * gamma * i4_c - d * r4_c + converted_r4)
    dd4_c = non_negative(fatality_rate_c * gamma * i4_c)
    da4_c = non_negative(d * n4_c)

    # Pagan compartments in Zone 4
    ds4_p = non_negative(b * n4_p - beta * updated_delta_4_p * (i4_c + i4_p) / max_delta - d * s4_p - converted_s4)
    de4_p = non_negative(beta * updated_delta_4_p * (i4_c + i4_p) / max_delta - sigma * e4_p - d * e4_p - converted_e4)
    di4_p = non_negative(sigma * e4_p - gamma * i4_p - d * i4_p - converted_i4)
    dr4_p = non_negative((1 - fatality_rate_p) * gamma * i4_p - d * r4_p - converted_r4)
    dd4_p = non_negative(fatality_rate_p * gamma * i4_p)
    da4_p = non_negative(d * n4_p)

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p,
        updated_delta_1_c, updated_delta_1_p, updated_delta_2_c, updated_delta_2_p,
        updated_delta_3_c, updated_delta_3_p, updated_delta_4_c, updated_delta_4_p
    ]


def direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model(y, t, parameters):
    """
    New take on a seir model with interaction rate delta in 4 zones.

    Basing it on
    def direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_conversion_in_pairs_seird_model
    """
    (
        s1_c, e1_c, i1_c, r1_c, d1_c, a1_c, s1_p, e1_p, i1_p, r1_p, d1_p, a1_p,
        s2_c, e2_c, i2_c, r2_c, d2_c, a2_c, s2_p, e2_p, i2_p, r2_p, d2_p, a2_p,
        s3_c, e3_c, i3_c, r3_c, d3_c, a3_c, s3_p, e3_p, i3_p, r3_p, d3_p, a3_p,
        s4_c, e4_c, i4_c, r4_c, d4_c, a4_c, s4_p, e4_p, i4_p, r4_p, d4_p, a4_p,
        delta_1_c, delta_1_p, delta_2_c, delta_2_p, delta_3_c, delta_3_p, delta_4_c, delta_4_p
    ) = y

    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    unified_deltas = parameters.unified_deltas
    fatality_rate_p = parameters.fatality_rate_p
    fatality_rate_c = parameters.fatality_rate_c
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily
    max_delta = parameters.max_delta

    epsilon = 1e-4  # Small value to prevent division by zero

    if unified_deltas:
        ddelta_1_c = adjust_delta(
            delta_prev=delta_1_c, delta_0=zeleners_initial_delta_1, s_n=s1_c + s1_p, i_n=i1_c + i1_p, r_n=r1_c + r1_p
        )
        ddelta_1_p = ddelta_1_c

        ddelta_2_c = adjust_delta(
            delta_prev=delta_2_c, delta_0=zeleners_initial_delta_2, s_n=s2_c + s2_p, i_n=i2_c + i2_p, r_n=r2_c + r2_p
        )
        ddelta_2_p = ddelta_2_c

        ddelta_3_c = adjust_delta(
            delta_prev=delta_3_c, delta_0=zeleners_initial_delta_3, s_n=s3_c + s3_p, i_n=i3_c + i3_p, r_n=r3_c + r3_p
        )
        ddelta_3_p = ddelta_3_c

        ddelta_4_c = adjust_delta(
            delta_prev=delta_4_c, delta_0=zeleners_initial_delta_4, s_n=s4_c + s4_p, i_n=i4_c + i4_p, r_n=r4_c + r4_p
        )
        ddelta_4_p = ddelta_4_c

    else:
        ddelta_1_c = adjust_delta(
            delta_prev=delta_1_c, delta_0=zeleners_initial_delta_1, s_n=s1_c, i_n=i1_c, r_n=r1_c
        )
        ddelta_1_p = adjust_delta(
            delta_prev=delta_1_p, delta_0=zeleners_initial_delta_1, s_n=s1_p, i_n=i1_p, r_n=r1_p
        )
        ddelta_2_c = adjust_delta(
            delta_prev=delta_2_c, delta_0=zeleners_initial_delta_2, s_n=s2_c, i_n=i2_c, r_n=r2_c
        )
        ddelta_2_p = adjust_delta(
            delta_prev=delta_2_p, delta_0=zeleners_initial_delta_2, s_n=s2_p, i_n=i2_p, r_n=r2_p
        )
        ddelta_3_c = adjust_delta(
            delta_prev=delta_3_c, delta_0=zeleners_initial_delta_3, s_n=s3_c, i_n=i3_c, r_n=r3_c
        )
        ddelta_3_p = adjust_delta(
            delta_prev=delta_3_p, delta_0=zeleners_initial_delta_3, s_n=s3_p, i_n=i3_p, r_n=r3_p
        )
        ddelta_4_c = adjust_delta(
            delta_prev=delta_4_c, delta_0=zeleners_initial_delta_4, s_n=s4_c, i_n=i4_c, r_n=r4_c
        )
        ddelta_4_p = adjust_delta(
            delta_prev=delta_4_p, delta_0=zeleners_initial_delta_4, s_n=s4_p, i_n=i4_p, r_n=r4_p
        )

    # Totals of subpopulations in each zone and whole population
    n1_c = max(s1_c + e1_c + i1_c + r1_c, epsilon)
    n1_p = max(s1_p + e1_p + i1_p + r1_p, epsilon)
    n1 = max(n1_c + n1_p, epsilon)

    n2_c = max(s2_c + e2_c + i2_c + r2_c, epsilon)
    n2_p = max(s2_p + e2_p + i2_p + r2_p, epsilon)
    n2 = max(n2_c + n2_p, epsilon)

    n3_c = max(s3_c + e3_c + i3_c + r3_c, epsilon)
    n3_p = max(s3_p + e3_p + i3_p + r3_p, epsilon)
    n3 = max(n3_c + n3_p, epsilon)

    n4_c = max(s4_c + e4_c + i4_c + r4_c, epsilon)
    n4_p = max(s4_p + e4_p + i4_p + r4_p, epsilon)
    n4 = max(n4_c + n4_p, epsilon)

    # Pagans converted to Christianity for all living compartments in each zone
    converted_s1 = min(conversion_rate * s1_p, s1_p)
    converted_e1 = min(conversion_rate * e1_p, e1_p)
    converted_i1 = min(conversion_rate * i1_p, i1_p)
    converted_r1 = min(conversion_rate * r1_p, r1_p)

    converted_s2 = min(conversion_rate * s2_p, s2_p)
    converted_e2 = min(conversion_rate * e2_p, e2_p)
    converted_i2 = min(conversion_rate * i2_p, i2_p)
    converted_r2 = min(conversion_rate * r2_p, r2_p)

    converted_s3 = min(conversion_rate * s3_p, s3_p)
    converted_e3 = min(conversion_rate * e3_p, e3_p)
    converted_i3 = min(conversion_rate * i3_p, i3_p)
    converted_r3 = min(conversion_rate * r3_p, r3_p)

    converted_s4 = min(conversion_rate * s4_p, s4_p)
    converted_e4 = min(conversion_rate * e4_p, e4_p)
    converted_i4 = min(conversion_rate * i4_p, i4_p)
    converted_r4 = min(conversion_rate * r4_p, r4_p)

    # Zone 1
    # Christian compartments in Zone 1
    ds1_c = b * n1_c - (ddelta_1_c / max_delta) * beta * s1_c * (i1_c + i1_p) / n1 - d * s1_c + converted_s1
    de1_c = (ddelta_1_c / max_delta) * beta * s1_c * (i1_c + i1_p) / n1 - sigma * e1_c - d * e1_c + converted_e1
    di1_c = sigma * e1_c - gamma * i1_c - d * i1_c + converted_i1
    dr1_c = (1 - fatality_rate_c) * gamma * i1_c - d * r1_c + converted_r1
    dd1_c = fatality_rate_c * gamma * i1_c
    da1_c = d * n1_c

    # Pagan compartments in Zone 1
    ds1_p = b * n1_p - (ddelta_1_p / max_delta) * beta * s1_p * (i1_c + i1_p) / n1 - d * s1_p - converted_s1
    de1_p = (ddelta_1_p / max_delta) * beta * s1_p * (i1_c + i1_p) / n1 - sigma * e1_p - d * e1_p - converted_e1
    di1_p = sigma * e1_p - gamma * i1_p - d * i1_p - converted_i1
    dr1_p = (1 - fatality_rate_p) * gamma * i1_p - d * r1_p - converted_r1
    dd1_p = fatality_rate_p * gamma * i1_p
    da1_p = d * n1_p

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = b * n2_c - (ddelta_2_c / max_delta) * beta * s2_c * (i2_c + i2_p) / n2 - d * s2_c + converted_s2
    de2_c = (ddelta_2_c / max_delta) * beta * s2_c * (i2_c + i2_p) / n2 - sigma * e2_c - d * e2_c + converted_e2
    di2_c = sigma * e2_c - gamma * i2_c - d * i2_c + converted_i2
    dr2_c = (1 - fatality_rate_c) * gamma * i2_c - d * r2_c + converted_r2
    dd2_c = fatality_rate_c * gamma * i2_c
    da2_c = d * n2_c

    # Pagan compartments in Zone 2
    ds2_p = b * n2_p - (ddelta_2_p / max_delta) * beta * s2_p * (i2_c + i2_p) / n2 - d * s2_p - converted_s2
    de2_p = (ddelta_2_p / max_delta) * beta * s2_p * (i2_c + i2_p) / n2 - sigma * e2_p - d * e2_p - converted_e2
    di2_p = sigma * e2_p - gamma * i2_p - d * i2_p - converted_i2
    dr2_p = (1 - fatality_rate_p) * gamma * i2_p - d * r2_p - converted_r2
    dd2_p = fatality_rate_p * gamma * i2_p
    da2_p = d * n2_p

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = b * n3_c - (ddelta_3_c / max_delta) * beta * s3_c * (i3_c + i3_p) / n3 - d * s3_c + converted_s3
    de3_c = (ddelta_3_c / max_delta) * beta * s3_c * (i3_c + i3_p) / n3 - sigma * e3_c - d * e3_c + converted_e3
    di3_c = sigma * e3_c - gamma * i3_c - d * i3_c + converted_i3
    dr3_c = (1 - fatality_rate_c) * gamma * i3_c - d * r3_c + converted_r3
    dd3_c = fatality_rate_c * gamma * i3_c
    da3_c = d * n3_c

    # Pagan compartments in Zone 3
    ds3_p = b * n3_p - (ddelta_3_p / max_delta) * beta * s3_p * (i3_c + i3_p) / n3 - d * s3_p - converted_s3
    de3_p = (ddelta_3_p / max_delta) * beta * s3_p * (i3_c + i3_p) / n3 - sigma * e3_p - d * e3_p - converted_e3
    di3_p = sigma * e3_p - gamma * i3_p - d * i3_p - converted_i3
    dr3_p = (1 - fatality_rate_p) * gamma * i3_p - d * r3_p - converted_r3
    dd3_p = fatality_rate_p * gamma * i3_p
    da3_p = d * n3_p

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = b * n4_c - (ddelta_4_c / max_delta) * beta * s4_c * (i4_c + i4_p) / n4 - d * s4_c + converted_s4
    de4_c = (ddelta_4_c / max_delta) * beta * s4_c * (i4_c + i4_p) / n4 - sigma * e4_c - d * e4_c + converted_e4
    di4_c = sigma * e4_c - gamma * i4_c - d * i4_c + converted_i4
    dr4_c = (1 - fatality_rate_c) * gamma * i4_c - d * r4_c + converted_r4
    dd4_c = fatality_rate_c * gamma * i4_c
    da4_c = d * n4_c

    # Pagan compartments in Zone 4
    ds4_p = b * n4_p - (ddelta_4_p / max_delta) * beta * s4_p * (i4_c + i4_p) / n4 - d * s4_p - converted_s4
    de4_p = (ddelta_4_p / max_delta) * beta * s4_p * (i4_c + i4_p) / n4 - sigma * e4_p - d * e4_p - converted_e4
    di4_p = sigma * e4_p - gamma * i4_p - d * i4_p - converted_i4
    dr4_p = (1 - fatality_rate_p) * gamma * i4_p - d * r4_p - converted_r4
    dd4_p = fatality_rate_p * gamma * i4_p
    da4_p = d * n4_p

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p,
        ddelta_1_c, ddelta_1_p, ddelta_2_c, ddelta_2_p, ddelta_3_c, ddelta_3_p, ddelta_4_c, ddelta_4_p
    ]


def simple_demographic_model_subpopulation_pairs_with_conversion_to_dense_zones(y, t, parameters):
    """
    Extension of def simple_demographic_model_with_conversion by four zones.
    """

    (
        s1_c, e1_c, i1_c, r1_c, d1_c, a1_c, s1_p, e1_p, i1_p, r1_p, d1_p, a1_p,
        s2_c, e2_c, i2_c, r2_c, d2_c, a2_c, s2_p, e2_p, i2_p, r2_p, d2_p, a2_p,
        s3_c, e3_c, i3_c, r3_c, d3_c, a3_c, s3_p, e3_p, i3_p, r3_p, d3_p, a3_p,
        s4_c, e4_c, i4_c, r4_c, d4_c, a4_c, s4_p, e4_p, i4_p, r4_p, d4_p, a4_p
    ) = y
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily

    # Totals of subpopulations in each zone
    n1_c = s1_c + e1_c + i1_c + r1_c
    n1_p = s1_p + e1_p + i1_p + r1_p

    n2_c = s2_c + e2_c + i2_c + r2_c
    n2_p = s2_p + e2_p + i2_p + r2_p

    n3_c = s3_c + e3_c + i3_c + r3_c
    n3_p = s3_p + e3_p + i3_p + r3_p

    n4_c = s4_c + e4_c + i4_c + r4_c
    n4_p = s4_p + e4_p + i4_p + r4_p

    # Pagans converted to Christianity for all living compartments in each zone
    converted_s1 = min(conversion_rate * s1_p, s1_p)
    converted_e1 = min(conversion_rate * e1_p, e1_p)
    converted_i1 = min(conversion_rate * i1_p, i1_p)
    converted_r1 = min(conversion_rate * r1_p, r1_p)

    converted_s2 = min(conversion_rate * s2_p, s2_p)
    converted_e2 = min(conversion_rate * e2_p, e2_p)
    converted_i2 = min(conversion_rate * i2_p, i2_p)
    converted_r2 = min(conversion_rate * r2_p, r2_p)

    converted_s3 = min(conversion_rate * s3_p, s3_p)
    converted_e3 = min(conversion_rate * e3_p, e3_p)
    converted_i3 = min(conversion_rate * i3_p, i3_p)
    converted_r3 = min(conversion_rate * r3_p, r3_p)

    converted_s4 = min(conversion_rate * s4_p, s4_p)
    converted_e4 = min(conversion_rate * e4_p, e4_p)
    converted_i4 = min(conversion_rate * i4_p, i4_p)
    converted_r4 = min(conversion_rate * r4_p, r4_p)

    def non_negative(value):
        # return max(0, value)
        return value

    # Zone 1
    # Christian compartments in Zone 1
    ds1_c = non_negative(b * n1_c - d * s1_c + converted_s1 + converted_s2)
    de1_c = non_negative(-d * e1_c + converted_e1)
    di1_c = non_negative(-d * i1_c + converted_i1)
    dr1_c = non_negative(-d * r1_c + converted_r1 + converted_s2)
    dd1_c = 0
    da1_c = non_negative(d * n1_c)

    # Pagan compartments in Zone 1
    ds1_p = non_negative(b * n1_p - d * s1_p - converted_s1)
    de1_p = non_negative(-d * e1_p - converted_e1)
    di1_p = non_negative(-d * i1_p - converted_i1)
    dr1_p = non_negative(-d * r1_p - converted_r1)
    dd1_p = 0
    da1_p = non_negative(d * n1_p)

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = non_negative(b * n2_c - d * s2_c + converted_s3)
    de2_c = non_negative(-d * e2_c + converted_e2)
    di2_c = non_negative(-d * i2_c + converted_i2)
    dr2_c = non_negative(-d * r2_c + converted_r3)
    dd2_c = 0
    da2_c = non_negative(d * n2_c)

    # Pagan compartments in Zone 2
    ds2_p = non_negative(b * n2_p - d * s2_p - converted_s2)
    de2_p = non_negative(-d * e2_p - converted_e2)
    di2_p = non_negative(-d * i2_p - converted_i2)
    dr2_p = non_negative(-d * r2_p - converted_r2)
    dd2_p = 0
    da2_p = non_negative(d * n2_p)

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = non_negative(b * n3_c - d * s3_c + converted_s4)
    de3_c = non_negative(-d * e3_c + converted_e3)
    di3_c = non_negative(-d * i3_c + converted_i3)
    dr3_c = non_negative(-d * r3_c + converted_r4)
    dd3_c = 0
    da3_c = non_negative(d * n3_c)

    # Pagan compartments in Zone 3
    ds3_p = non_negative(b * n3_p - d * s3_p - converted_s3)
    de3_p = non_negative(-d * e3_p - converted_e3)
    di3_p = non_negative(-d * i3_p - converted_i3)
    dr3_p = non_negative(-d * r3_p - converted_r3)
    dd3_p = 0
    da3_p = non_negative(d * n3_p)

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = non_negative(b * n4_c - d * s4_c + converted_s4)
    de4_c = non_negative(-d * e4_c + converted_e4)
    di4_c = non_negative(-d * i4_c + converted_i4)
    dr4_c = non_negative(-d * r4_c + converted_r4)
    dd4_c = 0
    da4_c = non_negative(d * n4_c)

    # Pagan compartments in Zone 4
    ds4_p = non_negative(b * n4_p - d * s4_p - converted_s4)
    de4_p = non_negative(-d * e4_p - converted_e4)
    di4_p = non_negative(-d * i4_p - converted_i4)
    dr4_p = non_negative(-d * r4_p - converted_r4)
    dd4_p = 0
    da4_p = non_negative(d * n4_p)

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p
    ]


def direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_converts_to_dense_zones_seird_model(y, t, parameters):
    """
    An extenstion of the following model:
    direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model

    In this model the Christians and Pagans still form subpopulations in 4 zones, but converstion
    if Pagans to Christians now happens from the lower density zones to the higher density zones.
    This means that individuals convert from zone 4 to 3, 3 to 2, 2 to 1, and 1 to 1, and in doing so
    represent the idea that Christianity was an urban phenomenon, happening towards areas with more
    towns and cities.
    """
    (
        s1_c, e1_c, i1_c, r1_c, d1_c, a1_c, s1_p, e1_p, i1_p, r1_p, d1_p, a1_p,
        s2_c, e2_c, i2_c, r2_c, d2_c, a2_c, s2_p, e2_p, i2_p, r2_p, d2_p, a2_p,
        s3_c, e3_c, i3_c, r3_c, d3_c, a3_c, s3_p, e3_p, i3_p, r3_p, d3_p, a3_p,
        s4_c, e4_c, i4_c, r4_c, d4_c, a4_c, s4_p, e4_p, i4_p, r4_p, d4_p, a4_p,
        delta_1_c, delta_1_p, delta_2_c, delta_2_p, delta_3_c, delta_3_p, delta_4_c, delta_4_p
    ) = y

    beta = parameters.beta
    sigma = parameters.sigma
    gamma = parameters.gamma
    unified_deltas = parameters.unified_deltas
    fatality_rate_p = parameters.fatality_rate_p
    fatality_rate_c = parameters.fatality_rate_c
    b = parameters.natural_birth_rate
    d = parameters.natural_death_rate
    conversion_rate = parameters.conversion_rate_daily
    max_delta = parameters.max_delta

    epsilon = 1e-4  # Small value to prevent division by zero

    if unified_deltas:
        ddelta_1_c = adjust_delta(
            delta_prev=delta_1_c, delta_0=zeleners_initial_delta_1, s_n=s1_c + s1_p, i_n=i1_c + i1_p, r_n=r1_c + r1_p
        )
        ddelta_1_p = ddelta_1_c

        ddelta_2_c = adjust_delta(
            delta_prev=delta_2_c, delta_0=zeleners_initial_delta_2, s_n=s2_c + s2_p, i_n=i2_c + i2_p, r_n=r2_c + r2_p
        )
        ddelta_2_p = ddelta_2_c

        ddelta_3_c = adjust_delta(
            delta_prev=delta_3_c, delta_0=zeleners_initial_delta_3, s_n=s3_c + s3_p, i_n=i3_c + i3_p, r_n=r3_c + r3_p
        )
        ddelta_3_p = ddelta_3_c

        ddelta_4_c = adjust_delta(
            delta_prev=delta_4_c, delta_0=zeleners_initial_delta_4, s_n=s4_c + s4_p, i_n=i4_c + i4_p, r_n=r4_c + r4_p
        )
        ddelta_4_p = ddelta_4_c

    else:
        ddelta_1_c = adjust_delta(
            delta_prev=delta_1_c, delta_0=zeleners_initial_delta_1, s_n=s1_c, i_n=i1_c, r_n=r1_c
        )
        ddelta_1_p = adjust_delta(
            delta_prev=delta_1_p, delta_0=zeleners_initial_delta_1, s_n=s1_p, i_n=i1_p, r_n=r1_p
        )
        ddelta_2_c = adjust_delta(
            delta_prev=delta_2_c, delta_0=zeleners_initial_delta_2, s_n=s2_c, i_n=i2_c, r_n=r2_c
        )
        ddelta_2_p = adjust_delta(
            delta_prev=delta_2_p, delta_0=zeleners_initial_delta_2, s_n=s2_p, i_n=i2_p, r_n=r2_p
        )
        ddelta_3_c = adjust_delta(
            delta_prev=delta_3_c, delta_0=zeleners_initial_delta_3, s_n=s3_c, i_n=i3_c, r_n=r3_c
        )
        ddelta_3_p = adjust_delta(
            delta_prev=delta_3_p, delta_0=zeleners_initial_delta_3, s_n=s3_p, i_n=i3_p, r_n=r3_p
        )
        ddelta_4_c = adjust_delta(
            delta_prev=delta_4_c, delta_0=zeleners_initial_delta_4, s_n=s4_c, i_n=i4_c, r_n=r4_c
        )
        ddelta_4_p = adjust_delta(
            delta_prev=delta_4_p, delta_0=zeleners_initial_delta_4, s_n=s4_p, i_n=i4_p, r_n=r4_p
        )

    # Totals of subpopulations in each zone and whole population
    n1_c = max(s1_c + e1_c + i1_c + r1_c, epsilon)
    n1_p = max(s1_p + e1_p + i1_p + r1_p, epsilon)
    n1 = max(n1_c + n1_p, epsilon)

    n2_c = max(s2_c + e2_c + i2_c + r2_c, epsilon)
    n2_p = max(s2_p + e2_p + i2_p + r2_p, epsilon)
    n2 = max(n2_c + n2_p, epsilon)

    n3_c = max(s3_c + e3_c + i3_c + r3_c, epsilon)
    n3_p = max(s3_p + e3_p + i3_p + r3_p, epsilon)
    n3 = max(n3_c + n3_p, epsilon)

    n4_c = max(s4_c + e4_c + i4_c + r4_c, epsilon)
    n4_p = max(s4_p + e4_p + i4_p + r4_p, epsilon)
    n4 = max(n4_c + n4_p, epsilon)

    # Pagans converted to Christianity for all living compartments in each zone
    converted_s1 = min(conversion_rate * s1_p, s1_p)
    converted_e1 = min(conversion_rate * e1_p, e1_p)
    converted_i1 = min(conversion_rate * i1_p, i1_p)
    converted_r1 = min(conversion_rate * r1_p, r1_p)

    converted_s2 = min(conversion_rate * s2_p, s2_p)
    converted_e2 = min(conversion_rate * e2_p, e2_p)
    converted_i2 = min(conversion_rate * i2_p, i2_p)
    converted_r2 = min(conversion_rate * r2_p, r2_p)

    converted_s3 = min(conversion_rate * s3_p, s3_p)
    converted_e3 = min(conversion_rate * e3_p, e3_p)
    converted_i3 = min(conversion_rate * i3_p, i3_p)
    converted_r3 = min(conversion_rate * r3_p, r3_p)

    converted_s4 = min(conversion_rate * s4_p, s4_p)
    converted_e4 = min(conversion_rate * e4_p, e4_p)
    converted_i4 = min(conversion_rate * i4_p, i4_p)
    converted_r4 = min(conversion_rate * r4_p, r4_p)

    # Zone 1
    # Christian compartments in Zone 1
    ds1_c = b * n1_c - (ddelta_1_c / max_delta) * beta * s1_c * (i1_c + i1_p) / n1 - d * s1_c + converted_s1 + converted_s2
    de1_c = (ddelta_1_c / max_delta) * beta * s1_c * (i1_c + i1_p) / n1 - sigma * e1_c - d * e1_c + converted_e1
    di1_c = sigma * e1_c - gamma * i1_c - d * i1_c + converted_i1
    dr1_c = (1 - fatality_rate_c) * gamma * i1_c - d * r1_c + converted_r1 + converted_r2
    dd1_c = fatality_rate_c * gamma * i1_c
    da1_c = d * n1_c

    # Pagan compartments in Zone 1
    ds1_p = b * n1_p - (ddelta_1_p / max_delta) * beta * s1_p * (i1_c + i1_p) / n1 - d * s1_p - converted_s1
    de1_p = (ddelta_1_p / max_delta) * beta * s1_p * (i1_c + i1_p) / n1 - sigma * e1_p - d * e1_p - converted_e1
    di1_p = sigma * e1_p - gamma * i1_p - d * i1_p - converted_i1
    dr1_p = (1 - fatality_rate_p) * gamma * i1_p - d * r1_p - converted_r1
    dd1_p = fatality_rate_p * gamma * i1_p
    da1_p = d * n1_p

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = b * n2_c - (ddelta_2_c / max_delta) * beta * s2_c * (i2_c + i2_p) / n2 - d * s2_c + converted_s3
    de2_c = (ddelta_2_c / max_delta) * beta * s2_c * (i2_c + i2_p) / n2 - sigma * e2_c - d * e2_c + converted_e2
    di2_c = sigma * e2_c - gamma * i2_c - d * i2_c + converted_i2
    dr2_c = (1 - fatality_rate_c) * gamma * i2_c - d * r2_c + converted_r3
    dd2_c = fatality_rate_c * gamma * i2_c
    da2_c = d * n2_c

    # Pagan compartments in Zone 2
    ds2_p = b * n2_p - (ddelta_2_p / max_delta) * beta * s2_p * (i2_c + i2_p) / n2 - d * s2_p - converted_s2
    de2_p = (ddelta_2_p / max_delta) * beta * s2_p * (i2_c + i2_p) / n2 - sigma * e2_p - d * e2_p - converted_e2
    di2_p = sigma * e2_p - gamma * i2_p - d * i2_p - converted_i2
    dr2_p = (1 - fatality_rate_p) * gamma * i2_p - d * r2_p - converted_r2
    dd2_p = fatality_rate_p * gamma * i2_p
    da2_p = d * n2_p

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = b * n3_c - (ddelta_3_c / max_delta) * beta * s3_c * (i3_c + i3_p) / n3 - d * s3_c + converted_s4
    de3_c = (ddelta_3_c / max_delta) * beta * s3_c * (i3_c + i3_p) / n3 - sigma * e3_c - d * e3_c + converted_e3
    di3_c = sigma * e3_c - gamma * i3_c - d * i3_c + converted_i3
    dr3_c = (1 - fatality_rate_c) * gamma * i3_c - d * r3_c + converted_r4
    dd3_c = fatality_rate_c * gamma * i3_c
    da3_c = d * n3_c

    # Pagan compartments in Zone 3
    ds3_p = b * n3_p - (ddelta_3_p / max_delta) * beta * s3_p * (i3_c + i3_p) / n3 - d * s3_p - converted_s3
    de3_p = (ddelta_3_p / max_delta) * beta * s3_p * (i3_c + i3_p) / n3 - sigma * e3_p - d * e3_p - converted_e3
    di3_p = sigma * e3_p - gamma * i3_p - d * i3_p - converted_i3
    dr3_p = (1 - fatality_rate_p) * gamma * i3_p - d * r3_p - converted_r3
    dd3_p = fatality_rate_p * gamma * i3_p
    da3_p = d * n3_p

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = b * n4_c - (ddelta_4_c / max_delta) * beta * s4_c * (i4_c + i4_p) / n4 - d * s4_c
    de4_c = (ddelta_4_c / max_delta) * beta * s4_c * (i4_c + i4_p) / n4 - sigma * e4_c - d * e4_c + converted_e4
    di4_c = sigma * e4_c - gamma * i4_c - d * i4_c + converted_i4
    dr4_c = (1 - fatality_rate_c) * gamma * i4_c - d * r4_c
    dd4_c = fatality_rate_c * gamma * i4_c
    da4_c = d * n4_c

    # Pagan compartments in Zone 4
    ds4_p = b * n4_p - (ddelta_4_p / max_delta) * beta * s4_p * (i4_c + i4_p) / n4 - d * s4_p - converted_s4
    de4_p = (ddelta_4_p / max_delta) * beta * s4_p * (i4_c + i4_p) / n4 - sigma * e4_p - d * e4_p - converted_e4
    di4_p = sigma * e4_p - gamma * i4_p - d * i4_p - converted_i4
    dr4_p = (1 - fatality_rate_p) * gamma * i4_p - d * r4_p - converted_r4
    dd4_p = fatality_rate_p * gamma * i4_p
    da4_p = d * n4_p

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p,
        ddelta_1_c, ddelta_1_p, ddelta_2_c, ddelta_2_p, ddelta_3_c, ddelta_3_p, ddelta_4_c, ddelta_4_p
    ]



def plot_seir_model(
    solution,
    t,
    start_year,
    end_year,
    compartment_indices,
    compartment_labels,
    every_nth_year=5,  # TODO: rename to something more label-indicating
    y_tick_interval=100_000,
    display_y_label_every_n_ticks=1,
    x_label="Time (years)",
    y_label="Number of Individuals",
    plot_title="SEIR Model Over Time",
):
    """
    Plots the compartments of an SEIR model given the solution object, time array,
    and the start and end year for x-axis labeling.

    Parameters:
    - solution: The solution object returned by scipy.integrate.solve_ivp.
    - t: The time points at which the solution was evaluated.
    - start_year: The starting year of the simulation for labeling the x-axis.
    - end_year: The ending year of the simulation for labeling the x-axis.
    - compartment_indices: A list of indices for the compartments to be plotted.
    - compartment_labels: A list of labels corresponding to the compartments.
    - every_nth_year: Interval for labeling years on the x-axis, defaults to 5.
    - y_tick_interval: Interval between y-axis ticks.
    - display_y_label_every_n_ticks: Controls which y-axis labels are displayed.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - plot_title: Title of the plot.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the specified compartments
    for index, label in zip(compartment_indices, compartment_labels):
        ax.plot(solution.t, solution.y[index, :], label=label)

    # Set up the x-axis with year ticks and labels
    # total_days = (end_year - start_year + 1) * 365
    # year_tick_positions = np.arange(0, total_days, 365)
    # year_tick_labels = np.arange(start_year, end_year + 1)
    # ax.set_xticks(year_tick_positions)
    # ax.set_xticklabels(year_tick_labels, rotation=45)
    total_days = (end_year - start_year + 1) * 365
    year_tick_positions = np.arange(0, total_days, 365)
    year_tick_labels = np.arange(start_year, end_year + 1)
    ax.set_xticks(year_tick_positions)
    ax.set_xticklabels(year_tick_labels, rotation=45)

    # Optionally show every nth year label to avoid clutter
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % every_nth_year != 0:
            label.set_visible(False)

    # Set up the y-axis with custom ticks based on the maximum value across specified compartments
    y_max = min(np.max([np.max(solution.y[i, :]) for i in compartment_indices]), 1e8)
    print(f"y_max: {y_max}")
    y_ticks = np.arange(0, y_max + y_tick_interval, y_tick_interval)
    y_labels = [
        str(int(y)) if index % display_y_label_every_n_ticks == 0 else ""
        for index, y in enumerate(y_ticks)
    ]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Set axis' labels and plot title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.legend()

    plt.tight_layout()
    plt.show()


def proof_of_concept_solve_and_plot_ap_as_smallpox_in_rome(
    start_year=165, end_year=189
):
    """
    Solve and plot the case of my PLOS ONE article as a proof of concept (showcase this approach works).
    """
    # Initial conditions (just the population of the city of Rome and 1 infected - no other people,
    # and not distinguishing Christians and Pagans.
    y0 = [0, 0, 0, 0, 0, 0, 923405, 0, 1, 0, 0, 0]

    # Timeframe of simulation
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the ODE
    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_one_population_as_in_plos_paper(
            y, t, default_seir_params
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )

    # Solution indices and labels relevant only to the Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [6, 7, 8, 9, 10]
    compartment_labels = [
        "Susceptible Pagans",
        "Exposed Pagans",
        "Infected Pagans",
        "Recovered Pagans",
        "Deceased Pagans",
    ]
    plot_seir_model(
        solution,
        t,
        start_year=start_year,
        end_year=end_year,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
    )


def proof_of_concept_solve_and_plot_ap_as_smallpox_over_two_subpopulations_in_empire(
    start_year=165,
    end_year=189,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # Initial conditions (Christian and Pagan populations defined in src.parameters.params)
    y0 = [
        initial_christian_population - 1,
        0,
        1,
        0,
        0,
        0,
        initial_pagan_population - 1,
        0,
        1,
        0,
        0,
        0,
    ]

    # Timeframe of simulation
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the ODE
    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_two_connected_subpopulations_seird_model(
            y, t, default_seir_params
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    compartment_labels = [
        "Susceptible Christians",
        "Exposed Christians",
        "Infected Christians",
        "Recovered Christians",
        "Deceased Christians",
        "Susceptible Pagans",
        "Exposed Pagans",
        "Infected Pagans",
        "Recovered Pagans",
        "Deceased Pagans",
    ]
    plot_seir_model(
        solution,
        t,
        start_year=start_year,
        end_year=end_year,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=1_000_000,
        display_y_label_every_n_ticks=10,
        plot_title="Antonine Plague as smallpox over two subpopulations in the whole Empire",
    )


def proof_of_concept_solve_and_plot_basic_demographic_development_after_ap(
    solution, start_year=190, end_year=248, parameters=default_seir_params
):
    """
    Continues the simulation with a simple demographic model
    using the final state from a previous model as the initial condition.

    Parameters:
    - solution: The solution object from the previous model.
    - start_year: The starting year for the demographic model simulation.
    - end_year: The ending year for the demographic model simulation.
    - parameters: The parameters object for the demographic model.
    """
    # Initial conditions for the demographic model are the final state from the previous solution
    # y0 = solution.y[:, -1]
    compartments = solution.y[:, -1]
    y0 = [max(0, compartment) for compartment in compartments]

    # Timeframe of simulation for the demographic model
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model(y, t, parameters)

    solution = solve_ivp(
        wrapper_for_solve_ivp_demographic, [0, total_days], y0, method="BDF", t_eval=t
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    # compartment_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    compartment_indices = [0, 6]
    compartment_labels = [
        "Susceptible Christians",
        # 'Exposed Christians',
        # 'Infected Christians',
        # 'Recovered Christians',
        # 'Deceased Christians',
        "Susceptible Pagans",
        # 'Exposed Pagans',
        # 'Infected Pagans',
        # 'Recovered Pagans',
        # 'Deceased Pagans'
    ]
    plot_seir_model(
        solution,
        t,
        start_year=start_year,
        end_year=end_year,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="Demographic development after the Antonine Plague over two subpopulations in the whole Empire",
    )


def proof_of_concept_solve_and_plot_ap_as_smallpox_over_two_subpopulations_in_empire_and_continue_with_demographic_dev(
    start_year=165,
    end_year=189,
    demographic_end_year=248,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # Initial conditions (Christian and Pagan populations defined in src.parameters.params)
    y0 = [
        initial_christian_population - 1,
        0,
        1,
        0,
        0,
        0,
        initial_pagan_population - 1,
        0,
        1,
        0,
        0,
        0,
    ]

    # Timeframe of simulation
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the ODE for the Antonine Plague
    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_two_connected_subpopulations_seird_model(
            y, t, default_seir_params
        )

    solution_ap = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )

    proof_of_concept_solve_and_plot_basic_demographic_development_after_ap(
        solution_ap, end_year + 1, demographic_end_year, default_seir_params
    )


def proof_of_concept_solve_and_plot_ap_as_smallpox_over_two_subpopulations_with_two_cfrs_in_empire(
    start_year=165,
    end_year=189,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # Initial conditions (Christian and Pagan populations defined in src.parameters.params)
    y0 = [
        initial_christian_population - 1,
        0,
        1,
        0,
        0,
        0,
        initial_pagan_population - 1,
        0,
        1,
        0,
        0,
        0,
    ]

    # Timeframe of simulation
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the ODE
    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model(
            y, t, default_two_cfrs_params
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    compartment_labels = [
        "Susceptible Christians",
        "Exposed Christians",
        "Infected Christians",
        "Recovered Christians",
        "Deceased Christians",
        "Susceptible Pagans",
        "Exposed Pagans",
        "Infected Pagans",
        "Recovered Pagans",
        "Deceased Pagans",
    ]
    plot_seir_model(
        solution,
        t,
        start_year=start_year,
        end_year=end_year,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=1_000_000,
        display_y_label_every_n_ticks=10,
        plot_title="Antonine Plague as smallpox with two CFRs over two subpopulations in the whole Empire",
    )


def proof_of_concept_solve_and_plot_ap_demography_and_cp_with_two_subpopulations_and_smaller_cfrs_for_christians(
    start_year_ap=165,
    end_year_ap=189,
    end_year_demographic=248,
    end_year_cp=270,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # The Antonine Plague part:
    # Initial conditions (Christian and Pagan populations defined in src.parameters.params)
    y0_ap = [
        initial_christian_population - 1,
        0,
        1,
        0,
        0,
        0,
        initial_pagan_population - 1,
        0,
        1,
        0,
        0,
        0,
    ]

    # Timeframe of simulation
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    t_ap = np.arange(0, total_days_ap)

    # Solve the ODE for the Antonine Plague
    def wrapper_for_solve_ivp_ap(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model(
            y, t, default_two_cfrs_params
        )

    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="BDF", t_eval=t_ap
    )

    # The demographic-development-without-a_specific-disease part:
    compartments_demographic = solution_ap.y[:, -1]
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    t_demographic = np.arange(0, total_days_demographic)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model(y, t, default_seir_params)

    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="BDF",
        t_eval=t_demographic,
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]

    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]

    # Move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Reshuffling for Christians
    susceptible_christians_sum = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum
    y0_cp[1] = 0  # Exposed Christians
    y0_cp[2] = 1  # Infected Christians
    y0_cp[3] = 0  # Recovered Christians
    # y0_cp[4] = 0  # Deceased Christians (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum
    y0_cp[7] = 0  # Exposed Pagans
    y0_cp[8] = 1  # Infected Pagans
    y0_cp[9] = 0  # Recovered Pagans
    # y0_cp[10] = 0  # Deceased Pagans (set to 0 to see what CP does)

    # Timeframe of simulation for the demographic model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    t_cp = np.arange(0, total_days_cp)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_seird_model(
            y, t, measles_seir_params
        )

    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="BDF",
        t_eval=t_cp,
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    compartment_labels = [
        "Susceptible Christians",
        "Exposed Christians",
        "Infected Christians",
        "Recovered Christians",
        "Deceased Christians",
        "Susceptible Pagans",
        "Exposed Pagans",
        "Infected Pagans",
        "Recovered Pagans",
        "Deceased Pagans",
    ]
    plot_seir_model(
        solution_cp,
        t_cp,
        start_year=end_year_demographic,
        end_year=end_year_cp,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="CP as measles with two CFRs over two subpopulations in the whole Empire (after AP and demo)",
    )


def proof_of_concept_solve_and_plot_ap_only_demographic_development_with_conversion_in_empire(
    start_year=165,
    end_year=270,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # Initial conditions (Christian and Pagan populations defined in src.parameters.params)
    y0 = [
        initial_christian_population,
        0,
        0,
        0,
        0,
        0,
        initial_pagan_population,
        0,
        0,
        0,
        0,
        0,
    ]

    # Timeframe of simulation
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the ODE for the Antonine Plague
    def wrapper_for_solve_ivp(t, y):
        return simple_demographic_model_with_conversion(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    compartment_labels = [
        "Susceptible Christians",
        "Exposed Christians",
        "Infected Christians",
        "Recovered Christians",
        "Deceased Christians",
        "Susceptible Pagans",
        "Exposed Pagans",
        "Infected Pagans",
        "Recovered Pagans",
        "Deceased Pagans",
    ]
    plot_seir_model(
        solution,
        t,
        start_year=start_year,
        end_year=end_year,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=1_000_000,
        display_y_label_every_n_ticks=10,
        plot_title="Antonine Plague as no disease but instead conversion from P to C in the whole Empire",
    )


def proof_of_concept_solve_and_plot_ap_with_conversion_and_smaller_cfr_for_christians(
    start_year=165,
    end_year=189,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # Initial conditions (Christian and Pagan populations defined in src.parameters.params)
    y0 = [
        initial_christian_population - 1,
        0,
        1,
        0,
        0,
        0,
        initial_pagan_population - 1,
        0,
        1,
        0,
        0,
        0,
    ]

    # Timeframe of simulation
    total_days = (end_year - start_year + 1) * 365
    t = np.arange(0, total_days)

    # Solve the ODE
    def wrapper_for_solve_ivp(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_and_conversion_seird_model(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    solution = solve_ivp(
        wrapper_for_solve_ivp, [0, total_days], y0, method="BDF", t_eval=t
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    compartment_labels = [
        "Susceptible Christians",
        "Exposed Christians",
        "Infected Christians",
        "Recovered Christians",
        "Deceased Christians",
        "Susceptible Pagans",
        "Exposed Pagans",
        "Infected Pagans",
        "Recovered Pagans",
        "Deceased Pagans",
    ]
    plot_seir_model(
        solution,
        t,
        start_year=start_year,
        end_year=end_year,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=1_000_000,
        display_y_label_every_n_ticks=10,
        plot_title="Antonine Plague as smallpox with smaller CFR for Christians and conversion in whole Empire",
    )


def proof_of_concept_solve_and_plot_ap_demo_cp_with_conversion_and_smaller_cfr_for_christians(
    start_year_ap=165,
    end_year_ap=189,
    end_year_demographic=248,
    end_year_cp=270,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
    # The Antonine Plague part:
    y0_ap = [
        initial_christian_population - 1,
        0,
        1,
        0,
        0,
        0,
        initial_pagan_population - 1,
        0,
        1,
        0,
        0,
        0,
    ]

    # Timeframe of simulation
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    t_ap = np.arange(0, total_days_ap)

    # Solve the ODE for the Antonine Plague
    def wrapper_for_solve_ivp_ap(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_and_conversion_seird_model(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="BDF", t_eval=t_ap
    )

    # The demographic-development-without-a_specific-disease part:
    compartments_demographic = solution_ap.y[:, -1]
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    t_demographic = np.arange(0, total_days_demographic)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model_with_conversion(y, t, smallpox_seir_params_with_starks_conversion)

    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="BDF",
        t_eval=t_demographic,
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]

    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]

    # Move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Reshuffling for Christians
    susceptible_christians_sum = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum
    y0_cp[1] = 0  # Exposed Christians
    y0_cp[2] = 1  # Infected Christians
    y0_cp[3] = 0  # Recovered Christians
    # y0_cp[4] = 0  # Deceased Christians (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum
    y0_cp[7] = 0  # Exposed Pagans
    y0_cp[8] = 1  # Infected Pagans
    y0_cp[9] = 0  # Recovered Pagans
    # y0_cp[10] = 0  # Deceased Pagans (set to 0 to see what CP does)

    # Timeframe of simulation for the demographic model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    t_cp = np.arange(0, total_days_cp)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_over_two_connected_subpopulations_with_two_cfrs_and_conversion_seird_model(
            y, t, measles_seir_params_with_lower_cfr_for_c_and_starks_conversion
        )

    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="BDF",
        t_eval=t_cp,
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    # compartment_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    compartment_indices = [0, 1, 2, 3, 6, 7, 8, 9]
    compartment_labels = [
        "Susceptible Christians",
        "Exposed Christians",
        "Infected Christians",
        "Recovered Christians",
        # "Deceased Christians",
        "Susceptible Pagans",
        "Exposed Pagans",
        "Infected Pagans",
        "Recovered Pagans",
        # "Deceased Pagans",
    ]
    plot_seir_model(
        solution_cp,
        t_cp,
        start_year=end_year_demographic,
        end_year=end_year_cp,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="CP as measles with smaller CFR for C and conversion in the whole Empire (after AP and demo)",
    )

    # Print the final values of each compartment for debugging purposes
    compartments_after_conversion = solution_cp.y[:, -1]

    s_c = compartments_after_conversion[0]
    e_c = compartments_after_conversion[1]
    i_c = compartments_after_conversion[2]
    r_c = compartments_after_conversion[3]
    d_c = compartments_after_conversion[4]
    a_c = compartments_after_conversion[5]
    s_p = compartments_after_conversion[6]
    e_p = compartments_after_conversion[7]
    i_p = compartments_after_conversion[8]
    r_p = compartments_after_conversion[9]
    d_p = compartments_after_conversion[10]
    a_p = compartments_after_conversion[11]
    alive_c = s_c + e_c + i_c + r_c
    alive_p = s_c + e_p + i_p + r_p

    print(
        f"s_c = {s_c}\n"
        f"e_c = {e_c}\n"
        f"i_c = {i_c}\n"
        f"r_c = {r_c}\n"
        f"d_c = {d_c}\n"
        f"a_c = {a_c}\n"
        f"s_p = {s_p}\n"
        f"e_p = {e_p}\n"
        f"i_p = {i_p}\n"
        f"r_p = {r_p}\n"
        f"d_p = {d_p}\n"
        f"a_p = {a_p}\n"
        f"alive_c = {alive_c}\n"
        f"alive_p = {alive_p}"
    )


def proof_of_concept_solve_and_plot_ap_demo_cp_with_conversion_and_smaller_cfr_for_christians_in_four_separate_zones(
    start_year_ap=165,
    end_year_ap=189,
    end_year_demographic=248,
    end_year_cp=270,
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
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
    ]

    # Timeframe of simulation
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    t_ap = np.arange(0, total_days_ap)

    # Solve the ODE for the Antonine Plague
    def wrapper_for_solve_ivp_ap(t, y):
        return (
            direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_conversion_in_pairs_seird_model(
                y, t, smallpox_seir_params_with_starks_conversion
            )
        )

    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="BDF", t_eval=t_ap
    )

    # The demographic-development-without-a_specific-disease part:
    compartments_demographic = solution_ap.y[:, -1]
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    t_demographic = np.arange(0, total_days_demographic)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model_subpopulation_pairs_with_conversion_in_four_zones(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="BDF",
        t_eval=t_demographic,
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]

    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]

    # In each zone move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Zone 1
    # Reshuffling for Christians
    susceptible_christians_sum_zone_1 = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum_zone_1
    y0_cp[1] = 0  # Exposed Christians in Zone 1
    y0_cp[2] = 1  # Infected Christians in Zone 1
    y0_cp[3] = 0  # Recovered Christians in Zone 1
    # y0_cp[4] = 0  # Deceased Christians  in Zone 1 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_1 = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum_zone_1
    y0_cp[7] = 0  # Exposed Pagans in Zone 1
    y0_cp[8] = 1  # Infected Pagans in Zone 1
    y0_cp[9] = 0  # Recovered Pagans in Zone 1
    # y0_cp[10] = 0  # Deceased Pagans in Zone 1 (set to 0 to see what CP does)

    # Zone 2
    # Reshuffling for Christians
    susceptible_christians_sum_zone_2 = sum(y0_cp[12:16])
    y0_cp[12] = susceptible_christians_sum_zone_2
    y0_cp[13] = 0  # Exposed Christians in Zone 2
    y0_cp[14] = 1  # Infected Christians in Zone 2
    y0_cp[15] = 0  # Recovered Christians in Zone 2
    # y0_cp[16] = 0  # Deceased Christians  in Zone 2 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_2 = sum(y0_cp[18:22])
    y0_cp[18] = susceptible_pagans_sum_zone_2
    y0_cp[19] = 0  # Exposed Pagans in Zone 2
    y0_cp[20] = 1  # Infected Pagans in Zone 2
    y0_cp[21] = 0  # Recovered Pagans in Zone 2
    # y0_cp[22] = 0  # Deceased Pagans in Zone 2 (set to 0 to see what CP does)

    # Zone 3
    # Reshuffling for Christians
    susceptible_christians_sum_zone_3 = sum(y0_cp[24:28])
    y0_cp[24] = susceptible_christians_sum_zone_3
    y0_cp[25] = 0  # Exposed Christians in Zone 3
    y0_cp[26] = 1  # Infected Christians in Zone 3
    y0_cp[27] = 0  # Recovered Christians in Zone 3
    # y0_cp[28] = 0  # Deceased Christians  in Zone 3 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_3 = sum(y0_cp[30:34])
    y0_cp[30] = susceptible_pagans_sum_zone_3
    y0_cp[31] = 0  # Exposed Pagans in Zone 3
    y0_cp[32] = 1  # Infected Pagans in Zone 3
    y0_cp[33] = 0  # Recovered Pagans in Zone 3
    # y0_cp[34] = 0  # Deceased Pagans in Zone 3 (set to 0 to see what CP does)

    # Zone 4
    # Reshuffling for Christians
    susceptible_christians_sum_zone_4 = sum(y0_cp[36:40])
    y0_cp[36] = susceptible_christians_sum_zone_4
    y0_cp[37] = 0  # Exposed Christians in Zone 4
    y0_cp[38] = 1  # Infected Christians in Zone 4
    y0_cp[39] = 0  # Recovered Christians in Zone 4
    # y0_cp[40] = 0  # Deceased Christians  in Zone 4 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_4 = sum(y0_cp[42:46])
    y0_cp[42] = susceptible_pagans_sum_zone_4
    y0_cp[43] = 0  # Exposed Pagans in Zone 4
    y0_cp[44] = 1  # Infected Pagans in Zone 4
    y0_cp[45] = 0  # Recovered Pagans in Zone 4
    # y0_cp[46] = 0  # Deceased Pagans in Zone 4 (set to 0 to see what CP does)

    # Timeframe of simulation for the demographic model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    t_cp = np.arange(0, total_days_cp)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_conversion_in_pairs_seird_model(
            y, t, measles_seir_params_with_lower_cfr_for_c_and_starks_conversion
        )

    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="BDF",
        t_eval=t_cp,
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [
        0, 1, 2, 3,  # Christians in Zone 1
        6, 7, 8, 9,  # Pagans in Zone 1
        12, 13, 14, 15,  # Christians in Zone 2
        18, 19, 20, 21,  # Pagans in Zone 2
        24, 25, 26, 27,  # Christians in Zone 3
        30, 31, 32, 33,  # Pagans in Zone 3
        36, 37, 38, 39,  # Christians in Zone 4
        42, 43, 44, 45  # Pagans in Zone 4
    ]
    compartment_labels = [
        # Zone 1
        "Susceptible Christians in Zone 1",
        "Exposed Christians in Zone 1",
        "Infected Christians in Zone 1",
        "Recovered Christians in Zone 1",
        # "Deceased Christians in Zone 1",

        "Susceptible Pagans in Zone 1",
        "Exposed Pagans in Zone 1",
        "Infected Pagans in Zone 1",
        "Recovered Pagans in Zone 1",
        # "Deceased Pagans in Zone 1",

        # Zone 2
        "Susceptible Christians in Zone 2",
        "Exposed Christians in Zone 2",
        "Infected Christians in Zone 2",
        "Recovered Christians in Zone§ 2",
        # "Deceased Christians in Zone 2",

        "Susceptible Pagans in Zone 2",
        "Exposed Pagans in Zone 2",
        "Infected Pagans in Zone 2",
        "Recovered Pagans in Zone 2",
        # "Deceased Pagans in Zone 2",

        # Zone 3
        "Susceptible Christians in Zone 3",
        "Exposed Christians in Zone 3",
        "Infected Christians in Zone 3",
        "Recovered Christians in Zone 3",
        # "Deceased Christians in Zone 3",

        "Susceptible Pagans in Zone 3",
        "Exposed Pagans in Zone 3",
        "Infected Pagans in Zone 3",
        "Recovered Pagans in Zone 3",
        # "Deceased Pagans in Zone 3",

        # Zone 4
        "Susceptible Christians in Zone 4",
        "Exposed Christians in Zone 4",
        "Infected Christians in Zone 4",
        "Recovered Christians in Zone 4",
        # "Deceased Christians in Zone 4",

        "Susceptible Pagans in Zone 4",
        "Exposed Pagans in Zone 4",
        "Infected Pagans in Zone 4",
        "Recovered Pagans in Zone 4",
        # "Deceased Pagans in Zone 4"
    ]
    plot_seir_model(
        solution_cp,
        t_cp,
        start_year=end_year_demographic,
        end_year=end_year_cp,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="CP in 4 separate zones as measles with smaller CFR for C and conversion in the whole Empire (after AP and demo)",
    )

    # Function to plot the totals of compartments
    def plot_total_compartments():
        total_compartment_labels = [
            "Susceptible Christians",
            "Exposed Christians",
            "Infected Christians",
            "Recovered Christians",
            "Deceased Christians",
            "Susceptible Pagans",
            "Exposed Pagans",
            "Infected Pagans",
            "Recovered Pagans",
            "Deceased Pagans",
        ]

        # Calculate the total compartments for each group
        susceptible_christians = solution_cp.y[0, :] + solution_cp.y[12, :] + solution_cp.y[24, :] + solution_cp.y[36, :]
        exposed_christians = solution_cp.y[1, :] + solution_cp.y[13, :] + solution_cp.y[25, :] + solution_cp.y[37, :]
        infected_christians = solution_cp.y[2, :] + solution_cp.y[14, :] + solution_cp.y[26, :] + solution_cp.y[38, :]
        recovered_christians = solution_cp.y[3, :] + solution_cp.y[15, :] + solution_cp.y[27, :] + solution_cp.y[39, :]
        deceased_christians = solution_cp.y[4, :] + solution_cp.y[16, :] + solution_cp.y[28, :] + solution_cp.y[40, :]

        susceptible_pagans = solution_cp.y[6, :] + solution_cp.y[18, :] + solution_cp.y[30, :] + solution_cp.y[42, :]
        exposed_pagans = solution_cp.y[7, :] + solution_cp.y[19, :] + solution_cp.y[31, :] + solution_cp.y[43, :]
        infected_pagans = solution_cp.y[8, :] + solution_cp.y[20, :] + solution_cp.y[32, :] + solution_cp.y[44, :]
        recovered_pagans = solution_cp.y[9, :] + solution_cp.y[21, :] + solution_cp.y[33, :] + solution_cp.y[45, :]
        deceased_pagans = solution_cp.y[10, :] + solution_cp.y[22, :] + solution_cp.y[34, :] + solution_cp.y[46, :]

        total_compartments = [
            susceptible_christians, exposed_christians, infected_christians, recovered_christians, deceased_christians,
            susceptible_pagans, exposed_pagans, infected_pagans, recovered_pagans, deceased_pagans
        ]

        # Plot the totals
        plot_seir_model(
            solution_cp,
            t_cp,
            start_year=end_year_demographic,
            end_year=end_year_cp,
            compartment_indices=range(len(total_compartments)),
            compartment_labels=total_compartment_labels,
            every_nth_year=5,
            y_tick_interval=100_000,
            display_y_label_every_n_ticks=10,
            plot_title="Total compartments across all zones",
        )

    # Call the function to plot totals
    plot_total_compartments()

    # Print the final values of each compartment for debugging purposes
    compartments_after_conversion = solution_cp.y[:, -1]

    # Zone 1
    s1_c = compartments_after_conversion[0]
    e1_c = compartments_after_conversion[1]
    i1_c = compartments_after_conversion[2]
    r1_c = compartments_after_conversion[3]
    d1_c = compartments_after_conversion[4]
    a1_c = compartments_after_conversion[5]
    s1_p = compartments_after_conversion[6]
    e1_p = compartments_after_conversion[7]
    i1_p = compartments_after_conversion[8]
    r1_p = compartments_after_conversion[9]
    d1_p = compartments_after_conversion[10]
    a1_p = compartments_after_conversion[11]

    # Zone 2
    s2_c = compartments_after_conversion[12]
    e2_c = compartments_after_conversion[13]
    i2_c = compartments_after_conversion[14]
    r2_c = compartments_after_conversion[15]
    d2_c = compartments_after_conversion[16]
    a2_c = compartments_after_conversion[17]
    s2_p = compartments_after_conversion[18]
    e2_p = compartments_after_conversion[19]
    i2_p = compartments_after_conversion[20]
    r2_p = compartments_after_conversion[21]
    d2_p = compartments_after_conversion[22]
    a2_p = compartments_after_conversion[23]

    # Zone 3
    s3_c = compartments_after_conversion[24]
    e3_c = compartments_after_conversion[25]
    i3_c = compartments_after_conversion[26]
    r3_c = compartments_after_conversion[27]
    d3_c = compartments_after_conversion[28]
    a3_c = compartments_after_conversion[29]
    s3_p = compartments_after_conversion[30]
    e3_p = compartments_after_conversion[31]
    i3_p = compartments_after_conversion[32]
    r3_p = compartments_after_conversion[33]
    d3_p = compartments_after_conversion[34]
    a3_p = compartments_after_conversion[35]

    # Zone 4
    s4_c = compartments_after_conversion[36]
    e4_c = compartments_after_conversion[37]
    i4_c = compartments_after_conversion[38]
    r4_c = compartments_after_conversion[39]
    d4_c = compartments_after_conversion[40]
    a4_c = compartments_after_conversion[41]
    s4_p = compartments_after_conversion[42]
    e4_p = compartments_after_conversion[43]
    i4_p = compartments_after_conversion[44]
    r4_p = compartments_after_conversion[45]
    d4_p = compartments_after_conversion[46]
    a4_p = compartments_after_conversion[47]

    # Summing alive individuals for Christians and Pagans in all zones
    alive1_c = s1_c + e1_c + i1_c + r1_c
    alive1_p = s1_p + e1_p + i1_p + r1_p

    alive2_c = s2_c + e2_c + i2_c + r2_c
    alive2_p = s2_p + e2_p + i2_p + r2_p

    alive3_c = s3_c + e3_c + i3_c + r3_c
    alive3_p = s3_p + e3_p + i3_p + r3_p

    alive4_c = s4_c + e4_c + i4_c + r4_c
    alive4_p = s4_p + e4_p + i4_p + r4_p

    print(
        f"s1_c = {s1_c}\n"
        f"e1_c = {e1_c}\n"
        f"i1_c = {i1_c}\n"
        f"r1_c = {r1_c}\n"
        f"d1_c = {d1_c}\n"
        f"a1_c = {a1_c}\n"
        f"s1_p = {s1_p}\n"
        f"e1_p = {e1_p}\n"
        f"i1_p = {i1_p}\n"
        f"r1_p = {r1_p}\n"
        f"d1_p = {d1_p}\n"
        f"a1_p = {a1_p}\n"
        f"alive1_c = {alive1_c}\n"
        f"alive1_p = {alive1_p}\n"
        f"s2_c = {s2_c}\n"
        f"e2_c = {e2_c}\n"
        f"i2_c = {i2_c}\n"
        f"r2_c = {r2_c}\n"
        f"d2_c = {d2_c}\n"
        f"a2_c = {a2_c}\n"
        f"s2_p = {s2_p}\n"
        f"e2_p = {e2_p}\n"
        f"i2_p = {i2_p}\n"
        f"r2_p = {r2_p}\n"
        f"d2_p = {d2_p}\n"
        f"a2_p = {a2_p}\n"
        f"alive2_c = {alive2_c}\n"
        f"alive2_p = {alive2_p}\n"
        f"s3_c = {s3_c}\n"
        f"e3_c = {e3_c}\n"
        f"i3_c = {i3_c}\n"
        f"r3_c = {r3_c}\n"
        f"d3_c = {d3_c}\n"
        f"a3_c = {a3_c}\n"
        f"s3_p = {s3_p}\n"
        f"e3_p = {e3_p}\n"
        f"i3_p = {i3_p}\n"
        f"r3_p = {r3_p}\n"
        f"d3_p = {d3_p}\n"
        f"a3_p = {a3_p}\n"
        f"alive3_c = {alive3_c}\n"
        f"alive3_p = {alive3_p}\n"
        f"s4_c = {s4_c}\n"
        f"e4_c = {e4_c}\n"
        f"i4_c = {i4_c}\n"
        f"r4_c = {r4_c}\n"
        f"d4_c = {d4_c}\n"
        f"a4_c = {a4_c}\n"
        f"s4_p = {s4_p}\n"
        f"e4_p = {e4_p}\n"
        f"i4_p = {i4_p}\n"
        f"r4_p = {r4_p}\n"
        f"d4_p = {d4_p}\n"
        f"a4_p = {a4_p}\n"
        f"alive4_c = {alive4_c}\n"
        f"alive4_p = {alive4_p}"
    )


def proof_of_concept_solve_and_plot_ap_demo_cp_with_conversion_and_smaller_cfr_for_christians_with_four_dynamic_deltas(
    start_year_ap=165,  # 165
    end_year_ap=171,  # 189
    # end_year_ap=189,  # 189
    end_year_demographic=177,  # 248
    # end_year_demographic=248,  # 248
    end_year_cp=183,  # 270
    # end_year_cp=270,  # 270
    initial_christian_population=initial_christian_population,
    initial_pagan_population=initial_pagan_population,
):
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
        0,  # a4_p,
        zeleners_initial_delta_1,  # delta_1_c
        zeleners_initial_delta_1,  # delta_1_p
        zeleners_initial_delta_2,  # delta_2_c
        zeleners_initial_delta_2,  # delta_2_p
        zeleners_initial_delta_3,  # delta_3_c
        zeleners_initial_delta_3,  # delta_3_p
        zeleners_initial_delta_4,  # delta_4_c
        zeleners_initial_delta_4,  # delta_4_p
    ]

    # Timeframe of simulation
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    # t_ap = np.arange(0, total_days_ap)
    t_ap = np.linspace(0, total_days_ap, total_days_ap)

    # Solve the ODE for the Antonine Plague
    def wrapper_for_solve_ivp_ap(t, y):
        return (
            direct_transmission_with_four_dynamic_deltas_two_cfrs_and_conversion_in_pairs_seird_model(
                y, t, smallpox_seir_params_with_starks_conversion
            )
        )

    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="RK45", t_eval=t_ap,
    )

    # The demographic-development-without-a_specific-disease part:
    compartments_demographic = solution_ap.y[:-8, -1]  # Exclude the last 8 elements (4 deltas for Christians + Pagans)
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    # t_demographic = np.arange(0, total_days_demographic)
    t_demographic = np.linspace(0, total_days_demographic, total_days_demographic)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model_subpopulation_pairs_with_conversion_in_four_zones(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="RK45",
        t_eval=t_demographic,
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]


    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]
    y0_cp.extend([
        zeleners_initial_delta_1, zeleners_initial_delta_1,  # Zone 1 deltas
        zeleners_initial_delta_2, zeleners_initial_delta_2,  # Zone 2 deltas
        zeleners_initial_delta_3, zeleners_initial_delta_3,  # Zone 3 deltas
        zeleners_initial_delta_4, zeleners_initial_delta_4  # Zone 4 deltas
    ])

    # In each zone move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Zone 1
    # Reshuffling for Christians
    susceptible_christians_sum_zone_1 = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum_zone_1
    y0_cp[1] = 0  # Exposed Christians in Zone 1
    y0_cp[2] = 1  # Infected Christians in Zone 1
    y0_cp[3] = 0  # Recovered Christians in Zone 1
    # y0_cp[4] = 0  # Deceased Christians  in Zone 1 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_1 = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum_zone_1
    y0_cp[7] = 0  # Exposed Pagans in Zone 1
    y0_cp[8] = 1  # Infected Pagans in Zone 1
    y0_cp[9] = 0  # Recovered Pagans in Zone 1
    # y0_cp[10] = 0  # Deceased Pagans in Zone 1 (set to 0 to see what CP does)

    # Zone 2
    # Reshuffling for Christians
    susceptible_christians_sum_zone_2 = sum(y0_cp[12:16])
    y0_cp[12] = susceptible_christians_sum_zone_2
    y0_cp[13] = 0  # Exposed Christians in Zone 2
    y0_cp[14] = 1  # Infected Christians in Zone 2
    y0_cp[15] = 0  # Recovered Christians in Zone 2
    # y0_cp[16] = 0  # Deceased Christians  in Zone 2 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_2 = sum(y0_cp[18:22])
    y0_cp[18] = susceptible_pagans_sum_zone_2
    y0_cp[19] = 0  # Exposed Pagans in Zone 2
    y0_cp[20] = 1  # Infected Pagans in Zone 2
    y0_cp[21] = 0  # Recovered Pagans in Zone 2
    # y0_cp[22] = 0  # Deceased Pagans in Zone 2 (set to 0 to see what CP does)

    # Zone 3
    # Reshuffling for Christians
    susceptible_christians_sum_zone_3 = sum(y0_cp[24:28])
    y0_cp[24] = susceptible_christians_sum_zone_3
    y0_cp[25] = 0  # Exposed Christians in Zone 3
    y0_cp[26] = 1  # Infected Christians in Zone 3
    y0_cp[27] = 0  # Recovered Christians in Zone 3
    # y0_cp[28] = 0  # Deceased Christians  in Zone 3 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_3 = sum(y0_cp[30:34])
    y0_cp[30] = susceptible_pagans_sum_zone_3
    y0_cp[31] = 0  # Exposed Pagans in Zone 3
    y0_cp[32] = 1  # Infected Pagans in Zone 3
    y0_cp[33] = 0  # Recovered Pagans in Zone 3
    # y0_cp[34] = 0  # Deceased Pagans in Zone 3 (set to 0 to see what CP does)

    # Zone 4
    # Reshuffling for Christians
    susceptible_christians_sum_zone_4 = sum(y0_cp[36:40])
    y0_cp[36] = susceptible_christians_sum_zone_4
    y0_cp[37] = 0  # Exposed Christians in Zone 4
    y0_cp[38] = 1  # Infected Christians in Zone 4
    y0_cp[39] = 0  # Recovered Christians in Zone 4
    # y0_cp[40] = 0  # Deceased Christians  in Zone 4 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_4 = sum(y0_cp[42:46])
    y0_cp[42] = susceptible_pagans_sum_zone_4
    y0_cp[43] = 0  # Exposed Pagans in Zone 4
    y0_cp[44] = 1  # Infected Pagans in Zone 4
    y0_cp[45] = 0  # Recovered Pagans in Zone 4
    # y0_cp[46] = 0  # Deceased Pagans in Zone 4 (set to 0 to see what CP does)

    # Interaction rates
    y0_cp[48] = zeleners_initial_delta_1  # delta_1_c
    y0_cp[49] = zeleners_initial_delta_1  # delta_1_p
    y0_cp[50] = zeleners_initial_delta_2  # delta_2_c
    y0_cp[51] = zeleners_initial_delta_2  # delta_2_p
    y0_cp[52] = zeleners_initial_delta_3  # delta_3_c
    y0_cp[53] = zeleners_initial_delta_3  # delta_3_p
    y0_cp[54] = zeleners_initial_delta_4  # delta_4_c
    y0_cp[55] = zeleners_initial_delta_4  # delta_4_p


    # Timeframe of simulation for the Cyprianic Plague model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    # t_cp = np.arange(0, total_days_cp)
    t_cp = np.linspace(0, total_days_cp, total_days_cp + 1)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_with_four_dynamic_deltas_two_cfrs_and_conversion_in_pairs_seird_model(
            y, t, measles_seir_params_with_lower_cfr_for_c_and_starks_conversion
        )

    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="RK45",
        t_eval=t_cp,
        atol=1e-8,
        rtol=1e-6,
    )


    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [
        0, 1, 2, 3,  # Christians in Zone 1
        6, 7, 8, 9,  # Pagans in Zone 1
        12, 13, 14, 15,  # Christians in Zone 2
        18, 19, 20, 21,  # Pagans in Zone 2
        24, 25, 26, 27,  # Christians in Zone 3
        30, 31, 32, 33,  # Pagans in Zone 3
        36, 37, 38, 39,  # Christians in Zone 4
        42, 43, 44, 45  # Pagans in Zone 4
    ]
    compartment_labels = [
        # Zone 1
        "Susceptible Christians in Zone 1",
        "Exposed Christians in Zone 1",
        "Infected Christians in Zone 1",
        "Recovered Christians in Zone 1",
        # "Deceased Christians in Zone 1",

        "Susceptible Pagans in Zone 1",
        "Exposed Pagans in Zone 1",
        "Infected Pagans in Zone 1",
        "Recovered Pagans in Zone 1",
        # "Deceased Pagans in Zone 1",

        # Zone 2
        "Susceptible Christians in Zone 2",
        "Exposed Christians in Zone 2",
        "Infected Christians in Zone 2",
        "Recovered Christians in Zone 2",
        # "Deceased Christians in Zone 2",

        "Susceptible Pagans in Zone 2",
        "Exposed Pagans in Zone 2",
        "Infected Pagans in Zone 2",
        "Recovered Pagans in Zone 2",
        # "Deceased Pagans in Zone 2",

        # Zone 3
        "Susceptible Christians in Zone 3",
        "Exposed Christians in Zone 3",
        "Infected Christians in Zone 3",
        "Recovered Christians in Zone 3",
        # "Deceased Christians in Zone 3",

        "Susceptible Pagans in Zone 3",
        "Exposed Pagans in Zone 3",
        "Infected Pagans in Zone 3",
        "Recovered Pagans in Zone 3",
        # "Deceased Pagans in Zone 3",

        # Zone 4
        "Susceptible Christians in Zone 4",
        "Exposed Christians in Zone 4",
        "Infected Christians in Zone 4",
        "Recovered Christians in Zone 4",
        # "Deceased Christians in Zone 4",

        "Susceptible Pagans in Zone 4",
        "Exposed Pagans in Zone 4",
        "Infected Pagans in Zone 4",
        "Recovered Pagans in Zone 4",
        # "Deceased Pagans in Zone 4"
    ]
    plot_seir_model(
        solution_cp,
        t_cp,
        start_year=end_year_demographic,
        end_year=end_year_cp,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="CP in 4 separate zones as measles with smaller CFR for C and conversion in the whole Empire (after AP and demo)",
    )

    # Function to plot the totals of compartments
    def plot_total_compartments():
        total_compartment_labels = [
            "Susceptible Christians",
            "Exposed Christians",
            "Infected Christians",
            "Recovered Christians",
            "Deceased Christians",
            "Susceptible Pagans",
            "Exposed Pagans",
            "Infected Pagans",
            "Recovered Pagans",
            "Deceased Pagans",
        ]

        # Calculate the total compartments for each group
        susceptible_christians = solution_cp.y[0, :] + solution_cp.y[12, :] + solution_cp.y[24, :] + solution_cp.y[36, :]
        exposed_christians = solution_cp.y[1, :] + solution_cp.y[13, :] + solution_cp.y[25, :] + solution_cp.y[37, :]
        infected_christians = solution_cp.y[2, :] + solution_cp.y[14, :] + solution_cp.y[26, :] + solution_cp.y[38, :]
        recovered_christians = solution_cp.y[3, :] + solution_cp.y[15, :] + solution_cp.y[27, :] + solution_cp.y[39, :]
        deceased_christians = solution_cp.y[4, :] + solution_cp.y[16, :] + solution_cp.y[28, :] + solution_cp.y[40, :]

        susceptible_pagans = solution_cp.y[6, :] + solution_cp.y[18, :] + solution_cp.y[30, :] + solution_cp.y[42, :]
        exposed_pagans = solution_cp.y[7, :] + solution_cp.y[19, :] + solution_cp.y[31, :] + solution_cp.y[43, :]
        infected_pagans = solution_cp.y[8, :] + solution_cp.y[20, :] + solution_cp.y[32, :] + solution_cp.y[44, :]
        recovered_pagans = solution_cp.y[9, :] + solution_cp.y[21, :] + solution_cp.y[33, :] + solution_cp.y[45, :]
        deceased_pagans = solution_cp.y[10, :] + solution_cp.y[22, :] + solution_cp.y[34, :] + solution_cp.y[46, :]

        total_compartments = [
            susceptible_christians, exposed_christians, infected_christians, recovered_christians, deceased_christians,
            susceptible_pagans, exposed_pagans, infected_pagans, recovered_pagans, deceased_pagans
        ]

        # Plot the totals
        plot_seir_model(
            solution_cp,
            t_cp,
            start_year=end_year_demographic,
            end_year=end_year_cp,
            compartment_indices=range(len(total_compartments)),
            compartment_labels=total_compartment_labels,
            every_nth_year=5,
            y_tick_interval=100_000,
            display_y_label_every_n_ticks=10,
            plot_title="Total compartments across all zones",
        )

    # Call the function to plot totals
    plot_total_compartments()

    # Print the final values of each compartment for debugging purposes
    compartments_after_conversion = solution_cp.y[:, -1]

    # Zone 1
    s1_c = compartments_after_conversion[0]
    e1_c = compartments_after_conversion[1]
    i1_c = compartments_after_conversion[2]
    r1_c = compartments_after_conversion[3]
    d1_c = compartments_after_conversion[4]
    a1_c = compartments_after_conversion[5]
    s1_p = compartments_after_conversion[6]
    e1_p = compartments_after_conversion[7]
    i1_p = compartments_after_conversion[8]
    r1_p = compartments_after_conversion[9]
    d1_p = compartments_after_conversion[10]
    a1_p = compartments_after_conversion[11]

    # Zone 2
    s2_c = compartments_after_conversion[12]
    e2_c = compartments_after_conversion[13]
    i2_c = compartments_after_conversion[14]
    r2_c = compartments_after_conversion[15]
    d2_c = compartments_after_conversion[16]
    a2_c = compartments_after_conversion[17]
    s2_p = compartments_after_conversion[18]
    e2_p = compartments_after_conversion[19]
    i2_p = compartments_after_conversion[20]
    r2_p = compartments_after_conversion[21]
    d2_p = compartments_after_conversion[22]
    a2_p = compartments_after_conversion[23]

    # Zone 3
    s3_c = compartments_after_conversion[24]
    e3_c = compartments_after_conversion[25]
    i3_c = compartments_after_conversion[26]
    r3_c = compartments_after_conversion[27]
    d3_c = compartments_after_conversion[28]
    a3_c = compartments_after_conversion[29]
    s3_p = compartments_after_conversion[30]
    e3_p = compartments_after_conversion[31]
    i3_p = compartments_after_conversion[32]
    r3_p = compartments_after_conversion[33]
    d3_p = compartments_after_conversion[34]
    a3_p = compartments_after_conversion[35]

    # Zone 4
    s4_c = compartments_after_conversion[36]
    e4_c = compartments_after_conversion[37]
    i4_c = compartments_after_conversion[38]
    r4_c = compartments_after_conversion[39]
    d4_c = compartments_after_conversion[40]
    a4_c = compartments_after_conversion[41]
    s4_p = compartments_after_conversion[42]
    e4_p = compartments_after_conversion[43]
    i4_p = compartments_after_conversion[44]
    r4_p = compartments_after_conversion[45]
    d4_p = compartments_after_conversion[46]
    a4_p = compartments_after_conversion[47]

    # Summing alive individuals for Christians and Pagans in all zones
    alive1_c = s1_c + e1_c + i1_c + r1_c
    alive1_p = s1_p + e1_p + i1_p + r1_p

    alive2_c = s2_c + e2_c + i2_c + r2_c
    alive2_p = s2_p + e2_p + i2_p + r2_p

    alive3_c = s3_c + e3_c + i3_c + r3_c
    alive3_p = s3_p + e3_p + i3_p + r3_p

    alive4_c = s4_c + e4_c + i4_c + r4_c
    alive4_p = s4_p + e4_p + i4_p + r4_p

    print(
        f"s1_c = {s1_c}\n"
        f"e1_c = {e1_c}\n"
        f"i1_c = {i1_c}\n"
        f"r1_c = {r1_c}\n"
        f"d1_c = {d1_c}\n"
        f"a1_c = {a1_c}\n"
        f"s1_p = {s1_p}\n"
        f"e1_p = {e1_p}\n"
        f"i1_p = {i1_p}\n"
        f"r1_p = {r1_p}\n"
        f"d1_p = {d1_p}\n"
        f"a1_p = {a1_p}\n"
        f"alive1_c = {alive1_c}\n"
        f"alive1_p = {alive1_p}\n"
        f"s2_c = {s2_c}\n"
        f"e2_c = {e2_c}\n"
        f"i2_c = {i2_c}\n"
        f"r2_c = {r2_c}\n"
        f"d2_c = {d2_c}\n"
        f"a2_c = {a2_c}\n"
        f"s2_p = {s2_p}\n"
        f"e2_p = {e2_p}\n"
        f"i2_p = {i2_p}\n"
        f"r2_p = {r2_p}\n"
        f"d2_p = {d2_p}\n"
        f"a2_p = {a2_p}\n"
        f"alive2_c = {alive2_c}\n"
        f"alive2_p = {alive2_p}\n"
        f"s3_c = {s3_c}\n"
        f"e3_c = {e3_c}\n"
        f"i3_c = {i3_c}\n"
        f"r3_c = {r3_c}\n"
        f"d3_c = {d3_c}\n"
        f"a3_c = {a3_c}\n"
        f"s3_p = {s3_p}\n"
        f"e3_p = {e3_p}\n"
        f"i3_p = {i3_p}\n"
        f"r3_p = {r3_p}\n"
        f"d3_p = {d3_p}\n"
        f"a3_p = {a3_p}\n"
        f"alive3_c = {alive3_c}\n"
        f"alive3_p = {alive3_p}\n"
        f"s4_c = {s4_c}\n"
        f"e4_c = {e4_c}\n"
        f"i4_c = {i4_c}\n"
        f"r4_c = {r4_c}\n"
        f"d4_c = {d4_c}\n"
        f"a4_c = {a4_c}\n"
        f"s4_p = {s4_p}\n"
        f"e4_p = {e4_p}\n"
        f"i4_p = {i4_p}\n"
        f"r4_p = {r4_p}\n"
        f"d4_p = {d4_p}\n"
        f"a4_p = {a4_p}\n"
        f"alive4_c = {alive4_c}\n"
        f"alive4_p = {alive4_p}"
    )


def new_poc_solve_and_plot_ap_demo_cp_with_conversion_and_smaller_cfr_for_christians_with_four_deltas(
    start_year_ap=165,
    end_year_ap=189,
    end_year_demographic=248,
    end_year_cp=270
):
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

    # Timeframe of simulation
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    t_ap = np.linspace(0, total_days_ap, total_days_ap + 1)

    # Solve the ODE for the Antonine Plague
    # TODO: implement the seir function with some delta usage - a new one, tho!
    def wrapper_for_solve_ivp_ap(t, y):
        return (
            direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model(
                y, t, smallpox_seir_params_with_starks_conversion
            )
        )

    print(y0_ap)
    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="RK45", t_eval=t_ap
    )

    # The demographic-development-without-a_specific-disease part:
    # compartments_demographic = solution_ap.y[:, -1]
    compartments_demographic = solution_ap.y[:-8, -1]  # Exclude the last 8 elements (4 deltas for Christians + Pagans)
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    t_demographic = np.linspace(0, total_days_demographic, total_days_demographic + 1)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model_subpopulation_pairs_with_conversion_in_four_zones(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    print(y0_demographic)
    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="RK45",
        t_eval=t_demographic
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]

    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]
    # TODO: check if this extension is not a mistake
    y0_cp.extend([
        zeleners_initial_delta_1, zeleners_initial_delta_1,  # Zone 1 deltas
        zeleners_initial_delta_2, zeleners_initial_delta_2,  # Zone 2 deltas
        zeleners_initial_delta_3, zeleners_initial_delta_3,  # Zone 3 deltas
        zeleners_initial_delta_4, zeleners_initial_delta_4  # Zone 4 deltas
    ])

    # In each zone move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Zone 1
    # Reshuffling for Christians
    susceptible_christians_sum_zone_1 = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum_zone_1
    y0_cp[1] = 0  # Exposed Christians in Zone 1
    y0_cp[2] = 1  # Infected Christians in Zone 1
    y0_cp[3] = 0  # Recovered Christians in Zone 1
    # y0_cp[4] = 0  # Deceased Christians  in Zone 1 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_1 = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum_zone_1
    y0_cp[7] = 0  # Exposed Pagans in Zone 1
    y0_cp[8] = 1  # Infected Pagans in Zone 1
    y0_cp[9] = 0  # Recovered Pagans in Zone 1
    # y0_cp[10] = 0  # Deceased Pagans in Zone 1 (set to 0 to see what CP does)

    # Zone 2
    # Reshuffling for Christians
    susceptible_christians_sum_zone_2 = sum(y0_cp[12:16])
    y0_cp[12] = susceptible_christians_sum_zone_2
    y0_cp[13] = 0  # Exposed Christians in Zone 2
    y0_cp[14] = 1  # Infected Christians in Zone 2
    y0_cp[15] = 0  # Recovered Christians in Zone 2
    # y0_cp[16] = 0  # Deceased Christians  in Zone 2 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_2 = sum(y0_cp[18:22])
    y0_cp[18] = susceptible_pagans_sum_zone_2
    y0_cp[19] = 0  # Exposed Pagans in Zone 2
    y0_cp[20] = 1  # Infected Pagans in Zone 2
    y0_cp[21] = 0  # Recovered Pagans in Zone 2
    # y0_cp[22] = 0  # Deceased Pagans in Zone 2 (set to 0 to see what CP does)

    # Zone 3
    # Reshuffling for Christians
    susceptible_christians_sum_zone_3 = sum(y0_cp[24:28])
    y0_cp[24] = susceptible_christians_sum_zone_3
    y0_cp[25] = 0  # Exposed Christians in Zone 3
    y0_cp[26] = 1  # Infected Christians in Zone 3
    y0_cp[27] = 0  # Recovered Christians in Zone 3
    # y0_cp[28] = 0  # Deceased Christians  in Zone 3 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_3 = sum(y0_cp[30:34])
    y0_cp[30] = susceptible_pagans_sum_zone_3
    y0_cp[31] = 0  # Exposed Pagans in Zone 3
    y0_cp[32] = 1  # Infected Pagans in Zone 3
    y0_cp[33] = 0  # Recovered Pagans in Zone 3
    # y0_cp[34] = 0  # Deceased Pagans in Zone 3 (set to 0 to see what CP does)

    # Zone 4
    # Reshuffling for Christians
    susceptible_christians_sum_zone_4 = sum(y0_cp[36:40])
    y0_cp[36] = susceptible_christians_sum_zone_4
    y0_cp[37] = 0  # Exposed Christians in Zone 4
    y0_cp[38] = 1  # Infected Christians in Zone 4
    y0_cp[39] = 0  # Recovered Christians in Zone 4
    # y0_cp[40] = 0  # Deceased Christians  in Zone 4 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_4 = sum(y0_cp[42:46])
    y0_cp[42] = susceptible_pagans_sum_zone_4
    y0_cp[43] = 0  # Exposed Pagans in Zone 4
    y0_cp[44] = 1  # Infected Pagans in Zone 4
    y0_cp[45] = 0  # Recovered Pagans in Zone 4
    # y0_cp[46] = 0  # Deceased Pagans in Zone 4 (set to 0 to see what CP does)

    # Interaction rates
    y0_cp[48] = zeleners_initial_delta_1  # delta_1_c
    y0_cp[49] = zeleners_initial_delta_1  # delta_1_p
    y0_cp[50] = zeleners_initial_delta_2  # delta_2_c
    y0_cp[51] = zeleners_initial_delta_2  # delta_2_p
    y0_cp[52] = zeleners_initial_delta_3  # delta_3_c
    y0_cp[53] = zeleners_initial_delta_3  # delta_3_p
    y0_cp[54] = zeleners_initial_delta_4  # delta_4_c
    y0_cp[55] = zeleners_initial_delta_4  # delta_4_p

    # Timeframe of simulation for the Cyprianic Plague model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    t_cp = np.linspace(0, total_days_cp, total_days_cp + 1)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model(
            y, t, measles_seir_params_with_lower_cfr_for_c_and_starks_conversion
        )

    print(y0_cp)
    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="RK45",
        t_eval=t_cp
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [
        0, 1, 2, 3,  # Christians in Zone 1
        6, 7, 8, 9,  # Pagans in Zone 1
        12, 13, 14, 15,  # Christians in Zone 2
        18, 19, 20, 21,  # Pagans in Zone 2
        24, 25, 26, 27,  # Christians in Zone 3
        30, 31, 32, 33,  # Pagans in Zone 3
        36, 37, 38, 39,  # Christians in Zone 4
        42, 43, 44, 45  # Pagans in Zone 4
    ]
    compartment_labels = [
        # Zone 1
        "Susceptible Christians in Zone 1",
        "Exposed Christians in Zone 1",
        "Infected Christians in Zone 1",
        "Recovered Christians in Zone 1",
        # "Deceased Christians in Zone 1",

        "Susceptible Pagans in Zone 1",
        "Exposed Pagans in Zone 1",
        "Infected Pagans in Zone 1",
        "Recovered Pagans in Zone 1",
        # "Deceased Pagans in Zone 1",

        # Zone 2
        "Susceptible Christians in Zone 2",
        "Exposed Christians in Zone 2",
        "Infected Christians in Zone 2",
        "Recovered Christians in Zone 2",
        # "Deceased Christians in Zone 2",

        "Susceptible Pagans in Zone 2",
        "Exposed Pagans in Zone 2",
        "Infected Pagans in Zone 2",
        "Recovered Pagans in Zone 2",
        # "Deceased Pagans in Zone 2",

        # Zone 3
        "Susceptible Christians in Zone 3",
        "Exposed Christians in Zone 3",
        "Infected Christians in Zone 3",
        "Recovered Christians in Zone 3",
        # "Deceased Christians in Zone 3",

        "Susceptible Pagans in Zone 3",
        "Exposed Pagans in Zone 3",
        "Infected Pagans in Zone 3",
        "Recovered Pagans in Zone 3",
        # "Deceased Pagans in Zone 3",

        # Zone 4
        "Susceptible Christians in Zone 4",
        "Exposed Christians in Zone 4",
        "Infected Christians in Zone 4",
        "Recovered Christians in Zone 4",
        # "Deceased Christians in Zone 4",

        "Susceptible Pagans in Zone 4",
        "Exposed Pagans in Zone 4",
        "Infected Pagans in Zone 4",
        "Recovered Pagans in Zone 4",
        # "Deceased Pagans in Zone 4"
    ]
    plot_seir_model(
        solution_cp,
        t_cp,
        start_year=end_year_demographic,
        end_year=end_year_cp,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="CP in 4 separate zones as measles with smaller CFR for C and conversion in the whole Empire (after AP and demo)",
    )

    # Function to plot the totals of compartments
    def plot_total_compartments():
        total_compartment_labels = [
            "Susceptible Christians",
            "Exposed Christians",
            "Infected Christians",
            "Recovered Christians",
            "Deceased Christians",
            "Susceptible Pagans",
            "Exposed Pagans",
            "Infected Pagans",
            "Recovered Pagans",
            "Deceased Pagans",
        ]

        # Calculate the total compartments for each group
        susceptible_christians = solution_cp.y[0, :] + solution_cp.y[12, :] + solution_cp.y[24, :] + solution_cp.y[36, :]
        exposed_christians = solution_cp.y[1, :] + solution_cp.y[13, :] + solution_cp.y[25, :] + solution_cp.y[37, :]
        infected_christians = solution_cp.y[2, :] + solution_cp.y[14, :] + solution_cp.y[26, :] + solution_cp.y[38, :]
        recovered_christians = solution_cp.y[3, :] + solution_cp.y[15, :] + solution_cp.y[27, :] + solution_cp.y[39, :]
        deceased_christians = solution_cp.y[4, :] + solution_cp.y[16, :] + solution_cp.y[28, :] + solution_cp.y[40, :]

        susceptible_pagans = solution_cp.y[6, :] + solution_cp.y[18, :] + solution_cp.y[30, :] + solution_cp.y[42, :]
        exposed_pagans = solution_cp.y[7, :] + solution_cp.y[19, :] + solution_cp.y[31, :] + solution_cp.y[43, :]
        infected_pagans = solution_cp.y[8, :] + solution_cp.y[20, :] + solution_cp.y[32, :] + solution_cp.y[44, :]
        recovered_pagans = solution_cp.y[9, :] + solution_cp.y[21, :] + solution_cp.y[33, :] + solution_cp.y[45, :]
        deceased_pagans = solution_cp.y[10, :] + solution_cp.y[22, :] + solution_cp.y[34, :] + solution_cp.y[46, :]

        total_compartments = [
            susceptible_christians, exposed_christians, infected_christians, recovered_christians, deceased_christians,
            susceptible_pagans, exposed_pagans, infected_pagans, recovered_pagans, deceased_pagans
        ]

        # Plot the totals
        plot_seir_model(
            solution_cp,
            t_cp,
            start_year=end_year_demographic,
            end_year=end_year_cp,
            compartment_indices=range(len(total_compartments)),
            compartment_labels=total_compartment_labels,
            every_nth_year=5,
            y_tick_interval=100_000,
            display_y_label_every_n_ticks=10,
            plot_title="Total compartments across all zones",
        )

    # Call the function to plot totals
    plot_total_compartments()

    # Print the final values of each compartment for debugging purposes
    compartments_after_conversion = solution_cp.y[:, -1]

    # Zone 1
    s1_c = compartments_after_conversion[0]
    e1_c = compartments_after_conversion[1]
    i1_c = compartments_after_conversion[2]
    r1_c = compartments_after_conversion[3]
    d1_c = compartments_after_conversion[4]
    a1_c = compartments_after_conversion[5]
    s1_p = compartments_after_conversion[6]
    e1_p = compartments_after_conversion[7]
    i1_p = compartments_after_conversion[8]
    r1_p = compartments_after_conversion[9]
    d1_p = compartments_after_conversion[10]
    a1_p = compartments_after_conversion[11]

    # Zone 2
    s2_c = compartments_after_conversion[12]
    e2_c = compartments_after_conversion[13]
    i2_c = compartments_after_conversion[14]
    r2_c = compartments_after_conversion[15]
    d2_c = compartments_after_conversion[16]
    a2_c = compartments_after_conversion[17]
    s2_p = compartments_after_conversion[18]
    e2_p = compartments_after_conversion[19]
    i2_p = compartments_after_conversion[20]
    r2_p = compartments_after_conversion[21]
    d2_p = compartments_after_conversion[22]
    a2_p = compartments_after_conversion[23]

    # Zone 3
    s3_c = compartments_after_conversion[24]
    e3_c = compartments_after_conversion[25]
    i3_c = compartments_after_conversion[26]
    r3_c = compartments_after_conversion[27]
    d3_c = compartments_after_conversion[28]
    a3_c = compartments_after_conversion[29]
    s3_p = compartments_after_conversion[30]
    e3_p = compartments_after_conversion[31]
    i3_p = compartments_after_conversion[32]
    r3_p = compartments_after_conversion[33]
    d3_p = compartments_after_conversion[34]
    a3_p = compartments_after_conversion[35]

    # Zone 4
    s4_c = compartments_after_conversion[36]
    e4_c = compartments_after_conversion[37]
    i4_c = compartments_after_conversion[38]
    r4_c = compartments_after_conversion[39]
    d4_c = compartments_after_conversion[40]
    a4_c = compartments_after_conversion[41]
    s4_p = compartments_after_conversion[42]
    e4_p = compartments_after_conversion[43]
    i4_p = compartments_after_conversion[44]
    r4_p = compartments_after_conversion[45]
    d4_p = compartments_after_conversion[46]
    a4_p = compartments_after_conversion[47]

    # Summing alive individuals for Christians and Pagans in all zones
    alive1_c = s1_c + e1_c + i1_c + r1_c
    alive1_p = s1_p + e1_p + i1_p + r1_p

    alive2_c = s2_c + e2_c + i2_c + r2_c
    alive2_p = s2_p + e2_p + i2_p + r2_p

    alive3_c = s3_c + e3_c + i3_c + r3_c
    alive3_p = s3_p + e3_p + i3_p + r3_p

    alive4_c = s4_c + e4_c + i4_c + r4_c
    alive4_p = s4_p + e4_p + i4_p + r4_p

    alive_total = alive1_c + alive1_p + alive2_c + alive2_p + alive3_c + alive3_p + alive4_c + alive4_p

    print(
        f"s1_c = {s1_c}\n"
        f"e1_c = {e1_c}\n"
        f"i1_c = {i1_c}\n"
        f"r1_c = {r1_c}\n"
        f"d1_c = {d1_c}\n"
        f"a1_c = {a1_c}\n"
        f"s1_p = {s1_p}\n"
        f"e1_p = {e1_p}\n"
        f"i1_p = {i1_p}\n"
        f"r1_p = {r1_p}\n"
        f"d1_p = {d1_p}\n"
        f"a1_p = {a1_p}\n"
        f"alive1_c = {alive1_c}\n"
        f"alive1_p = {alive1_p}\n"
        f"s2_c = {s2_c}\n"
        f"e2_c = {e2_c}\n"
        f"i2_c = {i2_c}\n"
        f"r2_c = {r2_c}\n"
        f"d2_c = {d2_c}\n"
        f"a2_c = {a2_c}\n"
        f"s2_p = {s2_p}\n"
        f"e2_p = {e2_p}\n"
        f"i2_p = {i2_p}\n"
        f"r2_p = {r2_p}\n"
        f"d2_p = {d2_p}\n"
        f"a2_p = {a2_p}\n"
        f"alive2_c = {alive2_c}\n"
        f"alive2_p = {alive2_p}\n"
        f"s3_c = {s3_c}\n"
        f"e3_c = {e3_c}\n"
        f"i3_c = {i3_c}\n"
        f"r3_c = {r3_c}\n"
        f"d3_c = {d3_c}\n"
        f"a3_c = {a3_c}\n"
        f"s3_p = {s3_p}\n"
        f"e3_p = {e3_p}\n"
        f"i3_p = {i3_p}\n"
        f"r3_p = {r3_p}\n"
        f"d3_p = {d3_p}\n"
        f"a3_p = {a3_p}\n"
        f"alive3_c = {alive3_c}\n"
        f"alive3_p = {alive3_p}\n"
        f"s4_c = {s4_c}\n"
        f"e4_c = {e4_c}\n"
        f"i4_c = {i4_c}\n"
        f"r4_c = {r4_c}\n"
        f"d4_c = {d4_c}\n"
        f"a4_c = {a4_c}\n"
        f"s4_p = {s4_p}\n"
        f"e4_p = {e4_p}\n"
        f"i4_p = {i4_p}\n"
        f"r4_p = {r4_p}\n"
        f"d4_p = {d4_p}\n"
        f"a4_p = {a4_p}\n"
        f"alive4_c = {alive4_c}\n"
        f"alive4_p = {alive4_p}\n"
        f"alive_total = {alive_total}\n"
    )

    print("Length of t_cp:", len(t_cp))
    print("Length of solution_cp.t:", len(solution_cp.t))
    print("solution_cp.t[-1]:", solution_cp.t[-1])

    for i, label in enumerate(compartment_labels):
        max_val = np.max(solution_cp.y[compartment_indices[i], :])
        print(f"{label}: max = {max_val}")

    print("Solver success:", solution_cp.success)
    print("Solver message:", solution_cp.message)

    print(
        "final deltas:\n"
        f"delta_1_c = {y0_cp[48]}\n"
        f"delta_1_p = {y0_cp[49]}\n"
        f"delta_2_c = {y0_cp[50]}\n"
        f"delta_2_p = {y0_cp[51]}\n"
        f"delta_3_c = {y0_cp[52]}\n"
        f"delta_3_p = {y0_cp[53]}\n"
        f"delta_4_c = {y0_cp[54]}\n"
        f"delta_4_p = {y0_cp[55]}\n"
    )
    # y0_cp[48] = zeleners_initial_delta_1  # delta_1_c
    # y0_cp[49] = zeleners_initial_delta_1  # delta_1_p
    # y0_cp[50] = zeleners_initial_delta_2  # delta_2_c
    # y0_cp[51] = zeleners_initial_delta_2  # delta_2_p
    # y0_cp[52] = zeleners_initial_delta_3  # delta_3_c
    # y0_cp[53] = zeleners_initial_delta_3  # delta_3_p
    # y0_cp[54] = zeleners_initial_delta_4  # delta_4_c
    # y0_cp[55] = zeleners_initial_delta_4  # delta_4_p

def poc_solve_and_plot_ap_demo_cp_with_converstion_to_denser_of_four_zones_and_smaller_cfr_for_christians(
    start_year_ap=165,
    end_year_ap=189,
    end_year_demographic=248,
    end_year_cp=270
):
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

    # Timeframe of simulation
    total_days_ap = (end_year_ap - start_year_ap + 1) * 365
    t_ap = np.linspace(0, total_days_ap, total_days_ap + 1)

    # Solve the ODE for the Antonine Plague
    # TODO: implement the seir function with some delta usage - a new one, tho!
    def wrapper_for_solve_ivp_ap(t, y):
        return (
            direct_transmission_over_four_pairs_of_connected_subpopulations_with_two_cfrs_and_converts_to_dense_zones_seird_model(
                y, t, smallpox_seir_params_with_starks_conversion
            )
        )

    print(y0_ap)
    solution_ap = solve_ivp(
        wrapper_for_solve_ivp_ap, [0, total_days_ap], y0_ap, method="RK45", t_eval=t_ap
    )

    # The demographic-development-without-a_specific-disease part:
    # compartments_demographic = solution_ap.y[:, -1]
    compartments_demographic = solution_ap.y[:-8, -1]  # Exclude the last 8 elements (4 deltas for Christians + Pagans)
    y0_demographic = [max(0, compartment) for compartment in compartments_demographic]

    # Timeframe of simulation for the demographic model
    total_days_demographic = (end_year_demographic - end_year_ap + 1) * 365
    t_demographic = np.linspace(0, total_days_demographic, total_days_demographic + 1)

    # Solve the demographic model ODE
    def wrapper_for_solve_ivp_demographic(t, y):
        return simple_demographic_model_subpopulation_pairs_with_conversion_to_dense_zones(
            y, t, smallpox_seir_params_with_starks_conversion
        )

    print(y0_demographic)
    solution_demographic = solve_ivp(
        wrapper_for_solve_ivp_demographic,
        [0, total_days_demographic],
        y0_demographic,
        method="RK45",
        t_eval=t_demographic
    )

    # The Cyprianic Plague part:
    compartments_cp = solution_demographic.y[:, -1]

    # Despite no compartment expected to be less than zero at this point,
    # ensure the lowest value indeed is zero.
    y0_cp = [max(0, compartment) for compartment in compartments_cp]
    # TODO: check if this extension is not a mistake
    y0_cp.extend([
        zeleners_initial_delta_1, zeleners_initial_delta_1,  # Zone 1 deltas
        zeleners_initial_delta_2, zeleners_initial_delta_2,  # Zone 2 deltas
        zeleners_initial_delta_3, zeleners_initial_delta_3,  # Zone 3 deltas
        zeleners_initial_delta_4, zeleners_initial_delta_4  # Zone 4 deltas
    ])

    # In each zone move individuals around compartments (1 infected in each subpopulation,
    # all remaining alive should go to susceptible compartments).
    # Zone 1
    # Reshuffling for Christians
    susceptible_christians_sum_zone_1 = sum(y0_cp[0:4])
    y0_cp[0] = susceptible_christians_sum_zone_1
    y0_cp[1] = 0  # Exposed Christians in Zone 1
    y0_cp[2] = 1  # Infected Christians in Zone 1
    y0_cp[3] = 0  # Recovered Christians in Zone 1
    # y0_cp[4] = 0  # Deceased Christians  in Zone 1 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_1 = sum(y0_cp[6:10])
    y0_cp[6] = susceptible_pagans_sum_zone_1
    y0_cp[7] = 0  # Exposed Pagans in Zone 1
    y0_cp[8] = 1  # Infected Pagans in Zone 1
    y0_cp[9] = 0  # Recovered Pagans in Zone 1
    # y0_cp[10] = 0  # Deceased Pagans in Zone 1 (set to 0 to see what CP does)

    # Zone 2
    # Reshuffling for Christians
    susceptible_christians_sum_zone_2 = sum(y0_cp[12:16])
    y0_cp[12] = susceptible_christians_sum_zone_2
    y0_cp[13] = 0  # Exposed Christians in Zone 2
    y0_cp[14] = 1  # Infected Christians in Zone 2
    y0_cp[15] = 0  # Recovered Christians in Zone 2
    # y0_cp[16] = 0  # Deceased Christians  in Zone 2 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_2 = sum(y0_cp[18:22])
    y0_cp[18] = susceptible_pagans_sum_zone_2
    y0_cp[19] = 0  # Exposed Pagans in Zone 2
    y0_cp[20] = 1  # Infected Pagans in Zone 2
    y0_cp[21] = 0  # Recovered Pagans in Zone 2
    # y0_cp[22] = 0  # Deceased Pagans in Zone 2 (set to 0 to see what CP does)

    # Zone 3
    # Reshuffling for Christians
    susceptible_christians_sum_zone_3 = sum(y0_cp[24:28])
    y0_cp[24] = susceptible_christians_sum_zone_3
    y0_cp[25] = 0  # Exposed Christians in Zone 3
    y0_cp[26] = 1  # Infected Christians in Zone 3
    y0_cp[27] = 0  # Recovered Christians in Zone 3
    # y0_cp[28] = 0  # Deceased Christians  in Zone 3 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_3 = sum(y0_cp[30:34])
    y0_cp[30] = susceptible_pagans_sum_zone_3
    y0_cp[31] = 0  # Exposed Pagans in Zone 3
    y0_cp[32] = 1  # Infected Pagans in Zone 3
    y0_cp[33] = 0  # Recovered Pagans in Zone 3
    # y0_cp[34] = 0  # Deceased Pagans in Zone 3 (set to 0 to see what CP does)

    # Zone 4
    # Reshuffling for Christians
    susceptible_christians_sum_zone_4 = sum(y0_cp[36:40])
    y0_cp[36] = susceptible_christians_sum_zone_4
    y0_cp[37] = 0  # Exposed Christians in Zone 4
    y0_cp[38] = 1  # Infected Christians in Zone 4
    y0_cp[39] = 0  # Recovered Christians in Zone 4
    # y0_cp[40] = 0  # Deceased Christians  in Zone 4 (set to 0 to see what CP does)

    # Reshuffling for Pagans
    susceptible_pagans_sum_zone_4 = sum(y0_cp[42:46])
    y0_cp[42] = susceptible_pagans_sum_zone_4
    y0_cp[43] = 0  # Exposed Pagans in Zone 4
    y0_cp[44] = 1  # Infected Pagans in Zone 4
    y0_cp[45] = 0  # Recovered Pagans in Zone 4
    # y0_cp[46] = 0  # Deceased Pagans in Zone 4 (set to 0 to see what CP does)

    # Interaction rates
    y0_cp[48] = zeleners_initial_delta_1  # delta_1_c
    y0_cp[49] = zeleners_initial_delta_1  # delta_1_p
    y0_cp[50] = zeleners_initial_delta_2  # delta_2_c
    y0_cp[51] = zeleners_initial_delta_2  # delta_2_p
    y0_cp[52] = zeleners_initial_delta_3  # delta_3_c
    y0_cp[53] = zeleners_initial_delta_3  # delta_3_p
    y0_cp[54] = zeleners_initial_delta_4  # delta_4_c
    y0_cp[55] = zeleners_initial_delta_4  # delta_4_p

    # Timeframe of simulation for the Cyprianic Plague model
    total_days_cp = (end_year_cp - end_year_demographic + 1) * 365
    t_cp = np.linspace(0, total_days_cp, total_days_cp + 1)

    # Solve the ODE for the Cyprianic Plague model
    def wrapper_for_solve_ivp_cp(t, y):
        return direct_transmission_with_four_deltas_two_cfrs_and_conversion_in_pairs_seird_model(
            y, t, measles_seir_params_with_lower_cfr_for_c_and_starks_conversion
        )

    print(y0_cp)
    solution_cp = solve_ivp(
        wrapper_for_solve_ivp_cp,
        [0, total_days_cp],
        y0_cp,
        method="RK45",
        t_eval=t_cp
    )

    # Solution indices and labels relevant to both Christian and Pagan compartments
    # (except for the A compartment, dead due to age and other natural causes).
    compartment_indices = [
        0, 1, 2, 3,  # Christians in Zone 1
        6, 7, 8, 9,  # Pagans in Zone 1
        12, 13, 14, 15,  # Christians in Zone 2
        18, 19, 20, 21,  # Pagans in Zone 2
        24, 25, 26, 27,  # Christians in Zone 3
        30, 31, 32, 33,  # Pagans in Zone 3
        36, 37, 38, 39,  # Christians in Zone 4
        42, 43, 44, 45  # Pagans in Zone 4
    ]
    compartment_labels = [
        # Zone 1
        "Susceptible Christians in Zone 1",
        "Exposed Christians in Zone 1",
        "Infected Christians in Zone 1",
        "Recovered Christians in Zone 1",
        # "Deceased Christians in Zone 1",

        "Susceptible Pagans in Zone 1",
        "Exposed Pagans in Zone 1",
        "Infected Pagans in Zone 1",
        "Recovered Pagans in Zone 1",
        # "Deceased Pagans in Zone 1",

        # Zone 2
        "Susceptible Christians in Zone 2",
        "Exposed Christians in Zone 2",
        "Infected Christians in Zone 2",
        "Recovered Christians in Zone 2",
        # "Deceased Christians in Zone 2",

        "Susceptible Pagans in Zone 2",
        "Exposed Pagans in Zone 2",
        "Infected Pagans in Zone 2",
        "Recovered Pagans in Zone 2",
        # "Deceased Pagans in Zone 2",

        # Zone 3
        "Susceptible Christians in Zone 3",
        "Exposed Christians in Zone 3",
        "Infected Christians in Zone 3",
        "Recovered Christians in Zone 3",
        # "Deceased Christians in Zone 3",

        "Susceptible Pagans in Zone 3",
        "Exposed Pagans in Zone 3",
        "Infected Pagans in Zone 3",
        "Recovered Pagans in Zone 3",
        # "Deceased Pagans in Zone 3",

        # Zone 4
        "Susceptible Christians in Zone 4",
        "Exposed Christians in Zone 4",
        "Infected Christians in Zone 4",
        "Recovered Christians in Zone 4",
        # "Deceased Christians in Zone 4",

        "Susceptible Pagans in Zone 4",
        "Exposed Pagans in Zone 4",
        "Infected Pagans in Zone 4",
        "Recovered Pagans in Zone 4",
        # "Deceased Pagans in Zone 4"
    ]
    plot_seir_model(
        solution_cp,
        t_cp,
        start_year=end_year_demographic,
        end_year=end_year_cp,
        compartment_indices=compartment_indices,
        compartment_labels=compartment_labels,
        every_nth_year=5,
        y_tick_interval=100_000,
        display_y_label_every_n_ticks=10,
        plot_title="CP in 4 separate zones as measles with smaller CFR for C and conversion in the whole Empire (after AP and demo)",
    )

    # Function to plot the totals of compartments
    def plot_total_compartments():
        total_compartment_labels = [
            "Susceptible Christians",
            "Exposed Christians",
            "Infected Christians",
            "Recovered Christians",
            "Deceased Christians",
            "Susceptible Pagans",
            "Exposed Pagans",
            "Infected Pagans",
            "Recovered Pagans",
            "Deceased Pagans",
        ]

        # Calculate the total compartments for each group
        susceptible_christians = solution_cp.y[0, :] + solution_cp.y[12, :] + solution_cp.y[24, :] + solution_cp.y[36, :]
        exposed_christians = solution_cp.y[1, :] + solution_cp.y[13, :] + solution_cp.y[25, :] + solution_cp.y[37, :]
        infected_christians = solution_cp.y[2, :] + solution_cp.y[14, :] + solution_cp.y[26, :] + solution_cp.y[38, :]
        recovered_christians = solution_cp.y[3, :] + solution_cp.y[15, :] + solution_cp.y[27, :] + solution_cp.y[39, :]
        deceased_christians = solution_cp.y[4, :] + solution_cp.y[16, :] + solution_cp.y[28, :] + solution_cp.y[40, :]

        susceptible_pagans = solution_cp.y[6, :] + solution_cp.y[18, :] + solution_cp.y[30, :] + solution_cp.y[42, :]
        exposed_pagans = solution_cp.y[7, :] + solution_cp.y[19, :] + solution_cp.y[31, :] + solution_cp.y[43, :]
        infected_pagans = solution_cp.y[8, :] + solution_cp.y[20, :] + solution_cp.y[32, :] + solution_cp.y[44, :]
        recovered_pagans = solution_cp.y[9, :] + solution_cp.y[21, :] + solution_cp.y[33, :] + solution_cp.y[45, :]
        deceased_pagans = solution_cp.y[10, :] + solution_cp.y[22, :] + solution_cp.y[34, :] + solution_cp.y[46, :]

        total_compartments = [
            susceptible_christians, exposed_christians, infected_christians, recovered_christians, deceased_christians,
            susceptible_pagans, exposed_pagans, infected_pagans, recovered_pagans, deceased_pagans
        ]

        # Plot the totals
        plot_seir_model(
            solution_cp,
            t_cp,
            start_year=end_year_demographic,
            end_year=end_year_cp,
            compartment_indices=range(len(total_compartments)),
            compartment_labels=total_compartment_labels,
            every_nth_year=5,
            y_tick_interval=100_000,
            display_y_label_every_n_ticks=10,
            plot_title="Total compartments across all zones",
        )

    # Call the function to plot totals
    plot_total_compartments()

    # Print the final values of each compartment for debugging purposes
    compartments_after_conversion = solution_cp.y[:, -1]

    # Zone 1
    s1_c = compartments_after_conversion[0]
    e1_c = compartments_after_conversion[1]
    i1_c = compartments_after_conversion[2]
    r1_c = compartments_after_conversion[3]
    d1_c = compartments_after_conversion[4]
    a1_c = compartments_after_conversion[5]
    s1_p = compartments_after_conversion[6]
    e1_p = compartments_after_conversion[7]
    i1_p = compartments_after_conversion[8]
    r1_p = compartments_after_conversion[9]
    d1_p = compartments_after_conversion[10]
    a1_p = compartments_after_conversion[11]

    # Zone 2
    s2_c = compartments_after_conversion[12]
    e2_c = compartments_after_conversion[13]
    i2_c = compartments_after_conversion[14]
    r2_c = compartments_after_conversion[15]
    d2_c = compartments_after_conversion[16]
    a2_c = compartments_after_conversion[17]
    s2_p = compartments_after_conversion[18]
    e2_p = compartments_after_conversion[19]
    i2_p = compartments_after_conversion[20]
    r2_p = compartments_after_conversion[21]
    d2_p = compartments_after_conversion[22]
    a2_p = compartments_after_conversion[23]

    # Zone 3
    s3_c = compartments_after_conversion[24]
    e3_c = compartments_after_conversion[25]
    i3_c = compartments_after_conversion[26]
    r3_c = compartments_after_conversion[27]
    d3_c = compartments_after_conversion[28]
    a3_c = compartments_after_conversion[29]
    s3_p = compartments_after_conversion[30]
    e3_p = compartments_after_conversion[31]
    i3_p = compartments_after_conversion[32]
    r3_p = compartments_after_conversion[33]
    d3_p = compartments_after_conversion[34]
    a3_p = compartments_after_conversion[35]

    # Zone 4
    s4_c = compartments_after_conversion[36]
    e4_c = compartments_after_conversion[37]
    i4_c = compartments_after_conversion[38]
    r4_c = compartments_after_conversion[39]
    d4_c = compartments_after_conversion[40]
    a4_c = compartments_after_conversion[41]
    s4_p = compartments_after_conversion[42]
    e4_p = compartments_after_conversion[43]
    i4_p = compartments_after_conversion[44]
    r4_p = compartments_after_conversion[45]
    d4_p = compartments_after_conversion[46]
    a4_p = compartments_after_conversion[47]

    # Summing alive individuals for Christians and Pagans in all zones
    alive1_c = s1_c + e1_c + i1_c + r1_c
    alive1_p = s1_p + e1_p + i1_p + r1_p

    alive2_c = s2_c + e2_c + i2_c + r2_c
    alive2_p = s2_p + e2_p + i2_p + r2_p

    alive3_c = s3_c + e3_c + i3_c + r3_c
    alive3_p = s3_p + e3_p + i3_p + r3_p

    alive4_c = s4_c + e4_c + i4_c + r4_c
    alive4_p = s4_p + e4_p + i4_p + r4_p

    alive_total = alive1_c + alive1_p + alive2_c + alive2_p + alive3_c + alive3_p + alive4_c + alive4_p
    alive_c = alive1_c + alive2_c + alive3_c + alive4_c
    alive_p = alive1_p + alive2_p + alive3_p + alive4_p
    alive_c_percentage_of_total = (alive_c / alive_total) * 100

    print(
        f"s1_c = {s1_c}\n"
        f"e1_c = {e1_c}\n"
        f"i1_c = {i1_c}\n"
        f"r1_c = {r1_c}\n"
        f"d1_c = {d1_c}\n"
        f"a1_c = {a1_c}\n"
        f"s1_p = {s1_p}\n"
        f"e1_p = {e1_p}\n"
        f"i1_p = {i1_p}\n"
        f"r1_p = {r1_p}\n"
        f"d1_p = {d1_p}\n"
        f"a1_p = {a1_p}\n"
        f"alive1_c = {alive1_c}\n"
        f"alive1_p = {alive1_p}\n"
        f"s2_c = {s2_c}\n"
        f"e2_c = {e2_c}\n"
        f"i2_c = {i2_c}\n"
        f"r2_c = {r2_c}\n"
        f"d2_c = {d2_c}\n"
        f"a2_c = {a2_c}\n"
        f"s2_p = {s2_p}\n"
        f"e2_p = {e2_p}\n"
        f"i2_p = {i2_p}\n"
        f"r2_p = {r2_p}\n"
        f"d2_p = {d2_p}\n"
        f"a2_p = {a2_p}\n"
        f"alive2_c = {alive2_c}\n"
        f"alive2_p = {alive2_p}\n"
        f"s3_c = {s3_c}\n"
        f"e3_c = {e3_c}\n"
        f"i3_c = {i3_c}\n"
        f"r3_c = {r3_c}\n"
        f"d3_c = {d3_c}\n"
        f"a3_c = {a3_c}\n"
        f"s3_p = {s3_p}\n"
        f"e3_p = {e3_p}\n"
        f"i3_p = {i3_p}\n"
        f"r3_p = {r3_p}\n"
        f"d3_p = {d3_p}\n"
        f"a3_p = {a3_p}\n"
        f"alive3_c = {alive3_c}\n"
        f"alive3_p = {alive3_p}\n"
        f"s4_c = {s4_c}\n"
        f"e4_c = {e4_c}\n"
        f"i4_c = {i4_c}\n"
        f"r4_c = {r4_c}\n"
        f"d4_c = {d4_c}\n"
        f"a4_c = {a4_c}\n"
        f"s4_p = {s4_p}\n"
        f"e4_p = {e4_p}\n"
        f"i4_p = {i4_p}\n"
        f"r4_p = {r4_p}\n"
        f"d4_p = {d4_p}\n"
        f"a4_p = {a4_p}\n"
        f"alive4_c = {alive4_c}\n"
        f"alive4_p = {alive4_p}\n"
        f"alive_total = {alive_total}\n"
        f"alive_c = {alive_c}\n"
        f"alive_p = {alive_p}\n"
        f"alive_c_percentage_of_total = {alive_c_percentage_of_total} %\n"
    )

    print("Length of t_cp:", len(t_cp))
    print("Length of solution_cp.t:", len(solution_cp.t))
    print("solution_cp.t[-1]:", solution_cp.t[-1])

    for i, label in enumerate(compartment_labels):
        max_val = np.max(solution_cp.y[compartment_indices[i], :])
        print(f"{label}: max = {max_val}")

    print("Solver success:", solution_cp.success)
    print("Solver message:", solution_cp.message)

    print(
        "final deltas:\n"
        f"delta_1_c = {y0_cp[48]}\n"
        f"delta_1_p = {y0_cp[49]}\n"
        f"delta_2_c = {y0_cp[50]}\n"
        f"delta_2_p = {y0_cp[51]}\n"
        f"delta_3_c = {y0_cp[52]}\n"
        f"delta_3_p = {y0_cp[53]}\n"
        f"delta_4_c = {y0_cp[54]}\n"
        f"delta_4_p = {y0_cp[55]}\n"
    )



if __name__ == "__main__":
    # proof_of_concept_solve_and_plot_ap_demo_cp_with_conversion_and_smaller_cfr_for_christians_in_four_separate_zones()
    # proof_of_concept_solve_and_plot_ap_demo_cp_with_conversion_and_smaller_cfr_for_christians_with_four_dynamic_deltas()
    # new_poc_solve_and_plot_ap_demo_cp_with_conversion_and_smaller_cfr_for_christians_with_four_deltas()
    poc_solve_and_plot_ap_demo_cp_with_converstion_to_denser_of_four_zones_and_smaller_cfr_for_christians()

