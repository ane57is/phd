# Demographic parameters and initial populations shared across models
demographic_params = {
    "natural_birth_rate": 1 / (25 * 365),
    "natural_death_rate": 1 / (25 * 365),
}

# Total initial population
initial_christian_population = 40_000
initial_pagan_population = 150_000_000

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
    "initial_christian_population_zone_1": (initial_pagan_population / 6) * 0.0006667,
    "initial_pagan_population_zone_1": (initial_pagan_population / 6),
    "initial_christian_population_zone_2": (initial_pagan_population / 6) * 0.0013333,
    "initial_pagan_population_zone_2": (initial_pagan_population / 3),
    "initial_christian_population_zone_3": (initial_pagan_population / 6) * 0.0013333,
    "initial_pagan_population_zone_3": (initial_pagan_population / 3),
    "initial_christian_population_zone_4": (initial_pagan_population / 6) * 0.0006667,
    "initial_pagan_population_zone_4": (initial_pagan_population / 6),
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

# Smallpox SEIR parameters with different configurations:
# - with conversion and with hardcoded lower CFR
# - without conversion and with hardcoded lower CFR
# - with conversion and with CFR based on literature
# - without conversion and with CFR based on literature
smallpox_seir_params_with_starks_conversion_and_with_hardcoded_lower_cfr = BaseModelParameters(
    beta=0.584,
    sigma=1 / 12,
    gamma=1 / 9.5,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.95,
    conversion_rate_decennial=0.4,
    **demographic_params
)
smallpox_seir_params_without_conversion_and_with_hardcoded_lower_cfr = BaseModelParameters(
    beta=0.584,
    sigma=1 / 12,
    gamma=1 / 9.5,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.95,
    conversion_rate_decennial=0,
    **demographic_params
)
smallpox_seir_params_with_starks_conversion_and_2_cfrs_from_literature = BaseModelParameters(
    beta=0.584,
    sigma=1 / 12,
    gamma=1 / 9.5,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.95,
    fatality_rate_c=0.3,  # lower CFR based on literature
    conversion_rate_decennial=0.4,
    **demographic_params
)
smallpox_seir_params_without_conversion_and_2_cfrs_from_literature = BaseModelParameters(
    beta=0.584,
    sigma=1 / 12,
    gamma=1 / 9.5,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.95,
    fatality_rate_c=0.3,  # lower CFR based on literature
    conversion_rate_decennial=0,
    **demographic_params
)

# Measles SEIR parameters with different configurations:
# - with conversion and with hardcoded lower CFR
# - without conversion and with hardcoded lower CFR
# - with conversion and with CFR based on literature
# - without conversion and with CFR based on literature
measles_seir_params_with_starks_conversion_and_with_hardcoded_lower_cfr = BaseModelParameters(
    beta=1.175,
    sigma=1 / 10,
    gamma=1 / 13.5,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.34,
    conversion_rate_decennial=0.4,
    **demographic_params
)
measles_seir_params_without_conversion_and_with_hardcoded_lower_cfr = BaseModelParameters(
    beta=1.175,
    sigma=1 / 10,
    gamma=1 / 13.5,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.34,
    conversion_rate_decennial=0,
    **demographic_params
)
measles_seir_params_with_starks_conversion_and_2_cfrs_from_literature = BaseModelParameters(
    beta=1.175,
    sigma=1 / 10,
    gamma=1 / 13.5,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.34,
    fatality_rate_c=0.03,  # lower CFR based on literature
    conversion_rate_decennial=0.4,
    **demographic_params
)
measles_seir_params_without_conversion_and_2_cfrs_from_literature = BaseModelParameters(
    beta=1.175,
    sigma=1 / 10,
    gamma=1 / 13.5,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.34,
    fatality_rate_c=0.03,  # lower CFR based on literature
    conversion_rate_decennial=0,
    **demographic_params
)

# Crimean–Congo hemorrhagic fever (CCHf) SEIR parameters with different configurations:
# - with conversion and with hardcoded lower CFR
# - without conversion and with hardcoded lower CFR
# - with conversion and with CFR based on literature
# - without conversion and with CFR based on literature
cchf_seir_params_with_starks_conversion_and_with_hardcoded_lower_cfr = BaseModelParameters(
    beta=0.5,
    sigma=1 / 9,
    gamma=1 / 3,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.4,  # CFR based on literature, lower CFR 3 times less than for Christians
    conversion_rate_decennial=0.4,
    **demographic_params
)
cchf_seir_params_without_conversion_and_with_hardcoded_lower_cfr = BaseModelParameters(
    beta=0.5,
    sigma=1 / 9,
    gamma=1 / 3,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.4,  # CFR based on literature, lower CFR 3 times less than for Christians
    conversion_rate_decennial=0,
    **demographic_params
)
cchf_seir_params_with_starks_conversion_and_2_cfrs_from_literature = BaseModelParameters(
    beta=0.5,
    sigma=1 / 9,
    gamma=1 / 3,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.4,  # CFR based on literature
    fatality_rate_c=0.1,  # CFR based on literature
    conversion_rate_decennial=0.4,
    **demographic_params
)
cchf_seir_params_without_conversion_and_2_cfrs_from_literature = BaseModelParameters(
    beta=0.5,
    sigma=1 / 9,
    gamma=1 / 3,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.4,  # CFR based on literature
    fatality_rate_c=0.1,  # CFR based on literature
    conversion_rate_decennial=0,
    **demographic_params
)

# Ebola virus disease (EVD) SEIR parameters with different configurations:
# - with conversion and with hardcoded lower CFR
# - without conversion and with hardcoded lower CFR
# - with conversion and with CFR based on literature
# - without conversion and with CFR based on literature
evd_seir_params_with_starks_conversion_and_with_hardcoded_lower_cfr = BaseModelParameters(
    beta=0.243,
    sigma=1 / 12.7,
    gamma=1 / 10.1,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.89,  # CFR based on literature, lower CFR 3 times less than for Christians
    conversion_rate_decennial=0.4,
    **demographic_params
)
evd_seir_params_without_conversion_and_with_hardcoded_lower_cfr = BaseModelParameters(
    beta=0.243,
    sigma=1 / 12.7,
    gamma=1 / 10.1,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.89,  # CFR based on literature, lower CFR 3 times less than for Christians
    conversion_rate_decennial=0,
    **demographic_params
)
evd_seir_params_with_starks_conversion_and_2_cfrs_from_literature = BaseModelParameters(
    beta=0.243,
    sigma=1 / 12.7,
    gamma=1 / 10.1,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.89,  # CFR based on literature
    fatality_rate_c=0.41,  # CFR based on literature
    conversion_rate_decennial=0.4,
    **demographic_params
)
evd_seir_params_without_conversion_and_2_cfrs_from_literature = BaseModelParameters(
    beta=0.243,
    sigma=1 / 12.7,
    gamma=1 / 10.1,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.89,  # CFR based on literature
    fatality_rate_c=0.41,  # CFR based on literature
    conversion_rate_decennial=0,
    **demographic_params
)

# Lassa fever SEIR parameters with different configurations:
# - with conversion and with hardcoded lower CFR
# - without conversion and with hardcoded lower CFR
# - with conversion and with CFR based on literature
# - without conversion and with CFR based on literature
lassa_seir_params_with_starks_conversion_and_with_hardcoded_lower_cfr = BaseModelParameters(
    beta=0.001085,
    sigma=1 / 10.25,
    gamma=1 / 14,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.33,  # CFR based on literature, lower CFR 3 times less than for Christians
    conversion_rate_decennial=0.4,
    **demographic_params
)
lassa_seir_params_without_conversion_and_with_hardcoded_lower_cfr = BaseModelParameters(
    beta=0.001085,
    sigma=1 / 10.25,
    gamma=1 / 14,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.33,  # CFR based on literature, lower CFR 3 times less than for Christians
    conversion_rate_decennial=0,
    **demographic_params
)
lassa_seir_params_with_starks_conversion_and_2_cfrs_from_literature = BaseModelParameters(
    beta=0.001085,
    sigma=1 / 10.25,
    gamma=1 / 14,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.33,  # CFR based on literature
    fatality_rate_c=0.0005,  # CFR based on literature
    conversion_rate_decennial=0.4,
    **demographic_params
)
lassa_seir_params_without_conversion_and_2_cfrs_from_literature = BaseModelParameters(
    beta=0.001085,
    sigma=1 / 10.25,
    gamma=1 / 14,
    initial_delta_1=zeleners_initial_delta_1,
    initial_delta_2=zeleners_initial_delta_2,
    initial_delta_3=zeleners_initial_delta_3,
    initial_delta_4=zeleners_initial_delta_4,
    max_delta=12,
    unified_deltas=True,
    fatality_rate_p=0.33,  # CFR based on literature
    fatality_rate_c=0.0005,  # CFR based on literature
    conversion_rate_decennial=0,
    **demographic_params
)

# Grouped parameters
smallpox_param_sets = {
    "with_conversion_hardcoded_cfr": smallpox_seir_params_with_starks_conversion_and_with_hardcoded_lower_cfr,
    "without_conversion_hardcoded_cfr": smallpox_seir_params_without_conversion_and_with_hardcoded_lower_cfr,
    "with_conversion_literature_cfr": smallpox_seir_params_with_starks_conversion_and_2_cfrs_from_literature,
    "without_conversion_literature_cfr": smallpox_seir_params_without_conversion_and_2_cfrs_from_literature,
}

measles_param_sets = {
    "with_conversion_hardcoded_cfr": measles_seir_params_with_starks_conversion_and_with_hardcoded_lower_cfr,
    "without_conversion_hardcoded_cfr": measles_seir_params_without_conversion_and_with_hardcoded_lower_cfr,
    "with_conversion_literature_cfr": measles_seir_params_with_starks_conversion_and_2_cfrs_from_literature,
    "without_conversion_literature_cfr": measles_seir_params_without_conversion_and_2_cfrs_from_literature,
}

cchf_param_sets = {
    "with_conversion_hardcoded_cfr": cchf_seir_params_with_starks_conversion_and_with_hardcoded_lower_cfr,
    "without_conversion_hardcoded_cfr": cchf_seir_params_without_conversion_and_with_hardcoded_lower_cfr,
    "with_conversion_literature_cfr": cchf_seir_params_with_starks_conversion_and_2_cfrs_from_literature,
    "without_conversion_literature_cfr": cchf_seir_params_without_conversion_and_2_cfrs_from_literature,
}

evd_param_sets = {
    "with_conversion_hardcoded_cfr": evd_seir_params_with_starks_conversion_and_with_hardcoded_lower_cfr,
    "without_conversion_hardcoded_cfr": evd_seir_params_without_conversion_and_with_hardcoded_lower_cfr,
    "with_conversion_literature_cfr": evd_seir_params_with_starks_conversion_and_2_cfrs_from_literature,
    "without_conversion_literature_cfr": evd_seir_params_without_conversion_and_2_cfrs_from_literature,
}

lassa_param_sets = {
    "with_conversion_hardcoded_cfr": lassa_seir_params_with_starks_conversion_and_with_hardcoded_lower_cfr,
    "without_conversion_hardcoded_cfr": lassa_seir_params_without_conversion_and_with_hardcoded_lower_cfr,
    "with_conversion_literature_cfr": lassa_seir_params_with_starks_conversion_and_2_cfrs_from_literature,
    "without_conversion_literature_cfr": lassa_seir_params_without_conversion_and_2_cfrs_from_literature,
}
