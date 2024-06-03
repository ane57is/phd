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


class BaseModelParameters:
    def __init__(
        self,
        beta,  # transmission rate
        sigma,  # incubation period
        gamma,  # infectious period
        natural_birth_rate,
        natural_death_rate,
        fatality_rate=None,
        fatality_rate_p=None,
        fatality_rate_c=None,
        conversion_rate_decennial=None,
    ):
        # Disease-specific parameters
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma

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
    beta=1.175, sigma=1 / 10, gamma=1 / 13.5, fatality_rate_p=0.3, conversion_rate_decennial=0.4, **demographic_params
)
smallpox_seir_params_with_starks_conversion = BaseModelParameters(
    beta=0.584, sigma=1 / 12, gamma=1 / 9.5, fatality_rate_p=0.9, conversion_rate_decennial=0.4, **demographic_params
)
