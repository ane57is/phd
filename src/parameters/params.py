initial_christian_population = 40_000
initial_pagan_population = 60_000_000

# Demographic parameters and initial populations shared across models
demographic_params = {
    'natural_birth_rate': 1 / (25 * 365),
    'natural_death_rate': 1 / (25 * 365)
}


class BaseModelParameters():
    def __init__(self, beta, sigma, gamma, fatality_rate, natural_birth_rate, natural_death_rate):
        # Disease-specific parameters
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.fatality_rate = fatality_rate

        # Demographic parameters of humans
        self.natural_birth_rate = natural_birth_rate
        self.natural_death_rate = natural_death_rate


# Instantiation of the parameters (using smallpox values for default)
default_seir_params = BaseModelParameters(beta=0.584, sigma=1/12, gamma=1/9.5, fatality_rate=0.9, **demographic_params)
# smallpox_seir_params = SmallpoxSEIRParams(beta=0.4, sigma=0.1, gamma=0.05, fatality_rate=0.3, **demographic_params)
