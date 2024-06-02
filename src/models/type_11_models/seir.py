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
    rate mechanism.
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
    ds1_c = b * n1_c - d * s1_c + converted_s1
    de1_c = -d * e1_c + converted_e1
    di1_c = -d * i1_c + converted_i1
    dr1_c = -d * r1_c + converted_r1
    dd1_c = 0
    da1_c = d * n1_c

    # Pagan compartments in Zone 1
    ds1_p = b * n1_p - d * s1_p - converted_s1
    de1_p = -d * e1_p - converted_e1
    di1_p = -d * i1_p - converted_i1
    dr1_p = -d * r1_p - converted_r1
    dd1_p = 0
    da1_p = d * n1_p

    # Zone 2
    # Christian compartments in Zone 2
    ds2_c = b * n2_c - d * s2_c + converted_s2
    de2_c = -d * e2_c + converted_e2
    di2_c = -d * i2_c + converted_i2
    dr2_c = -d * r2_c + converted_r2
    dd2_c = 0
    da2_c = d * n2_c

    # Pagan compartments in Zone 2
    ds2_p = b * n2_p - d * s2_p - converted_s2
    de2_p = -d * e2_p - converted_e2
    di2_p = -d * i2_p - converted_i2
    dr2_p = -d * r2_p - converted_r2
    dd2_p = 0
    da2_p = d * n2_p

    # Zone 3
    # Christian compartments in Zone 3
    ds3_c = b * n3_c - d * s3_c + converted_s3
    de3_c = -d * e3_c + converted_e3
    di3_c = -d * i3_c + converted_i3
    dr3_c = -d * r3_c + converted_r3
    dd3_c = 0
    da3_c = d * n3_c

    # Pagan compartments in Zone 3
    ds3_p = b * n3_p - d * s3_p - converted_s3
    de3_p = -d * e3_p - converted_e3
    di3_p = -d * i3_p - converted_i3
    dr3_p = -d * r3_p - converted_r3
    dd3_p = 0
    da3_p = d * n3_p

    # Zone 4
    # Christian compartments in Zone 4
    ds4_c = b * n4_c - d * s4_c + converted_s4
    de4_c = -d * e4_c + converted_e4
    di4_c = -d * i4_c + converted_i4
    dr4_c = -d * r4_c + converted_r4
    dd4_c = 0
    da4_c = d * n4_c

    # Pagan compartments in Zone 4
    ds4_p = b * n4_p - d * s4_p - converted_s4
    de4_p = -d * e4_p - converted_e4
    di4_p = -d * i4_p - converted_i4
    dr4_p = -d * r4_p - converted_r4
    dd4_p = 0
    da4_p = d * n4_p

    # Return the derivatives in the same order
    return [
        ds1_c, de1_c, di1_c, dr1_c, dd1_c, da1_c, ds1_p, de1_p, di1_p, dr1_p, dd1_p, da1_p,
        ds2_c, de2_c, di2_c, dr2_c, dd2_c, da2_c, ds2_p, de2_p, di2_p, dr2_p, dd2_p, da2_p,
        ds3_c, de3_c, di3_c, dr3_c, dd3_c, da3_c, ds3_p, de3_p, di3_p, dr3_p, dd3_p, da3_p,
        ds4_c, de4_c, di4_c, dr4_c, dd4_c, da4_c, ds4_p, de4_p, di4_p, dr4_p, dd4_p, da4_p
    ]
