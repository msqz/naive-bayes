from math import pi, exp


def gaussian(x, mu, std):
    exponent = exp(-((x - mu)**2) / (2*(std**2)))
    denominator = 2*pi*(std**2)

    return exponent / denominator
