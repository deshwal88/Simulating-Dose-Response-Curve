from typing import List, Sequence, Tuple, Optional
import numpy as np

__all__ = [
    "calculate_even_dilution",
    "calculate_custom_dilution",
    "compute_4pl",
    "edge_case_parameter_sets",
]


def calculate_even_dilution(top_concentration: float, dilution_factor: float, points: int = 8) -> List[float]:
    """Return a list of `points` concentrations for an even dilution scheme.

    Example: top_concentration=100, dilution_factor=5 => [100, 20, 4, ...]
    """
    if top_concentration <= 0:
        raise ValueError("top_concentration must be positive")
    if dilution_factor <= 0:
        raise ValueError("dilution_factor must be positive")
    concentrations = [top_concentration / (dilution_factor ** i) for i in range(points)]
    return concentrations


def calculate_custom_dilution(top_concentration: float, factors: Sequence[float]) -> List[float]:
    """Return an 8-point concentration series from `top_concentration` using
    a sequence of 7 dilution factors applied sequentially.

    `factors` must have length 7. Each element is a positive number representing
    the factor by which the previous concentration is divided.
    """
    if len(factors) != 7:
        raise ValueError("factors must be a sequence of 7 positive numbers")
    c = float(top_concentration)
    series = [c]
    for f in factors:
        if f <= 0:
            raise ValueError("all dilution factors must be positive")
        c = c / f
        series.append(c)
    return series


def compute_4pl(A: float, B: float, C: float, D: float, x: Sequence[float], noise_sd: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
    """Compute 4-parameter logistic (4PL) responses for concentrations x.

    Model used: y = D + (A - D) / (1 + (X/C)**B) + epsilon

    - A: lower asymptote
    - B: slope/steepness
    - C: EC50 (concentration producing 50% response)
    - D: upper asymptote
    - x: sequence of concentrations (linear scale)
    - noise_sd: standard deviation of additive Gaussian noise
    """
    x_arr = np.asarray(x, dtype=float)
    if np.any(x_arr <= 0):
        # concentrations must be positive for the (X/C)**B term when using real powers
        # but model can accept >0; warn or handle by small offset if zeros appear.
        x_arr = np.where(x_arr <= 0, np.finfo(float).tiny, x_arr)
    # core 4PL model
    denom = 1.0 + (x_arr / float(C)) ** float(B)
    y = D + (A - D) / denom
    if noise_sd and noise_sd > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(loc=0.0, scale=float(noise_sd), size=y.shape)
    return y


def edge_case_parameter_sets(A_range: Tuple[float, float], B_range: Tuple[float, float], C_range: Tuple[float, float], D_range: Tuple[float, float]) -> List[Tuple[float, float, float, float]]:
    """Return the 16 edge-case parameter combinations from min/max of each range.

    Each range is provided as (min, max). The function returns a list of
    (A,B,C,D) tuples in a deterministic order (binary counting order).
    """
    A_min, A_max = A_range
    B_min, B_max = B_range
    C_min, C_max = C_range
    D_min, D_max = D_range
    combos = []
    for a_choice in (A_min, A_max):
        for b_choice in (B_min, B_max):
            for c_choice in (C_min, C_max):
                for d_choice in (D_min, D_max):
                    combos.append((a_choice, b_choice, c_choice, d_choice))
    return combos

def _classify_concentration(x: float, C: float) -> str:
    """
    Classify a concentration x relative to EC50 (C) for 4PL:
    Regions:
      - lower asymptote: x < 0.01*C
      - linear region:   0.2*C <= x <= 5*C
      - upper asymptote: x > 100*C
      - middle: ignored
    """
    ratio = x / C

    if ratio < 0.01:
        return "lower"
    elif 0.2 <= ratio <= 5:
        return "linear"
    elif ratio > 100:
        return "upper"
    else:
        return "middle"
    
def _evaluate_dilution_scheme(concentrations, C: float):
    """
    Count points in each important region for Task 4 requirements.
    Returns:
      (linear_count, lower_count, upper_count)
    """
    linear = lower = upper = 0

    for x in concentrations:
        region = _classify_concentration(x, C)
        if region == "linear":
            linear += 1
        elif region == "lower":
            lower += 1
        elif region == "upper":
            upper += 1

    return linear, lower, upper

def recommend_even_dilution_factors(top_concentration: float, C: float, factors_list=(2, 3, 4, 5, 6, 8, 10), points: int = 8):
    """
    Recommend dilution factors (even schemes like 3-fold, 5-fold) that satisfy:

       - >=3 points in linear region
       - >=1 point in lower asymptote
       - >=1 point in upper asymptote

    Returns list of tuples:
        (dilution_factor, concentration_series)
    """
    recommendations = []

    for f in factors_list:
        series = calculate_even_dilution(top_concentration, f, points)
        linear, lower, upper = _evaluate_dilution_scheme(series, C)

        if linear >= 3 and lower >= 1 and upper >= 1:
            recommendations.append((f, series))

    return recommendations


if __name__ == "__main__":
    print("This module provides the compute_response_curve function for import.")