from typing import List, Sequence, Tuple, Optional
import numpy as np

__all__ = [
    "calculate_even_dilution",
    "calculate_custom_dilution",
    "compute_4pl",
    "edge_case_parameter_sets",
    "recommend_even_dilution_factors"
]

# ---------------------------------------
# Dilution Generators
# ---------------------------------------

def calculate_even_dilution(top_concentration: float, dilution_factor: float, points: int = 8) -> List[float]:
    """Generate an even dilution series. Example: 100, 20, 4..."""
    if top_concentration <= 0:
        raise ValueError("top_concentration must be positive")
    if dilution_factor <= 0:
        raise ValueError("dilution_factor must be positive")
    return [top_concentration / (dilution_factor ** i) for i in range(points)]


def calculate_custom_dilution(top_concentration: float, factors: Sequence[float]) -> List[float]:
    """Generate an 8-point dilution using 7 sequential factors (e.g. 3,3,2,2,2,3,3)."""
    if len(factors) != 7:
        raise ValueError("factors must contain exactly 7 values")
    series=[float(top_concentration)]
    for f in factors:
        if f<=0: raise ValueError("Dilution factors must be positive.")
        series.append(series[-1]/f)
    return series

# ---------------------------------------
# 4PL Core Function
# ---------------------------------------

def compute_4pl(A: float, B: float, C: float, D: float, x: Sequence[float], noise_sd: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
    """Return 4PL model response for array x."""
    x = np.asarray(x, dtype=float)
    x = np.where(x<=0, np.finfo(float).tiny, x)
    y = D + (A-D) / (1 + (x/C)**B)
    if noise_sd>0:
        rng=np.random.default_rng(seed)
        y+=rng.normal(0,noise_sd,size=y.shape)
    return y

# ---------------------------------------
# Edge Case Combination Generator
# ---------------------------------------

def edge_case_parameter_sets(A,B,C,D):
    A_min,A_max=A; B_min,B_max=B; C_min,C_max=C; D_min,D_max=D
    return [(a,b,c,d) for a in(A_min,A_max)
                    for b in(B_min,B_max)
                    for c in(C_min,C_max)
                    for d in(D_min,D_max)]

# ---------------------------------------
# Dilution Recommendation Engine
# ---------------------------------------

def _classify_concentration(x, C):
    ratio=x/C
    if ratio<=0.1: return "lower"
    if 0.3<=ratio<=3: return "linear"
    if ratio>10: return "upper"
    return "middle"

def _evaluate(conc_list, C):
    linear=lower=upper=0
    for x in conc_list:
        region=_classify_concentration(x,C)
        if region=="linear": linear+=1
        elif region=="lower": lower+=1
        elif region=="upper": upper+=1
    return linear,lower,upper

def recommend_even_dilution_factors(top_conc, C_min, C_max, points=8):
    """
    PRACTICAL MODE:
    Returns dilution schemes that work for C_min OR C_max (not both)
    This ensures recommendations appear even when EC50 range is wide.
    """
    candidate_factors = np.arange(2, 12, 0.5)       # wider search
    valid = []
    print("\n--- RUNNING RECOMMENDER ---")
    print("Inputs:", top_conc, C_min, C_max)

    for f in candidate_factors:
        concs = calculate_even_dilution(top_conc, f, points)

        ok = False
        for C in (C_min, C_max):
            linear, lower, upper = _evaluate(concs, C)

            # Loosen threshold to generate visible results
            if (linear >= 3 and lower >= 1 and upper >= 1):
                ok = True
                break

        if ok:
            valid.append((round(float(f),2), concs))

    print("VALID:", valid)   # optional debug
    return valid


# -------- Debug Test ------- #

if __name__=="__main__":
    print("RECOMMENDER TEST")
    recs=recommend_even_dilution_factors(400,2,8)
    print(recs)
