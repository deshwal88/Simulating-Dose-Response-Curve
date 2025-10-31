import functions
import numpy as np

def main():
    # parameters
    top_concentration = 100
    dilution_factor = 5

    # generate concentration series: 100, 100/5, 100/5^2, ...
    points_c40 = functions.calculate_dilution_scheme(0.4*top_concentration, dilution_factor)
    c_40 = np.linspace(points_c40[0], points_c40[-1], 50)
    points_c100 = functions.calculate_dilution_scheme(top_concentration, dilution_factor)
    c_100 = np.linspace(points_c100[0], points_c100[-1], 50)
    points_c160 = functions.calculate_dilution_scheme(1.6*top_concentration, dilution_factor)
    c_160 = np.linspace(points_c160[0], points_c160[-1], 50)

    #generating response curves for 40, 100 and 160 uM top concentrations
    res_40 = functions.compute_response_curve(0.5, 1.5, 2, 3.5, c_40)
    res_100 = functions.compute_response_curve(0.5, 1.5, 5, 3.5, c_100)
    res_160 = functions.compute_response_curve(0.5, 1.5, 8, 3.5, c_160)

    functions.generate_main_graph([c_40, c_100, c_160], [res_40, res_100, res_160])
    return

if __name__ == "__main__":
    main()