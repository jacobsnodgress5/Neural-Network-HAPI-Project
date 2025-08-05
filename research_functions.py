import matplotlib.pyplot as plt
import numpy as np
from hapi import *
from scipy.stats import linregress
import copy

def import_HAPI_data(molecule, HITRAN_molecule_number, HITRAN_isotopologue_number,wavenumber_start, wavenumber_end, molecular_weight):
    '''This function automates the extraction of the wavenumber data.
    PARAMETERS/Inputs: 
    --molecule: put molecule you want in quotes, abbreviated ex:H2O
    --HITRAN_molecule_number: input hitran's molecule number associated with the molecule 
    --HITRAN_isotopologue_number: similarly, input hitran's isotope number associated with the molecule
    --wavenumber_start: provide the lower bound of wavenumber data you want
    --wavenumber_end: Input upper bound of wavenumber data desired 
    --molecular_weight: provide molecular weight of the molecule in kg/mol

    RETURNS the log mass absorption coefficient of the molecule and wavenumber, in log_mass_absorption_coef and nu respectively, also providing a plot 
    
    '''
    db_begin('data')
    
    fetch(molecule, HITRAN_molecule_number, HITRAN_isotopologue_number, wavenumber_start, wavenumber_end)

    nu, coef = absorptionCoefficient_Lorentz(
        SourceTables=molecule,
        Diluent={'air': 1.0},
        Environment={'T': 260, 'p': 0.5}  # Pressure is in atm, so 500 hPa = 0.5 atm
    )

    # Avogadro's number in molecules per mole
    avogadro_number = 6.022e23

    # Convert absorption coefficient to mass absorption coefficient (m^2/kg)
    mass_absorption_coef=coef/10000
    mass_absorption_coef = mass_absorption_coef* avogadro_number
    mass_absorption_coef = mass_absorption_coef/molecular_weight

    #log_mass_absorption_coef = np.where(mass_absorption_coef > 0, np.log10(mass_absorption_coef), 0)

    log_mass_absorption_coef= np.log10(mass_absorption_coef)  #OLD LOG MASS CALC

    # Plot the results
    plt.plot(nu, log_mass_absorption_coef,  linewidth=0.1, label="Mass Absorption (log scale)")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Log(Mass Absorption Coefficient) (log(m²/kg))")
    plt.title("Mass Absorption Coefficient vs Wavenumber (for given molecule)")

    plt.show()

    return nu, log_mass_absorption_coef

#help function for Linear_Regression
def expand_range_overlap(range_tuple, nu_array): 
    start, end = range_tuple
    range_width = end - start
    expanded_start = max(nu_array[0], start - 0.05 * range_width)  # Do not go below nu[0]
    expanded_end = min(nu_array[-1], end + 0.05 * range_width)     # Do not exceed nu[-1]
    return expanded_start, expanded_end

#antoher help function for LR
def find_intersection(slope1, intercept1, slope2, intercept2):
    """Calculate x-coordinate of intersection between two lines."""
    if slope1 == slope2:  # Parallel lines do not intersect
        return None
    return (intercept2 - intercept1) / (slope1 - slope2)

def validIntersection(intersection, range1, range2, tolerance =0.20):
    "check if intersection in within 10% of the wavenumber ranges"
    start1, end1 = range1
    start2, end2 = range2

    tolerance1 = (end1 - start1) * tolerance
    tolerance2 = (end2 - start2) * tolerance

    if (start1 - tolerance1 <= intersection <= end1 + tolerance1) and \
       (start2 - tolerance2 <= intersection <= end2 + tolerance2):
        return True
    return False
    

def Plot_Linear_Regression(wavenumber_data, absorption_data, wavenumber_ranges):
    import copy
    # Expand wavenumber ranges
    extended_ranges = [expand_range_overlap(wr, wavenumber_data) for wr in wavenumber_ranges]

    expanded_band_indices = [(np.searchsorted(wavenumber_data, start), np.searchsorted(wavenumber_data, end)) for start, end in extended_ranges]

    regression_results = []
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumber_data, absorption_data, linewidth=0.2, color='yellow', label="Original Data")

    for i, (start, end) in enumerate(expanded_band_indices):
        x_band = wavenumber_data[start:end]
        y_band = absorption_data[start:end]
    
        # Perform linear regression
        slope, intercept, _, _, _ = linregress(x_band, y_band)
        regression_results.append((slope, intercept))  # Store regression results
    
        # Plot regression line
        plt.plot(
            x_band,
            slope * x_band + intercept,
            label=f'Band {i + 1} Fit (Overlap)',
            linewidth=2.0,
            color=f'C{i}'
        )

    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Log(Mass Absorption Coefficient) (log(m²/kg))")
    plt.title("Regression Lines with Overlapping Bands")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Records intersections if they are there
    intersections = []
    for i in range(len(regression_results) - 1):
        slope1, intercept1 = regression_results[i]
        slope2, intercept2 = regression_results[i + 1]
        intersection = find_intersection(slope1, intercept1, slope2, intercept2)
        if intersection is not None and validIntersection(intersection, wavenumber_ranges[i], wavenumber_ranges[i+1]):
            intersections.append(intersection)

    # Creating refined wavenumber ranges based on intersections
    refined_wavenumber_ranges = []
    copy_waves = copy.copy(wavenumber_ranges)
    count = 0
    for intersection in intersections:
        while count + 1 < len(copy_waves) and copy_waves[count][1] != copy_waves[count + 1][0]:
            refined_wavenumber_ranges.append(copy_waves[count])  # Add same range to refined ranges
            count += 1

        if count < len(copy_waves):  # Ensure we don't go out of bounds
            refined_wavenumber_ranges.append((copy_waves[count][0], intersection))

        if count + 1 < len(copy_waves):
            copy_waves[count + 1] = (intersection, copy_waves[count + 1][1])

        count += 1
    
    # Add remaining ranges without modification
    while count < len(copy_waves):
        refined_wavenumber_ranges.append(copy_waves[count])
        count += 1

    

    # Perform regression on refined ranges
    refined_band_indices = [(np.searchsorted(wavenumber_data, start), np.searchsorted(wavenumber_data, end)) for start, end in refined_wavenumber_ranges]
    
    refined_regression_results = []
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumber_data, absorption_data, linewidth=0.2, color='yellow', label="Original Data")

    for i, (start, end) in enumerate(refined_band_indices):
        x_band = wavenumber_data[start:end]
        y_band = absorption_data[start:end]
    
        # Perform linear regression on refined ranges
        slope, intercept, _, _, _ = linregress(x_band, y_band)
        refined_regression_results.append((slope, intercept))  # Store refined regression results
    
        # Plot regression line for refined range
        plt.plot(
            x_band,
            slope * x_band + intercept,
            label=f'Refined Band {i + 1} Fit',
            linewidth=2.0,
            linestyle='--',
            color=f'C{i}'
        )

    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Log(Mass Absorption Coefficient) (log(m²/kg))")
    plt.title("Refined Regression Lines with Overlapping Bands")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    rounded_intersections = [round(num,2) for num in intersections]
    rounded_refined_ranges = [(round(x,2), round(y,2)) for x,y in refined_wavenumber_ranges]
    print("Rounded Intersections: ", rounded_intersections)
    print("Rounded Refined Ranges: ", rounded_refined_ranges)

    for (start, end), (slope, intercept) in zip(refined_wavenumber_ranges, refined_regression_results):
        print("Start: ", start, " Slope: ", slope, " End: ", end, " Intercept: ",intercept)

    return refined_wavenumber_ranges,refined_regression_results

def regressionError(wavenumber_data, coef_data, band_indices, regession_lines, bin_size =50):

    bin_edges = np.arange(wavenumber_data[0], wavenumber_data[-1] + bin_size, bin_size)

    # Compute binned averages for the original data
    k_ref = []
    bin_centers = []
    for i in range(len(bin_edges) - 1):
        bin_indices = (wavenumber_data >= bin_edges[i]) & (wavenumber_data < bin_edges[i + 1]) #gets wavenumber values that are within the bins
        if np.any(bin_indices):
            k_ref.append(10**(np.mean(coef_data[bin_indices]))) #appends the inverse logarithm data values to k_ref
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2) #lisr/array of bin centers

    k_ref = np.array(k_ref) #changes the lists into arrays
    bin_centers = np.array(bin_centers)

    # Step 7: Compute regression predictions for each bin center
    k_lla = []  #redo using bin center avgs, similar to Kref, 
    for center in bin_centers:
        regression_found =False
        for (start, end), (slope, intercept) in zip(band_indices, regession_lines): #refined_band_indices is the refined wavenumber ranges from the plot
            #regession results refined is a list of all the slopes and intercepts of the refined regression lines from the plots
            
            if start <= center < end:
                #print(f"Center: {center}, Slope: {slope}, Intercept: {intercept}")
                log_k_lla = slope * center + intercept
                k_lla.append(10**(log_k_lla))
                regression_found =True
                break
        if not regression_found:
            k_lla.append(0)  

    k_lla = np.array(k_lla)

    # Step 8: Compute deviation
    #log_k_ref = np.log10(k_ref)
    #log_k_lla= np.log10(k_lla)
    deviation= []
    for krVal, klVal in zip(k_ref,k_lla):
        if klVal ==0:
            deviation.append(0)
        else:
            deviation.append((krVal - klVal) / klVal)

    deviation = np.array(deviation)

    # Step 9: Plot deviation
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, deviation, marker='o', label="Deviation", color="red")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Deviation")
    plt.title("Deviation as a Function of Wavenumber")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return k_ref, k_lla, deviation

def Linear_Regression(wavenumber_data, absorption_data, wavenumber_ranges):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import linregress
    #expand wavenumber ranges
    #perform regression using the expansions 
    #plot original 
    extended_ranges = [expand_range_overlap(wr,wavenumber_data) for wr in wavenumber_ranges]

    expanded_band_indices = [(np.searchsorted(wavenumber_data, start), np.searchsorted(wavenumber_data, end)) for start, end in extended_ranges]

    regression_results = []
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumber_data, absorption_data, linewidth=0.2, color='yellow', label="Original Data")

    for i, (start, end) in enumerate(expanded_band_indices):
        x_band = wavenumber_data[start:end]
        y_band = absorption_data[start:end]
    
        # Perform linear regression
        slope, intercept, _, _, _ = linregress(x_band, y_band)
        regression_results.append((slope, intercept))
    
        # Plot regression line
        plt.plot(
            x_band,
            slope * x_band + intercept,
            linewidth=2.0,
            color=f'C{i}'
        )

    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Log(Mass Absorption Coefficient) (log(m²/kg))")
    plt.title("Regression Lines with Overlapping Bands")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
