import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from util.catalog_helper_functions import get_matches
from astropy.table import Table
from tqdm import tqdm

FITS = '~/radiative/fits/jwst_init_NRCB.fits'

class DataSelection(): 
    """
    Samples stars from fits catalog and calculates rough effective temperatures 
    and luminosities for each star by fitting a blackbody to band-luminosities. 

    Args: 
        * dist (float)    : The distance to the starcluster in cm
        * num_stars (int) : The number of stars you wish to sample 
        * filt1/2name (str) : the names of the two filts to match for 
                            the first star catalog subset  
        * num_bins (int)  : The number of bins across x-space to sample from 
        * fits (str)      : The directory of the fits catalog 

    """

    def __init__(
        self, 
        dist, 
        num_stars = 1000, 
        num_bins = 10, 
        filt1name='F115W', 
        filt2name='F212N',
        fits= FITS, all=False,
    ): 
        self.catalog   = Table.read(fits)
        self.filt1name = filt1name 
        self.filt2name = filt2name 
        self.dist = dist 
        self.num_stars = num_stars
        self.num_bins  = num_bins
        self.all = all
        return 


    def data(self, region='NRCB1'): 
        # Performs matching with filt1 and filt2 
        coord1, _, m1, _, _, _ = get_matches(
            self.catalog, 
            self.filt1name, region, 
            self.filt2name, region, 
        ) 

        # Choose filt1 for the x,y coord positions 
        # This doesn't really matter, the stars are the same 
        main_coord = coord1  # main_coord[0] = x positions, main_coord[1] = y positions
        main_mag   = m1 

        min_x, max_x = min(main_coord[0]), max(main_coord[0])

        # Evenly spaced bins 
        bin_edges = np.linspace(min_x, max_x, self.num_bins + 1)  

        data = pd.DataFrame()

        for i in range(len(bin_edges) - 1):  
            x_start, x_end = bin_edges[i], bin_edges[i + 1]

            x_mask = (main_coord[0] >= x_start) & (main_coord[0] < x_end)
            binned_coords = main_coord[:, x_mask]  
            binned_mags = main_mag[x_mask]  

            num_available_stars = binned_coords.shape[1]

            if not self.all:
                num_to_select = min(num_available_stars, self.num_stars // self.num_bins)
            else: 
                num_to_select = num_available_stars 

            if num_to_select > 0:
                chosen_star_idxs = np.random.choice(num_available_stars, num_to_select, replace=False)

                # Chosen star magnitudes in filt1 
                m1_mags = binned_mags[chosen_star_idxs]

                f115w_idx = 0 
                if region == 'NRCB2': 
                    f115w_idx = 1 
                if region == 'NRCB3': 
                    f115w_idx = 2 
                if region == 'NRCB4': 
                    f115w_idx = 3   

                # Extract other wavelength's identical star magnitudes
                catalog_idxs = np.where( 
                    np.isin(
                        self.catalog['m'], 
                        m1_mags, 
                    )
                )[0]
                catalog_mags = self.catalog[catalog_idxs]['m']
                catalog_x    = self.catalog[catalog_idxs]['x'][:, f115w_idx] 
                catalog_y    = self.catalog[catalog_idxs]['y'][:, f115w_idx]

                # Magnitude dict for each wavelength 
                mags = {
                    'mF115W': catalog_mags[:, f115w_idx], 
                    'mF212N': catalog_mags[:, f115w_idx+4], 
                    'mF323N': catalog_mags[:, 9], 
                    'mF405N': catalog_mags[:, 8], 
                } 

                def mag_to_lum(mag, filter): # erg s^-1 cm^-2 Hz^-1
                    # Flux zeropoints for NRCB1 
                    # All NRCBs are close, this is preliminary 
                    F115W_ZP = 1749.22
                    F212N_ZP = 675.30 
                    F323N_ZP = 320.86 
                    F405N_ZP = 206.99 

                    zp_map = { 
                        'F115W': F115W_ZP, 
                        'F212N': F212N_ZP, 
                        'F323N': F323N_ZP, 
                        'F405N': F405N_ZP, 
                    }

                    zp = zp_map.get(filter)
                    flux = zp * 10**(-0.4*mag) * 1e-23 # convert to cgs units

                    return 4*np.pi*self.dist**2*flux

                # Rough band-luminosities for each wavelength
                lums = {
                    'lF115W': mag_to_lum(mags['mF115W'], 'F115W'), 
                    'lF212N': mag_to_lum(mags['mF212N'], 'F212N'), 
                    'lF323N': mag_to_lum(mags['mF323N'], 'F323N'), 
                    'lF405N': mag_to_lum(mags['mF405N'], 'F405N'), 
                }

                # Append to main  
                data = pd.concat([
                    data,
                    pd.DataFrame(
                        {
                            'x': catalog_x,  # x positions
                            'y': catalog_y,  # y positions
                            'mF115W': mags['mF115W'], 
                            'mF212N': mags['mF212N'], 
                            'mF323N': mags['mF323N'], 
                            'mF405N': mags['mF405N'], 
                            'lF115W': lums['lF115W'], 
                            'lF212N': lums['lF212N'], 
                            'lF323N': lums['lF323N'], 
                            'lF405N': lums['lF405N'], 
                        }
                    )
                ], ignore_index=True)

        return data

    def fit_Teff(
        self, 
        L_115, L_212, L_323, L_405, 
        lam_115=1.15e-4, lam_212=2.12e-4, lam_323=3.23e-4, lam_405=4.05e-4,
    ):
        from scipy.optimize import curve_fit 

        c = 2.99792458e10       # Speed of light [cm/s]
        h = 6.62607004e-27      # Planck's constant [erg s]
        k_B = 1.38064852e-16    # Boltzmann's constant [erg/K]

        def planck_nu(nu, T): 
            numerator = 2.0 * h * nu**3 / c**2 
            denominator = np.exp(h*nu / (k_B * T)) - 1
            return numerator/denominator 

        def blackbody_lum(nu, T, scale): 
            return scale * planck_nu(nu, T) 

        def fit(): 
            nu_115 = c/lam_115 
            nu_212 = c/lam_212 
            nu_323 = c/lam_323 
            nu_405 = c/lam_405 

            nus = np.array([nu_115, nu_212, nu_323, nu_405]) 
            Ls  = np.array([L_115, L_212, L_323, L_405]) 

            # Mask out any NaNs 
            mask = ~np.isnan(Ls) 
            nu_used = nus[mask] 
            L_used  = Ls[mask] 

            initial_guess = [5000.0, 1e25] # [Temp, Scale]  
            bounds = ([1000, 1e10], [50000, 1e50]) 


            popt, pcov = curve_fit( 
                blackbody_lum,
                nu_used, L_used, 
                p0=initial_guess, 
                bounds=bounds, 
            )

            return popt, pcov, nu_used, L_used 

        def bolometric_luminosity(T, scale): 
            """
            Numerically integrate blackbody curve across all wavelengths 
            to estimate total bolometric luminosity in erg/s 

            """
            from scipy.integrate import quad 

            def lum_nu(nu): 
                return blackbody_lum(nu, T, scale)

            val, _ = quad(lum_nu, 0, 1e16)
            return val 

        def plot_blackbody_fit(L_115, L_212, L_323, L_405, T_fit, scale_fit): 
            lam_115, lam_212, lam_323, lam_405 = 1.15e-4, 2.12e-4, 3.23e-4, 4.05e-4
            nu_115, nu_212, nu_323, nu_405 = c / lam_115, c / lam_212, c / lam_323, c / lam_405

            nus_band = np.array([nu_115, nu_212, nu_323, nu_405])
            Ls_band = np.array([L_115, L_212, L_323, L_405])

            # Mask NaN values
            mask = ~np.isnan(Ls_band)
            nus_band = nus_band[mask]
            Ls_band = Ls_band[mask]

            nu_fit = np.logspace(np.log10(1e10), np.log10(1e16), 500)  
            L_fit = blackbody_lum(nu_fit, T_fit, scale_fit)

            plt.figure(figsize=(10, 6))
            plt.loglog(c/nu_fit, L_fit, label=f'Blackbody Fit (T={T_fit:.0f} K)', color='blue')
            plt.scatter(c/nus_band, Ls_band, color='red', s=10, label='Observed Band Luminosities')

            plt.xlabel(r'Wavelength $\lambda$ [cm]', fontsize=14)
            plt.ylabel(r'Luminosity $L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]', fontsize=14)
            plt.title('Blackbody Fit to Band-Luminosities', fontsize=16)
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig('run_imgs/blackbody_fit.png', dpi=300)
            return 

        popt, pcov, nu_used, L_used = fit()
        Teff, scale_fit = popt 
        bol_lum = bolometric_luminosity(Teff, scale_fit) 

        #plot_blackbody_fit(L_115, L_212, L_323, L_405, Teff, scale_fit)

        return popt, pcov, nu_used, L_used, bol_lum 


    def render(self, verbose=False): 
        data = self.data() 
        data['Luminosity'] = None   # initialize Bolometric Luminosity column 
        data['Teff'] = None         # initialize Teff column 
    
        if verbose: 
            print("Rendering") 
            print("{:>10} {:>15} {:>15} {:>15}".format(
                "idx",
                "Bol. Luminosity", 
                "Teff (K)",
                "Scale", 
            ))

        for idx in tqdm(range(len(data))):
            L_115 = data['lF115W'][idx]
            L_212 = data['lF212N'][idx] 
            L_323 = data['lF323N'][idx]
            L_405 = data['lF405N'][idx]

            popt, _, _, _, bol_lum = self.fit_Teff(
                L_115, L_212, L_323, L_405,
            )

            Teff, scale_fit = popt 

            if verbose:
                print("{:>10} {:>15.4e} {:>15.4f} {:>15.4e}".format(
                    idx, bol_lum, Teff, scale_fit, 
                ))

            data.at[idx, 'Luminosity'] = bol_lum 
            data.at[idx, 'Teff'] = Teff

        return data 












if __name__ == '__main__': 
    inst = DataSelection(dist=2.47e22, num_bins=1, num_stars=1)
    data = inst.render(verbose=False)




