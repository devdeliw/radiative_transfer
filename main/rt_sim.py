import os
import numpy as np 
import matplotlib.pyplot as plt

from data import DataSelection 
from hyperion.model import Model 
from hyperion.util.constants import pc, lsun

file_dir  = './run_files/' 
image_dir = './run_imgs/' 
dust_dir  = './dust_files/'

if not os.path.exists(file_dir): 
    os.makedirs(file_dir, exist_ok=True) 
if not os.path.exists(image_dir): 
    os.makedirs(image_dir, exist_ok=True) 

class RadiativeTransfer(DataSelection):
    """ 
    Runs a Hyperion Radiative Transfer Model and outputs synthetic 
    SEDs and images based on a chosen catalog. 

    Args: 
        * dist      (float) : The distance to the starcluster in cm
        * file_in   (str)   : The rtin file for the model input 
        * file_out  (str)   : The rtout file for the model results 
        * image_fil (str)   : Directory to place images 
        * num_stars (int)   : The number of stars you wish to sample
        * num_bins  (int)   : The number of bins across x-space to sample from 
        * sed       (bool)  : Whether to prepare a synthetic SED 
        * image     (bool)  : Whether to prepare a synthetic image 
        * all       (bool)  : Whether to just select all of the stars (Optional) 

    """

    def __init__(
        self, 
        dist, file_in, file_out,
        image_file, num_stars=1000, num_bins=10, 
        sed=True, image=False, all=False,
        overwrite=True,
    ):
        self.dist       = dist 
        self.num_stars  = num_stars 
        self.num_bins   = num_bins 
        self.sed        = sed 
        self.file_in    = file_in 
        self.file_out   = file_out 
        self.plot_dir   = image_file
        self.image      = image
        self.pc_dist    = dist * 3.086e-18
        self.all        = all 

        self.PLATE_SCALE= 0.031 
        self.ARCSEC_PC  = self.pc_dist/206265

        if overwrite == True:
            if os.path.exists(file_in): 
                os.remove(file_in) 
            if os.path.exists(file_out):
                os.remove(file_out) 
        return

    def initialize(self): 
        super().__init__(
                self.dist, all=self.all,
                num_stars=self.num_stars, num_bins=self.num_bins, 
        )
        self.df = super().render()

    def centroid_plot(self, filename='centroid_pxl.png'): 
        x = self.df['x'] 
        y = self.df['y'] 

        _, _ = plt.subplots(1, 1, figsize=(10, 8)) 
        plt.scatter(x, y, c='k', s=0.1) 
        plt.xlabel('$x$ centroid', fontsize=14) 
        plt.ylabel('$y$ centroid', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(os.path.join(self.plot_dir, filename))
        return 

    def rt_model(
        self, 
        n_iterations=2, 
        wave_range=(0.01, 1e4), num_wave_samples=500,
        num_grid_bins=200, 
        n_photons={
            'initial'           : 1e6, 
            'imaging'           : 1e7, 
            'raytracing_sources': 1e5, 
            'raytracing_dust'   : 1e7,
        }, 
    ): 
        self.m = Model() 

        x_ref = np.median(self.df['x']) 
        y_ref = np.median(self.df['y']) 

        x_coords, y_coords = [], [] 
        for row in self.df.itertuples(index=False): 
            xpc = (row.x - x_ref) * self.PLATE_SCALE * self.ARCSEC_PC # convert pixel to pc 
            ypc = (row.y - y_ref) * self.PLATE_SCALE * self.ARCSEC_PC # convert pixel to pc 
            zpc = 0. # all stars will lie on same plan (approximation) 

            luminosity  = row.Luminosity
            temperature = row.Teff

            source = self.m.add_point_source() 
            source.luminosity = luminosity
            source.position = (xpc*pc, ypc*pc, zpc*pc) 
            source.temperature = temperature 
            x_coords.append(xpc)
            y_coords.append(ypc)

        w = np.linspace(-self.pc_dist/2*pc, self.pc_dist/2*pc, num_grid_bins) 
        self.m.set_cartesian_grid(w, w, w)  
        self.m.set_raytracing(True) 
        self.m.set_mrw(True) 
        self.m.set_n_initial_iterations(n_iterations) 
        self.m.set_n_photons(
                initial            = n_photons['initial'], 
                imaging            = n_photons['imaging'], 
                raytracing_sources = n_photons['raytracing_sources'], 
                raytracing_dust    = n_photons['raytracing_dust'], 
        )

        self.sed = self.m.add_peeled_images(sed=self.sed, image=self.image)
        self.sed.set_wavelength_range(num_wave_samples, *wave_range) 
        self.sed.set_viewing_angles([0.], [0.]) 
        self.sed.set_track_origin('basic')

        if self.image:
            self.sed.set_image_size(400, 400) 
            self.sed.set_image_limits(
                    1.5*min(x_coords)*pc, 1.5*max(x_coords)*pc, 
                    1.5*min(y_coords)*pc, 1.5*max(y_coords)*pc,
            )

        self.w = w
        self.image_dist    = 1.5*max(x_coords) - 1.5*min(x_coords)
        self.num_grid_bins = num_grid_bins
        self.wave_range    = wave_range
        self.xpc = x_coords 
        self.ypc = y_coords
        return

    def rt_dust(self, transitions=[[0.2, 0.3], [0.5, 0.8]]):
        background_file = dust_dir + 'd03_3.1_6.0_A.hdf5'
        central_GC_file = dust_dir + 'd03_4.0_4.0_A.hdf5'
        inner_GC_file   = dust_dir + 'd03_5.5_3.0_A.hdf5'
        
        dusts = [background_file, central_GC_file, inner_GC_file]
        densities = [5e-26, 1e-25, 1e-24]
        
        n_dust = len(dusts)
        if len(densities) != n_dust:
            raise ValueError("The `densities` array must have the same length as `dusts`.")
        if len(transitions) != n_dust - 1:
            raise ValueError("The `transitions` list must have length (number of dust files-1).")
        
        num_cells = self.num_grid_bins - 1
        
        # 1D fractional positions along the line-of-sight (or chosen axis),
        f = np.linspace(0, 1, num_cells)
        
        # 1D density profile for each dust component 
        density_profiles = [np.zeros(num_cells) for _ in range(n_dust)]
        
        # First component
        t0_start, t0_end = transitions[0]

        # Pure region 
        mask_pure = f < t0_start
        density_profiles[0][mask_pure] = densities[0]
        
        # Transition region (dust 0 fades, dust 1 rises)
        mask_trans = (f >= t0_start) & (f <= t0_end)
        fraction = (f[mask_trans] - t0_start) / (t0_end - t0_start)
        density_profiles[0][mask_trans] = densities[0] * (1 - fraction) # fades 
        density_profiles[1][mask_trans] = densities[1] * fraction       # rises
        
        # Intermediate dust components 
        for i in range(1, n_dust - 1):
            # Previous transition: dust i rising 
            prev_start, prev_end = transitions[i - 1]
            mask_prev = (f >= prev_start) & (f <= prev_end)
            fraction_prev = (f[mask_prev] - prev_start) / (prev_end - prev_start)
            density_profiles[i][mask_prev] = densities[i] * fraction_prev
            
            # Pure region
            mask_const = (f > prev_end) & (f < transitions[i][0])
            density_profiles[i][mask_const] = densities[i]
            
            # Next transition: dust i fades, dust i+1 rises
            cur_start, cur_end = transitions[i]
            mask_cur = (f >= cur_start) & (f <= cur_end)
            fraction_cur = (f[mask_cur] - cur_start) / (cur_end - cur_start)
            density_profiles[i][mask_cur] = densities[i] * (1 - fraction_cur)
            density_profiles[i+1][mask_cur] = densities[i+1] * fraction_cur
    
        # Last component
        # Previous transition: dust -1 rising
        prev_start, prev_end = transitions[-1]
        mask_prev_last = (f >= prev_start) & (f <= prev_end)
        fraction_last = (f[mask_prev_last] - prev_start) / (prev_end - prev_start)
        density_profiles[-1][mask_prev_last] = densities[-1] * fraction_last
        
        # Pure region 
        mask_pure_last = f > prev_end
        density_profiles[-1][mask_pure_last] = densities[-1]
        
        density_grids = []
        for profile in density_profiles:
            profile_3d = np.tile(profile.reshape(num_cells, 1, 1), (1, num_cells, num_cells))
            density_grids.append(profile_3d)
        
        # Add dust components to hyperion model 
        for dust_file, dens_grid in zip(dusts, density_grids):
            self.m.add_density_grid(dens_grid, dust_file)
    
        return

    def rt_render(self): 
        self.m.write(self.file_in) 
        self.m.run(self.file_out, mpi=True) 
        return 

    def simulate(self): 
        self.initialize()
        self.rt_model()     # initialize hyperion model
        self.rt_dust()      # dust config 
        self.rt_render()    # run simulation 

    def sed_plot(self, filename = 'sed1000.png'): 
        from hyperion.model import ModelOutput 

        self.msed = ModelOutput(self.file_out) 

        fig, ax = plt.subplots(1, 1, figsize=(10,8)) 
        sed = self.msed.get_sed(inclination=0, aperture=-1, distance=self.pc_dist*pc)

        ax.loglog(sed.wav, sed.val)
        ax.set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=14) 
        ax.set_ylabel(r'$\lambda F_\lambda$ [ergs/s/cm$^2$]', fontsize=14)
        ax.set_xlim(self.wave_range[0], self.wave_range[1]) 

        fig.savefig(os.path.join(self.plot_dir, filename))
        return 

    def img_plot(
        self,
        wavelengths=[1.15, 2.12, 3.23, 400], 
        filename = 'img1000.png', 
    ): 
        from hyperion.model import ModelOutput 

        self.mimg = ModelOutput(self.file_out) 

        image = self.mimg.get_image(
            inclination = 0, 
            distance    = self.image_dist *pc, 
            units       = 'MJy/sr'
        )

        fig = plt.figure(figsize=(8,8))

        for i, wav in enumerate(wavelengths): 
            ax = fig.add_subplot(2, 2, i+1)

            # find closest wavelength 
            iwav = np.argmin(np.abs(wav-image.wav)) 

            # image width in arcminutes
            x_range = max(self.xpc) - min(self.xpc) 
            w = np.degrees((x_range*pc)/image.distance) * 60

            ax.imshow(
                    image.val[:, :, iwav], 
                    cmap=plt.cm.gist_heat, 
                    origin='lower', 
                    extent=[-w, w, -w, w]
            )

            ax.tick_params(axis = 'both', which='major', labelsize=10) 
            ax.set_xlabel('x (arcmin)', fontsize=14)
            ax.set_ylabel('y (arcmin)', fontsize=14)

            ax.set_title(str(wav) + 'microns', y = 0.88, x = 0.5, color='white')
        fig.savefig(os.path.join(self.plot_dir, filename), bbox_inches='tight')
        return 

    def centroid_pc_plot(self, filename='centroid_pc.png'): 
        _, _ = plt.subplots(1, 1, figsize=(10, 8))
        plt.scatter(self.xpc, self.ypc, c='k', s=0.1) 
        plt.xlabel('$x$ pc') 
        plt.ylabel('$y$ pc') 

        plt.savefig(os.path.join(self.plot_dir, filename))





if __name__ == '__main__': 

    file_in  = os.path.join(file_dir, 's1000.rtin') 
    file_out = os.path.join(file_dir, 's1000.rtout') 
    plot_dir = image_dir

    inst = RadiativeTransfer(
        dist=8000*pc, num_bins=10, num_stars=1000, all=False, 
        sed=True, file_in=file_in, file_out=file_out, 
        image_file=plot_dir, image=True, overwrite=True, 
    )

    #inst.centroid_plot() 
    inst.initialize() 
    inst.simulate()
    inst.img_plot() 
    inst.sed_plot()
    #inst.centroid_pc_plot()



