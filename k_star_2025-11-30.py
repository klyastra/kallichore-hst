from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

from astropy.io import fits
import numpy as np
from photutils.aperture import CircularAperture, ApertureStats
from astropy import units as u

from astropy.nddata import Cutout2D
from photutils.detection import DAOStarFinder
from photutils.background import MMMBackground

import uncertainties as unc
from uncertainties import umath # provides uncertainty-aware math functions

# -------------------------- #
# EXTRACT DATA
# -------------------------- #

# https://mast.stsci.edu/search/ui/#/hst/results?proposal_id=18215
filename_list = [
        'ifry01e5q',
        'ifry01e6q',
        'ifry01e7q',
        'ifry01e8q',
        'ifry01e9q',
        'ifry01eaq',
        'ifry01ebq',
        'ifry01ecq',
        ]
filepath = '2025-11-30/'
suffix = '_drc.fits'


# -------------------------- #
# Prepare 1st figure - set up GridSpec
# The following loop will consecutively fill out the grid squares
# -------------------------- #
vmin, vmax = 0, 1.5  # set min/max brightness levels
fig1 = plt.figure(figsize=(11, 8))

n_rows = 2
n_cols = 5
# make the colorbar axis thin (column 5)
gs = GridSpec(n_rows, n_cols, figure=fig1, width_ratios=[1, 1, 1, 1, 0.1], wspace=0, hspace=-0.49)  # no spacing between subplots


# Initialize empty lists, which we'll append to in the following loop
obstime_list = []
mag_list = []
mag_err_list = []

# MODIFY THIS
star_id = 2

# Set x, y positions of stars
starpos = np.array([(657.1, 703.3),
                (678.8, 721.0),
                (698.7, 744.0),
                (718.2, 765.9),
                (743.9, 788.4),
                (770.8, 814.5),
                (790.4, 839.1),
                (803.4, 855.7)])

if star_id == 1:
        starname = 'SDSS J074349.26+211524.8 (g=22.470, r=22.278)'
                
if star_id == 2:
        starname = 'Gaia DR3 673004092335513856 (G=19.361)'
        starpos = starpos + (360.2, 409.7)

for fn in filename_list:
        with fits.open(filepath+fn+suffix) as hdul:
                # hdul.info()  # show the HDU list (hdul)
                '''
                No.    Name      Ver    Type      Cards   Dimensions   Format
                0  PRIMARY       1 PrimaryHDU     811   ()
                1  SCI           1 ImageHDU        86   (2058, 2176)   float32
                2  WHT           1 ImageHDU        45   (2058, 2176)   float32
                3  CTX           1 ImageHDU        40   (2058, 2176)   int32
                4  HDRTAB        1 BinTableHDU    561   1R x 276C 
                '''
                data = hdul[1].data  # get image data from Science HDU, apply units
                data[data > 25.] = np.nan  # remove cosmic rays and hot pixels from image

                prim_header = hdul[0].header  # header from Primary HDU
                sci_header = hdul[1].header  # header from Science HDU

                # Get the value and comment of a keyword in the HDU header
                zp_mag = sci_header['PHOTZPT']  # magnitude of an object that produces 1 count per second on the detector
                photflam = sci_header['PHOTFLAM']  # flux of zero point
                pivot_wv = sci_header['PHOTPLAM']  # Pivot wavelength (Angstroms)

                obs_time = prim_header['EXPSTART']  # observation star time in MJD
                exp_time = prim_header['EXPTIME']  # exposure time in seconds

                # read and gain already corrected in drc WFC3 images, no need to use these
                # https://hst-docs.stsci.edu/wfc3dhb/chapter-5-wfc3-uvis-sources-of-error/5-1-gain-and-read-noise#id-5.1GainandReadNoise-5.1.25.1.2ReadNoise
                ## gain = prim_header['CCDGAIN']  # CCD gain noise
                ## read = 2.99 # read noise is 2.99 for UVIS-2K2C (see HST proposal)

                # data from Binary Table containing Kallichore's predicted position
                # (note: predicted position is always offset toward the upper right from actual position
                table_data = hdul[4].data 
                xpos, ypos = table_data['CRPIX1'][0], table_data['CRPIX2'][0]  # predicted location of Kallichore [px]. Note that original values are given as a single-entry list.


        # -------------------------- #
        # AUTOMATICALLY FIND LOCATION OF MOON
        # -------------------------- #
        print(f'--- {fn} ---')
        fn_index = filename_list.index(fn)
        # Define cutout location, with "position" being the cutout's center
        imwidth = 50  # image cutout box half-width [px]
        cutout = Cutout2D(data, position=(starpos[fn_index][0], starpos[fn_index][1]), size=imwidth)
        cut_data = cutout.data

        # Mask NaNs
        mask = np.isnan(cut_data)
        # Background estimate
        bkg_estimator = MMMBackground()
        bkg = bkg_estimator(cut_data[~mask])  # Note that this isn't a completely reliable way of measuring background level

        # Noise estimate
        sigma = np.nanstd(cut_data - bkg)

        try:
                # -------------------------- #
                # PHOTOMETRY (ST mag --> Vega mag)
                # Vega mag system is standard for Solar System objects; see https://sci-hub.se/10.1016/j.icarus.2008.10.025
                # -------------------------- #
                # Add circular photometric apertures to residual subplot
                moon_xy_loc = (starpos[fn_index][0], starpos[fn_index][1])
                if star_id == 2:
                        phot_radius = 10
                else:
                        phot_radius = 8


                # perform aperture photometry to get moon flux
                aperture = CircularAperture(moon_xy_loc, r=phot_radius)
                aperstats = ApertureStats(data, aperture)

                # ApertureStats returns centroid, mean, median, standard deviation (std), sum, sum_aper_area
                # we want background-subtracted sum & std.

                ### Obtain the background from the data cutout: ###
                bg = cut_data.copy()
                bg[bg > 0.2] = np.nan  # replace high values with NaN to isolate background
                bg_median = np.nanmedian(bg)  # take median of background, ignoring NaNs
                bg_std = np.nanstd(bg)  # take standard deviation of background, ignoring NaNs
                print(f'bg_median = {bg_median}')
                print(f'bg_std = {bg_std}')

                ### Compute flux and its uncertainty (flux_err) from aperture photometry ###
                # Note that aperstats.sum_aper_area outputs a Quantity object. Use the astropy.units ".value" suffix to remove unit.
                aperture_area = aperstats.sum_aper_area.value  # aperture area in px^2 equivalent to number of enclosed pixels
                bg_subtracted_flux = aperstats.sum - aperture_area * bkg  # subtract background from sum
                if fn_index == 6 and star_id == 2:
                        bg_subtracted_flux -= 10  # subtract cosmic ray

                # Use the flux uncertainty equation from https://web.ipac.caltech.edu/staff/fmasci/home/mystats/ApPhotUncert_corr.pdf
                flux_err = np.sqrt(bg_subtracted_flux/exp_time + aperture_area*(bg_std)**2)

                moon_flux = unc.ufloat(bg_subtracted_flux, flux_err)  # store into a float with uncertainty
                print(f'moon_flux = {moon_flux:.5g}')  # ":.5g" = print with 5 sigfigs
                # You can extract the nominal value and uncertainty respectively:
                ## print(f"Nominal value: {moon_flux.nominal_value}")
                ## print(f"Propagated error (std dev): {moon_flux.std_dev}")

                # Convert flux (electrons/s) to ST magnitudes using ST zero point & PhotFlam.
                # In drc-format images, exposure time has already been taken into account. DON'T USE EXPTIME FOR DRC!
                # https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
                moon_STmag = zp_mag - 2.5 * umath.log10(photflam * moon_flux)
                print(f"moon_STmag = {moon_STmag:.5g}")

                # Convert ST to AB magnitudes according to https://hst-docs.stsci.edu/acsdhb/chapter-5-acs-data-analysis/5-1-photometry
                moon_ABmag = moon_STmag - 5*umath.log10(pivot_wv) + 18.6921

                # Convert AB to Vega magnitude by subtracting the AB-Vega difference (0.09) according to the table in the following link:
                # https://hst-docs.stsci.edu/acsihb/chapter-10-imaging-reference-material/10-3-throughputs-and-correction-tables
                moon_Vegamag = moon_ABmag - 0.09
                print(f"moon_Vegamag = {moon_Vegamag:.5g}")

                ST_Vega_magdiff = moon_STmag - moon_Vegamag
                print(f"ST_Vega_magdiff = {ST_Vega_magdiff:.5g}")

        except Exception as e:
                print(f"Error: {e}")
                moon_Vegamag = np.nan
                moon_STmag = np.nan
                moon_ABmag = np.nan
                ST_Vega_magdiff = np.nan
                print('')
                continue

        # -------------------------- #
        # FILL SUBPLOT IN GRIDSPEC
        # -------------------------- #
        row_index = 0
        column_index = filename_list.index(fn)  # get index number of selected filename; use that for column index
        if column_index >= 4:
                column_index -= 4  # reset index to 0
                row_index = 1  # move on to next row

        ax = fig1.add_subplot(gs[row_index, column_index])
        data_image = ax.imshow(data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax, aspect='equal')  # 'auto' to remove gaps between subplots
        ax.set_box_aspect(1)
        ax.set_xlim(starpos[fn_index][0]-imwidth, starpos[fn_index][0]+imwidth)
        ax.set_ylim(starpos[fn_index][1]-imwidth, starpos[fn_index][1]+imwidth)
        ax.set_xticks([])
        ax.set_yticks([])  # disable ticks and labels

        # add aperture circle to subplot
        phot_aperture = Circle(moon_xy_loc, radius=phot_radius,
                edgecolor='red', fill=False, alpha=1, lw=1)
        ax.add_patch(phot_aperture)
        

        # -------------------------- #
        # APPEND TO LIST AFTER LOOP ITERATION
        # -------------------------- #
        obstime_list.append(obs_time)  # MJD
        mag_list.append(moon_Vegamag.nominal_value)
        mag_err_list.append(moon_Vegamag.std_dev)
        print('')


# -------------------------- #
# CREATE 1ST FIGURE SHOWING FINAL IMAGE PREVIEW
# (show plot of final image, after loop ends)
# -------------------------- #
fig1.suptitle(f'Star {starname} - 2025-Nov-30 HST/WFC3 image cutouts', y=0.785, fontsize=14)

# Create separate axes for colorbar
cax = fig1.add_subplot(gs[0:2, 4])

# Reduce height of colorbar (thanks ChatGPT for providing a solution to this stupid GridSpec spacing behavior)
pos = cax.get_position()

new_height = 0.675 * pos.height        # 75% of total height
new_y0 = pos.y0 + 0.5 * (pos.height - new_height)

cax.set_position([
    pos.x0,
    new_y0,
    pos.width,
    new_height
])

# Add colorbar
fig1.colorbar(data_image, cax=cax, label='Flux (electrons/s)')  # let it steal axis

fig1.tight_layout()


# -------------------------- #
# CREATE 2ND FIGURE FOR LIGHTCURVE
# -------------------------- #

fig2, ax2 = plt.subplots(figsize=(11, 8))

# Convert time from MJD into elapsed minutes
obstime_arr = np.array(obstime_list)
elapsed_time_min = (obstime_arr - obstime_arr[0]) * 24 * 60

ax2.errorbar(elapsed_time_min, mag_list, yerr=mag_err_list,
            color='red', markersize=2, fmt='o',
            ecolor='orange', capsize=2)

ax2.set_xlabel(f'Time after MJD {obstime_list[0]} [min]')
ax2.set_ylabel('Apparent magnitude (F606W)')
ax2.set_title(f'Star {starname} - F606W Lightcurve (2025-Nov-30)')
ax2.invert_yaxis()


# Combine arrays horizontally into a single 2D array and then save as CSV
combined_array = np.column_stack((obstime_arr, np.array(mag_list), np.array(mag_err_list)))
if star_id == 1:        
        np.savetxt(
                '2025-11-30_star1_phot.csv',
                combined_array,
                delimiter=',',
                header='Time [MJD], Mag, MagErr',
                comments=''  # Avoids '#' at the start of the header line
                )
if star_id == 2:        
        np.savetxt(
                '2025-11-30_star2_phot.csv',
                combined_array,
                delimiter=',',
                header='Time [MJD], Mag, MagErr',
                comments=''  # Avoids '#' at the start of the header line
                )

fig2.tight_layout()
plt.show()