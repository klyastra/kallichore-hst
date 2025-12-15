from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle

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
        'ifry02kuq',
        'ifry02kvq',
        'ifry02kwq',
        'ifry02kxq',
        'ifry02kyq',
        'ifry02kzq',
        'ifry02l0q',
        'ifry02l1q',
        ]
filepath = '2025-12-05/'
suffix = '_drc.fits'

# Initialize empty lists, which we'll append to in the following loop
obstime_list = []
mag_list = []
mag_err_list = []

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
                # remove cosmic rays and hot pixels from image
                # this is a quick-and-dirty way to work around the inconsistent image quality
                if fn == 'ifry02kyq':
                        data[data > 2.5] = np.nan
                elif fn == 'ifry02kzq' or fn == 'ifry02l0q':
                        data[data > 5.0] = np.nan
                else:
                        data[data > 2.0] = np.nan

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
        # Define cutout location, with "position" being the cutout's center
        imwidth = 40  # image cutout box half-width [px]
        cutout = Cutout2D(data, position=(xpos, ypos), size=imwidth)
        cut_data = cutout.data

        # Mask NaNs
        mask = np.isnan(cut_data)
        # Background estimate
        bkg_estimator = MMMBackground()
        bkg = bkg_estimator(cut_data[~mask])  # Note that this isn't a completely reliable way of measuring background level

        # Noise estimate
        sigma = np.nanstd(cut_data - bkg)

        try:
                # Compute centroid in cutout's coordinates
                daofind = DAOStarFinder(fwhm=2.0, threshold=5.0*sigma)
                sources = daofind(cut_data - bkg)

                if sources is None or len(sources) == 0:
                        raise RuntimeError("No point source detected in cutout centered at moon's predicted location")

                # Choose the brightest detected source
                j = np.argmax(sources['flux'])
                src = sources[j]

                # Replace moon's predicted position (cutout center) with actual position obtained from cutout
                print(f"Predicted moon location at x={xpos:.2f}, y={ypos:.2f}")
                xpos = src['xcentroid'] + cutout.xmin_original
                ypos = src['ycentroid'] + cutout.ymin_original

                # quick-and-dirty manual intervention for certain images
                if fn == 'ifry02kuq':
                        xpos += -1.2  # manual offset to avoid cosmic ray
                        ypos += +0.9  #
                if fn == 'ifry02l0q':
                        xpos = 760.2
                        ypos = 835.5

                print(f"Detected point source at x={xpos:.2f}, y={ypos:.2f}")

        except Exception as e:
                print(f"No point source detected in cutout centered at moon's predicted location ({e}).")
                print(f"But we can quickly remedy this with a manual shift in xpos, ypos:")
                if fn == 'ifry02kzq':
                        xpos = 761.5
                        ypos = 833.3

                


        # -------------------------- #
        # PHOTOMETRY (ST mag --> Vega mag)
        # Vega mag system is standard for Solar System objects; see https://sci-hub.se/10.1016/j.icarus.2008.10.025
        # -------------------------- #
        # Add circular photometric apertures to residual subplot
        moon_xy_loc = (xpos, ypos)
        phot_radius = 6

        # plot aperture
        phot_aperture = Circle(moon_xy_loc, radius=phot_radius,
                edgecolor='red', fill=False, alpha=1, lw=1)

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
vmin, vmax = 0, 1.5

fig1, ax1 = plt.subplots(figsize=(11, 8))

# data
data_image = ax1.imshow(data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
fig1.colorbar(data_image, label='Flux (electrons/s)', shrink=0.75)  # let it steal axis

ax1.set_xlim(xpos-imwidth, xpos+imwidth)
ax1.set_ylim(ypos-imwidth, ypos+imwidth)
ax1.set_xlabel('x [px]')
ax1.set_ylabel('y [px]')
ax1.set_title(f'Kallichore WFC3 image ({filename_list[-1]})')
ax1.add_patch(phot_aperture)

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
ax2.set_title(f'Kallichore F606W Lightcurve (2025-Dec-05)')
ax2.invert_yaxis()


# Combine arrays horizontally into a single 2D array and then save as CSV
combined_array = np.column_stack((obstime_arr, np.array(mag_list), np.array(mag_err_list)))
np.savetxt(
        '2025-12-05_photometry.csv',
        combined_array,
        delimiter=',',
        header='Time [MJD], Mag, MagErr',
        comments=''  # Avoids '#' at the start of the header line
        )

plt.tight_layout()
fig2.savefig('Kallichore_2025-12-05_lightcurve.pdf', dpi=200)
plt.show()