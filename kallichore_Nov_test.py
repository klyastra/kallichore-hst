from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle

from astropy.io import fits
import numpy as np
from photutils.aperture import CircularAperture, ApertureStats

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

with fits.open(filepath+filename_list[0]+suffix) as hdul:
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
        data[data > 2.] = np.nan  # remove cosmic rays and hot pixels from image

        prim_header = hdul[0].header  # header from Primary HDU
        sci_header = hdul[1].header  # header from Science HDU

        # Get the value and comment of a keyword in the HDU header
        zp_mag = sci_header['PHOTZPT']  # magnitude of an object that produces 1 count per second on the detector
        zp_flux = sci_header['PHOTFLAM']  # flux of zero point

        obs_time = prim_header['EXPSTART']  # observation star time in MJD
        exp_time = prim_header['EXPTIME']  # exposure time in seconds

        # https://hst-docs.stsci.edu/wfc3dhb/chapter-5-wfc3-uvis-sources-of-error/5-1-gain-and-read-noise#id-5.1GainandReadNoise-5.1.25.1.2ReadNoise
        gain = prim_header['CCDGAIN']  # CCD gain noise
        read = 2.99 # read noise is 2.99 for UVIS-2K2C (see HST proposal)

        # data from Binary Table containing Kallichore's predicted position
        # (note: predicted position is always offset toward the upper right from actual position
        table_data = hdul[4].data 
        xpos, ypos = table_data['CRPIX1'][0], table_data['CRPIX2'][0]  # predicted location of Kallichore [px]. Note that original values are given as a single-entry list.


# -------------------------- #
# CREATE PLOT
# -------------------------- #
vmin, vmax = 0, 1.25

# create 3-column row of subplots
fig, ax = plt.subplots(figsize=(11, 8))

# data
data_image = ax.imshow(data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
fig.colorbar(data_image, label='Flux (electrons/s)', shrink=0.75)  # let it steal axis


imwidth = 40  # image cutout box half-width [px]
ax.set_xlim(xpos-imwidth, xpos+imwidth)
ax.set_ylim(ypos-imwidth, ypos+imwidth)
ax.set_xlabel('x [px]')
ax.set_ylabel('y [px]')
ax.set_title(f'Kallichore WFC3 image ({filename_list[0]})')


# -------------------------- #
# AUTOMATICALLY FIND LOCATION OF MOON
# -------------------------- #
# Define cutout location, with "position" being the cutout's center
cutout = Cutout2D(data, position=(xpos, ypos), size=imwidth)
cut_data = cutout.data

# Mask NaNs
mask = np.isnan(cut_data)
# Background estimate
bkg_estimator = MMMBackground()
bkg = bkg_estimator(cut_data[~mask])
# Noise estimate
sigma = np.nanstd(cut_data - bkg)

# Compute centroid in cutout's coordinates
daofind = DAOStarFinder(fwhm=2.0, threshold=5.0*sigma)
sources = daofind(cut_data - bkg)

if sources is None or len(sources) == 0:
    raise RuntimeError("No point source detected in cutout centered at moon's predicted location")

# Choose the brightest detected source
i = np.argmax(sources['flux'])
src = sources[i]

# Replace moon's predicted position (cutout center) with actual position obtained from cutout
print(f"Predicted moon location at x={xpos:.2f}, y={ypos:.2f}")
xpos = src['xcentroid'] + cutout.xmin_original
ypos = src['ycentroid'] + cutout.ymin_original
print(f"Detected point source at x={xpos:.2f}, y={ypos:.2f}")


# -------------------------- #
# PHOTOMETRY
# -------------------------- #
# Add circular photometric apertures to residual subplot
moon_xy_loc = (xpos, ypos)
phot_radius = 6

phot_aperture = Circle(moon_xy_loc, radius=phot_radius,
        edgecolor='red', fill=False, alpha=1, lw=1)

# show plot
ax.add_patch(phot_aperture)
plt.tight_layout()
plt.show()



# perform aperture photometry to get moon flux
aperture = CircularAperture(moon_xy_loc, r=phot_radius)
aperstats = ApertureStats(data, aperture)

# ApertureStats returns centroid, mean, median, standard deviation (std), sum
# we want sum & std
## moon_flux, moon_flux_std = aperstats.sum, aperstats.std
moon_flux = unc.ufloat(aperstats.sum, aperstats.std)
print(f'moon_flux = {moon_flux:.5g}')  # ":.5g" = print with 5 sigfigs
# You can extract the nominal value and uncertainty respectively:
## print(f"Nominal value: {moon_flux.nominal_value}")
## print(f"Propagated error (std dev): {moon_flux.std_dev}")

# Convert flux to magnitudes using zero point
# https://www.stsci.edu/hst/wfpc2/Wfpc2_dhb/wfpc2_ch52.html
moon_mag = -zp_mag - 2.5 * umath.log10(moon_flux/exp_time)
print(f"moon_mag = {moon_mag:.5g}")