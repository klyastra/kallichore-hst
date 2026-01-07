from matplotlib import pyplot as plt
import numpy as np

# -------------------------- #
# EXTRACT DATA FROM CSV
# -------------------------- #

# Ignore the column name with "skiprows=1"
kall_data = np.loadtxt('2025-11-30_photometry.csv', skiprows=1, delimiter=',')
kall_mag = kall_data[:,1]
kall_magmed = np.median(kall_mag)
kall_mag -= kall_magmed

star1_data = np.loadtxt('2025-11-30_star1_phot.csv', skiprows=1, delimiter=',')
star1_mag = star1_data[:,1]
star1_magmed = np.median(star1_mag)
star1_mag -= star1_magmed

star2_data = np.loadtxt('2025-11-30_star2_phot.csv', skiprows=1, delimiter=',')
star2_mag = star2_data[:,1]
star2_magmed = np.median(star2_mag)
star2_mag -= star2_magmed
# COLUMN 0 = time [MJD], COLUMN 1 = mag, COLUMN 2 = mag_err

# -------------------------- #
# CREATE FIGURE FOR LIGHTCURVE
# -------------------------- #

fig, ax = plt.subplots(figsize=(11,8))

# Convert time from MJD into elapsed minutes
# [:,0] = all rows, 0th columns
elapsed_time = (kall_data[:,0] - kall_data[0,0]) * 24

ax.plot(elapsed_time, kall_mag, color='red', label='Kallichore', marker='o')
ax.fill_between(elapsed_time, kall_mag-kall_data[:,2], kall_mag+kall_data[:,2],
                color='red', alpha=0.25)
ax.plot(elapsed_time, star1_mag, color='green', label='Faint star (SDSS)', marker='s')
ax.fill_between(elapsed_time, star1_mag-star1_data[:,2], star1_mag+star1_data[:,2],
                color='green', alpha=0.25)
ax.plot(elapsed_time, star2_mag, color='blue', label='Bright star (Gaia)', marker='^')
ax.fill_between(elapsed_time, star2_mag-star2_data[:,2], star2_mag+star2_data[:,2],
                color='blue', alpha=0.25)

'''
ax.errorbar(elapsed_time, kall_mag, yerr=kall_data[:,2],
            color='red', markersize=4, fmt='o',
            ecolor='salmon', capsize=4, label='Kallichore')

ax.errorbar(elapsed_time, star1_mag, yerr=star1_data[:,2],
            color='green', markersize=4, fmt='s',
            ecolor='yellowgreen', capsize=3, label='Faint star (SDSS)')

ax.errorbar(elapsed_time, star2_mag, yerr=star2_data[:,2],
            color='blue', markersize=4, fmt='^',
            ecolor='turquoise', capsize=2, label='Bright star (Gaia)')
'''

ax.set_xlabel(f'Time after MJD {kall_data[0,0]} [hours]')
ax.set_ylabel('Apparent magnitude (F606W)')
ax.set_title(f'F606W Mag Variations (2025-Nov-30)')
ax.invert_yaxis()

# Add a legend (default location is "best", usually upper left)
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('Lightcurve_magvariations_comparison.pdf')
plt.show()