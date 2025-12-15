from matplotlib import pyplot as plt
import numpy as np

# -------------------------- #
# EXTRACT DATA FROM CSV
# -------------------------- #

# Ignore the column name with "skiprows=1"
nov_data = np.loadtxt('2025-11-30_photometry.csv', skiprows=1, delimiter=',')
dec_data = np.loadtxt('2025-12-05_photometry.csv', skiprows=1, delimiter=',')
# COLUMN 0 = time [MJD], COLUMN 1 = mag, COLUMN 2 = mag_err

# Combine numpy arrays
data = np.concatenate((nov_data, dec_data), axis=0)

# -------------------------- #
# CREATE FIGURE FOR LIGHTCURVE
# -------------------------- #

fig, ax = plt.subplots(figsize=(16,4))

# Convert time from MJD into elapsed minutes
# [:,0] = all rows, 0th columns
elapsed_time = (data[:,0] - data[0,0]) * 24
mag = data[:,1]
mag_err = data[:,2]

ax.errorbar(elapsed_time, mag, yerr=mag_err,
            color='red', markersize=2, fmt='o',
            ecolor='orange', capsize=2)

ax.set_xlabel(f'Time after MJD {data[0,0]} [hours]')
ax.set_ylabel('Apparent magnitude (F606W)')
ax.set_title(f'Kallichore F606W Lightcurve (2025-Nov-30 and 2025-Dec-05)')
ax.invert_yaxis()

plt.tight_layout()
fig.savefig('Kallichore_combined_lightcurve.pdf', dpi=200)
plt.show()