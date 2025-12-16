For this code to work, you'll need to create the following folders in the working directory:
* 2025-11-30
* 2025-12-05

Each folder contains eight *.fits files corresponding to the HST/WFC3 (drc calibratred) images taken on those dates.

The following Python .py scripts generate these files:
* kallichore 2025-11-30.py
    *   2025-11-30_photometry.csv
    *   Kallichore_2025-11-30_lightcurve.pdf
    *   Kallichore_2025-11-30_cutouts.pdf
* kallichore 2025-12-05.py
    *   2025-12-05_photometry.csv
    *   Kallichore_2025-12-05_lightcurve.pdf
    *   Kallichore_2025-11-30_cutouts.pdf
* combined_kallichore_lightcurve.py
    *   Kallichore_combined_lightcurve.pdf

 The .py files are ideally run via Linux command line (e.g. Ubuntu)