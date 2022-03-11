# satellite
This repo contains code to load, manipulate, and plot satellite data for SASSIE.

* smap_beaufort_plots.ipynb : load & plot SMAP L2 and L3 data for one day
* convert_seaice_coords.ipynb : for sea ice concentration data stored locally, convert the polar grid to a lat/lon grid for easier reading. (There are probably smarter ways to do this, e.g., just plotting the data in their native format... maybe a good future issue?)
* plot_beaufort_seaice.ipynb : load the SIC data on lat/lon grid that was created with convert_seaice_coords.ipynb, and plot mean SIC as well as some contours for interesting years


