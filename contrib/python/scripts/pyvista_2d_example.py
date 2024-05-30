# Overview

# PyVista is a python-based 3D plotting and mesh analysis package,
# which is largely achieved through streamlined interface to the
# VTK package. Documentation for PyVista is located at
# https://docs.pyvista.org/version/stable/.

# While not inherently designed for plotting 2D data, various
# workflows can quickly produce high quality 2D plots through a
# scripting interface, while also allowing users to modify the
# underlying data (if desired) through standard python libararies
# (numpy, scipy, etc, etc). At minimum, pyvista thus provides
# a reliable and streamlined method for reading ASPECT pvtu files
# into python for further analysis.

# This contribution is designed to showcase this functionality,
# largely based on work by Dylan Vasey hosted at
# https://github.com/dyvasey/riftinversion and used in
# Vasey et al. 2024 (https://doi.org/10.1130/G51489.1).

# Future examples will illustrate how to visualize 3D models
# and this example will also be updated to highlight additional
# features for 2D model analysis and visualization.

# Installation Instructions
# While PyVista can be installed through Anaconda or PIP,
# the most straightforward way to ensure it works and does not
# produce conflicts with other python libraries is to create an
# anaconda environment with the package dependencies found
# at https://github.com/pyvista/pyvista/blob/main/environment.yml.
# After downloading this file, create a new anaconda environment
# using this file with "conda env create -f environment.yml".
# Once that environment has been created, activate it with
# "conda activate pyvista-env" and then install pyvist with
# "conda install -c conda-forge pyvista". You can check
# to see if pyvista is installed with "import pyvista as pv".


# Load modules
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# Read data from the pvtu file into a variable named mesh, which contains
# all information needed for the plot. In this case, use the output
# from the first timestep of the continental extension cookbook.
mesh = pv.read('output-continental_extension/solution/solution-00000.pvtu')

# The model bounds are x = 0-200 km, y = 0-100 km, and z = 0-0 km (2D model).
plot_spatial_bounds = [0,200e3,0,100e3, 0, 0]

# Here, we clip the mesh, but this should have no effect as the spatial
# boundaries specified above span the model domain. However, one could
# set lower y bound to 50e3, which would then only plot the upper 50 km.
mesh = mesh.clip_box(bounds=plot_spatial_bounds,invert=False)

# Set some properties about the plot.
pv.set_plot_theme("document")
plotter = pv.Plotter(off_screen=True)

# Define properties for the scalar bar arguments
sargs = dict(width=0.6, fmt='%.1e',height=0.2, title='Strain Rate Invariant (1/s)', \
             label_font_size=24, title_font_size=32, color='white', \
             position_x=0.2,position_y=0.01)

# Visualize the strain rate field on the current plotter
plotter.add_mesh(mesh,scalars='strain_rate',log_scale=True,scalar_bar_args=sargs)

# Set various properties of the plot
plotter.view_xy()
plotter.enable_depth_peeling(10)

# Calculate Camera Position from Bounds
###
bounds_array = np.array(plot_spatial_bounds)
xmag = float(abs(bounds_array[1] - bounds_array[0]))
ymag = float(abs(bounds_array[3] - bounds_array[2]))
aspect_ratio = ymag/xmag
plotter.window_size = (1024,int(1024*aspect_ratio))
xmid = xmag/2 + bounds_array[0] # X midpoint
ymid = ymag/2 + bounds_array[2] # Y midpoint
zoom = xmag*aspect_ratio*1.875 # Zoom level - not sure why 1.875 works

position = (xmid,ymid,zoom)
focal_point = (xmid,ymid,0)
viewup = (0,1,0)
camera = [position,focal_point,viewup]
plotter.camera_position = camera
plotter.camera_set = True
###

# Create the image via a screenshot
img = plotter.screenshot(transparent_background=True,return_img=True)

# Create the plot axis
ax = plt.gca()

# Plot the image with imshow
ax.imshow(img,aspect='equal',extent=plot_spatial_bounds)
plt.tight_layout()

# Save the python plot
plt.savefig('pyvista_2d_example.png')

# Close the figure
plt.close()