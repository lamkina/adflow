.. _adflow_slices:

Slices
======

ADflow can write output for planar slices of the surface flow solution as ASCII formatted files.
The slice cuts across the surface mesh and interpolates the solution at the intersection between the planar cut and the grid lines.
These files contain:
    - Variable Names, including x, y, and z coordinates
    - Corresponding data for each variable
    - Unsorted finite element bar connectivity

The ASCII files can be loaded into Tecplot natively for quick viewing and postprocessing.

Additionally, ADflow provides a postprocessing feature that will read the slice file, sort the connectivity, and return ordered finite element curves.
The finite element curves contain information about the parent slice (mesh family, type, number, etc.), the ordered coordinates, the ordered connectivity, and the ordered data.
Depending on the topology of the mesh and setup of the slices, each slice may contain one or more open or closed curve.
The postprocessing module can handle any number of open or closed curves contained in a single slice.

The following snippet is an example of how to postprocess slices to finite element curves::

    from adflow import SliceInterface
    from pathlib import Path

    # Set the path to the slice file
    sliceFile = Path("path/to/slice/file.dat")

    # Instantiate the interface
    interface = SliceInterface(sliceFile)

    # Create the curves
    curves = interface.extractCurves()

Also, you can extract the unordered raw slices::

    from adflow import SliceInterface 
    from pathlib import Path

    # Set the path to the slice file
    sliceFile = Path("path/to/slice/file.dat")

    # Instantiate the interface
    interface = SliceInterface(sliceFile)

    # Create the raw slices
    slices = interface.extractSlices()

Sometimes you may only want to process a select range of slices from the slice file.
To do this, we take advantage of the default python :py:data:`slice` object::

    from adflow import SliceInterface
    from pathlib import Path

    # Set the path to the slice file
    sliceFile = Path("path/to/slice/file.dat")

    # Instantiate the interface
    interface = SliceInterface(sliceFile)

    # Create the python slice range, i.e slice(start, stop, step)
    sliceRange = slice(2, 10)

    # Create only curves for slices 2 through 9
    curves = interface.extractCurves(sliceRange)

Once the curves are created, you can access the data as follows::

    # Get a select curve once we have created all the curves using the interface
    selectCurve = curves[0]
    coordinates = selectCurve.coordinates
    data = selectCurve.data
    info = selectCurve.info

    print(f"Available variables in the dataset include: {list(data.keys())}")

    cp = data["CoefPressure"]

The data can be easily plotted using matplotlib or some other postprocessing package::

    import matplotlib.pyplot as plt

    # Get a select curve once we have created all the curves using the interface
    selectCurve = curves[0]
    xoc = selectCurve.data["XoC"]
    cp = selectCurve.data["CoefPressure"]

    plt.plot(xoc, cp, color="b")
    plt.savefig("slice_cp.png")