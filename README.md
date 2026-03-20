# AirborneGeo
Tools for processing airborne geophysical survey data.

This package allows for a range of processing steps necessary for airborne geophysical data. Some of the functions are generic for many types of airborne surveys, such as splitting flights into segments, calculating quantities such as distance along lines, velocities, bearings, and cross-over errors, etc. However, there are many functions specifically focused on gravity and magnetic data. These include field reductions, 1D equivalent source inversion, and levelling of flight lines.

## Levelling
We offer a range of ways to perform levelling of your flight line data.

### Levelling to a grid
Supply a grid, for example satellite gravity data, and sample the grid values along each line. Then compare the grid values to the observed values to get a misfit at each point. Then level each line individually to minimize the misfit. This levelling can be a  vertical shift (trend order 0), tilting the line (trend or 1), or fitting a higher-order trend to the misfit.

### Levelling on cross-over errors
Instead of levelling your data to match a grid, if you have sets of orthogonal flight lines, you can calculated the cross-over errors and level lines to minimize them. This can be of any order, from a simple vertical shift to fitting a higher order trend to the misties.

### Iterative levelling
If you have specific flight lines and tie lines, can also iteratively level the lines to the ties, and then the ties to the lines.

### Weighted levelling
Instead of relying on all cross-over points to an equal amount, you can weight them individually. Calculated levelling correction values will depends strongly on cross-over points with high weights, and weakly on cross-over points with low weights. These weights can be decided based on a range of factors, such as distance to the nearest observation point, altitude difference between the crossing lines, the 1st or 2nd derivatives of either lines data or elevation, which might indicate the portion of the flight had turbulence.

### Upward continued cross-overs
If your crossing lines have drastically different altitudes at a cross-over point, the mistie value may reflect that your observation of your field are at different points in 3D space (same horizontal coordinates but different elevations) and therefore you would expect them to have different values. To account for this, when calculating the cross-over misties, you can choose to do it at the same point in 3D space. We do this by fitting equivalent sources individually to each line (in 1D), and predicting the field values at the same point in 3D space. This is just to determine the mistie, but leaves the data at its original observation locations.


[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][website-badge]][website-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![Zenodo][zenodo-badge]][zenodo-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/airbornegeo/airbornegeo/workflows/CI/badge.svg
[actions-link]:             https://github.com/airbornegeo/airbornegeo/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/airbornegeo
[conda-link]:               https://github.com/conda-forge/airbornegeo-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/airbornegeo/airbornegeo/discussions
[pypi-link]:                https://pypi.org/project/airbornegeo/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/airbornegeo
[pypi-version]:             https://img.shields.io/pypi/v/airbornegeo
[website-badge]:            https://github.com/airbornegeo/airbornegeo/actions/workflows/pages/pages-build-deployment/badge.svg
[website-link]:             https://airbornegeo.github.io/airbornegeo/
[zenodo-badge]:            https://zenodo.org/badge/DOI/10.5281/zenodo.zenodo_DOI.svg
[zenodo-link]:             https://doi.org/10.5281/zenodo.zenodo_DOI
<!-- prettier-ignore-end -->

<!-- SPHINX-END-badges -->
