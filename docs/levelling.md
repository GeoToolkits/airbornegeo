## Levelling
We offer a range of ways to perform levelling of your flight line data.

### Levelling to a grid
Supply a grid, for example satellite gravity data, and sample the grid values along each line. Then compare the grid values to the observed values to get a misfit at each point. Then level each line individually to minimize the misfit. This levelling can be a  vertical shift (trend order 0), tilting the line (trend or 1), or fitting a higher-order trend to the misfit.

```{nbgallery}
levelling_to_a_grid
```

### Levelling on cross-over errors
Instead of levelling your data to match a grid, you can calculated the cross-over errors between each of your lines and level lines to minimize them. You can either level between two categories of lines, such as tie lines and survey lines, referred to as tie-line levelling, or between any two intersecting lines, referred to as network-levelling. To obtain a levelling correction to apply to each line from it's cross-over errors, you can choose between fitting a trend (order 0 for DC-shift, 1 for a tilt, 2 for a polynomial), or applying a low-pass spatial filter to the cross-over errors.

```{nbgallery}
levelling_find_survey_intersections
levelling_cross_over_errors
levelling_cross_over_levelling_simple
levelling_network_levelling
levelling_cross_over_levelling_trend_vs_wavelength
levelling_cross_over_levelling_dependence_on_line_length
```

### Iterative levelling
If you have specific flight lines and tie lines, can also iteratively alternative between levelling the lines to the ties, and then the ties to the lines.

```{nbgallery}
levelling_cross_over_levelling_alternating
```

### Weighted levelling
Instead of relying on all cross-over points to an equal amount, you can weight them individually. Calculated levelling correction values will depends strongly on cross-over points with high weights, and weakly on cross-over points with low weights. These weights can be decided on based on a range of factors, such as distance to the nearest observation point, altitude difference between the crossing lines, the 1st or 2nd derivatives of either lines data or elevation, which might indicate the portion of the flight had turbulence.

### Upward continued cross-overs
If your crossing lines have drastically different altitudes at a cross-over point, the mistie value may reflect that your observation of your field are at different points in 3D space (same horizontal coordinates but different elevations) and therefore you would expect them to have different values. To account for this, when calculating the cross-over misties, you can choose to do it at the same point in 3D space. We do this by fitting equivalent sources individually to each line (in 1D), and predicting the field values at the same point in 3D space. This is just to determine the mistie, but leaves the data at its original observation locations.

```{nbgallery}
levelling_using_equivalent_sources_for_cross_over_errors
```

### Equivalent source levelling
If you don't have many of few cross-overs, you can level your lines to match the long-wavelength portion of other nearby lines. This works by fitting and equivalent source model to the nearby data to your line, then predicting the sources at your lines location, calculating a misfit, and fitting a trend, or low-pass filtering, this misfit to get a levelling correction for the line.

```{nbgallery}
levelling_equivalent_source_levelling
```

### Assessing levelling errors

```{nbgallery}
levelling_assessing_levelling_results
```
