# Overview


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
