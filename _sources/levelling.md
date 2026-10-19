## Levelling
We offer a range of ways to perform levelling of your flight line data.

### Levelling to a grid
Supply a grid, for example satellite gravity data, and sample the grid values along each line. Then compare the grid values to the observed values to get a misfit at each point. Then level each line individually to minimize the misfit. This levelling can be a  vertical shift (trend order 0), tilting the line (trend or 1), or fitting a higher-order trend to the misfit.

```{nbgallery}
levelling_01_to_a_grid
```

### Levelling on crossover errors
Instead of levelling your data to match a grid, you can calculated the crossover errors between each of your lines and level lines to minimize them. You can either level between two categories of lines, such as tie lines and survey lines, referred to as tie-line levelling, or between any two intersecting lines, referred to as network-levelling. To obtain a levelling correction to apply to each line from it's crossover errors, you can choose between fitting a trend (order 0 for DC-shift, 1 for a tilt, 2 for a polynomial), or applying a low-pass spatial filter to the crossover errors.

```{nbgallery}
levelling_02_crossover_levelling_simple
levelling_03_crossover_levelling_alternating
levelling_04_crossover_levelling_network
levelling_05_crossover_levelling_trend_vs_wavelength
levelling_06_crossover_levelling_dependence_on_line_length
```

### Equivalent source levelling
If you don't have many of few cross-overs, you can level your lines to match the long-wavelength portion of other nearby lines. This works by fitting and equivalent source model to the nearby data to your line, then predicting the sources at your lines location, calculating a misfit, and fitting a trend, or low-pass filtering, this misfit to get a levelling correction for the line.

```{nbgallery}
levelling_07_equivalent_source_levelling
```

### Assessing levelling errors

```{nbgallery}
levelling_08_assessing_levelling_results
```
