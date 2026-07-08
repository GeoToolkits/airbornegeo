## Crossovers
It is often useful in airborne and shipborne surveys to examine data at points where individual lines of flights cross each other. These points are called crossovers, and we offer a range of functions for determining them, and calculating the difference between data values between the two lines which are intersecting. These are referred to as crossover errors, or sometimes misties.

### Finding crossover points

The below notebooks goes over the basics of finding crossover points, and the two main methods we have for doing this; `groups`, and `network`.

If your survey has two types of of lines, such as survey lines and orthogonal tie lines, you can use the `groups` method to find crossovers only between lines of these different types. Intersections where 1 survey line crosses another survey line, for example, would not be included.

The other method, `network`, will find all intersections between any line, regardless of it's type. This will included intersections between to survey lines, or two tie lines.

```{nbgallery}
crossovers_01_basics
crossovers_02_grouped_intersections
crossovers_03_network_intersections
```

### Crossover errors

The main reason for finding crossovers is to compare data values from each line where they intersect. These notebooks show how to do this. First we need to interpolate data values at these crossover points, since they rarely occur directly where a measurement was made, but between measurements. We can then compute the crossover errors.

These crossovers are only in 2D space (map view) since flight heights can differ between the crossing lines. For some data types (i.e. gravity or magnetics) the data value can change strongly depending on the observation height, and therefore it is important to compare the crossing data values at the same point in 3D space (accounting for altitude). The last notebooks shows how to use equivalent sources to account for this.

```{nbgallery}
crossovers_04_interpolating_values
crossovers_05_crossover_errors
crossovers_06_update_crossovers_with_equivalent_sources
```
