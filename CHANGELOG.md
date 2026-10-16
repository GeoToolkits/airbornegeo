# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### 🚀 Added
- function `turn_rate` for calculating how fast the aircraft is turning, in degrees per kilometre, and the matching `Survey.turn_rate` method.
- function `split_at_turns` for splitting flights into lines wherever the aircraft turns, and the matching `Survey.split_at_turns` method.
- `split_into_segments` can now split existing lines via `segment_column`, naming the new sub-lines by extending the existing names (i.e. `12` -> `12`, `12_1`).
- `absolute` option in `split_into_segments` to split on decreases as well as increases.
- method `Survey.plot_intersections` for mapping crossover errors.
- method `Survey.estimate_error_wavelength` for estimating the along-line wavelength of levelling error.
- `Survey.line_sampling_rates` and `Survey.median_sampling_rates` statistics, now shown in `Survey.describe()`.
- `line` option in `Survey.plot` to plot a single line.
- `column_name` option in `Survey.along_track_distance`, which sets the survey's distance column.
- `plot_profiles` now supports up to 4 y axes and a title per axis, and colors an axis to match its series if it only holds one.

### ✏️ Changed
- `alternating_iterative_line_levelling` now levels the tie lines before the flight lines.
- `plot_line_and_crosses` now defaults to `plot_inters=True`.
- `Survey.plot_profiles` now requires a `line` and plots only that line.
- `Survey.split_into_segments` now splits each existing line on its own, if the line column exists.
- `split_into_segments` returns string names when `segment_column` is given.
- `Survey` now warns if supplied column names are not in the dataframe.
- `Survey.median_line_spacings` is computed on every 2nd point, for speed.
- `Survey.describe()` formats values to 3 significant figures, with units.

### 🐛 Fixed
- fixed `filter_line` failing on repeated coordinates (e.g. duplicated timestamps).
- fixed `level_to_grid` ignoring `groupby_column` when assigning corrections.
- fixed `plot_levelling_convergence` failing for runs which converged on their first iteration.
- fixed `plot_line_and_crosses` failing when intersections haven't been added to the data.


<!--
Below is an example for a release

## v0.0.1
Released on 2025-06-04
Contributors:
- author1
- author2

### 🚀 Added
- function `function2` for calculating something important.

### ✏️ Changed
- Changed the lower bounds on the pandas dependency from v1.5 to v1.2.

### ⚠️ Deprecated
- marked `function1` as deprecated, used the new `function2` instead.

### ❌ Removed
- removed the previously deprecated `function0`.

### 🐛 Fixed
- fixed a bug in `function1` which cause some error.

### 🛡️ Security
- removed a password from the code.

-->
