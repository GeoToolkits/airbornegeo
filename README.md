## Template Instructions

Steps:
- [ ] 7) Set up automated Zenodo releases (only for Public repositories)
    - [ ] if you haven't already, link your organization (or personal) GitHub account to [Zenodo](https://zenodo.org/) using `Linked accounts` under your Zenodo profile.
    - [ ] go to the `GitHub` menu on your Zenodo profile.
    - [ ] click the Sync button and then turn on the switch for your repository.
    - [ ] any future GitHub releases should now result in a new Zenodo release and DOI automatically.
- [ ] 8) Set up publishing on TestPyPI
    - before publishing to the real PyPI, we will publish to Test-PyPI to ensure everything works .
    - [ ] make an account on [TestPyPI](https://test.pypi.org/).
    - [ ] under 'Your Projects', and 'Publishing', 'Add a new pending publisher', fill out your info.
        - [ ] the project name and repository name should be what you chose for `samplepackagename`.
        - [ ] the owner should be what you used from `organizationname`.
        - [ ] Workflow name should be `cd.yml`
        - [ ] Environment name should be `pypi`
    - note that this doesn't reserve your package name until you make your first actual release!
- [ ] 9) Make a GitHub release
    - [ ] On the main GitHub page, on the right side click `Create a new release`.
    - [ ] click `Select tag` and type `v0.0.1`.
    - [ ] set a Release title: "Initial release"
    - [ ] click `Publish release`
    - this should automatically trigger a few things:
        1) a DOI will be added to your Zenodo, add this DOI to `docs/citing.md`.
        2) the GitHub action `CD` should be triggered, create a release of `v0.0.1` to TestPyPI.
            - [ ] to test this worked correctly, run the below commands to create a new conda environment using the TestPiPI release, and run your codes tests.
            ```bash
            mamba  create --name test_pypi python
            mamba activate test_pypi
            pip install -i https://test.pypi.org/simple/ samplepackagename
            nox -s test
            ```
- [ ] 10)  Make a PyPI (pip) release
    - if the install and test above worked then we can change from TestPyPI to normal PyPI.
    - [ ] in `.github/workflows/cd.yml` comment out or delete the last line: `repository-url: https://test.pypi.org/legacy/`. Now any future reruns of this action will release to PyPI.
    - [ ] in this case, since the GitHub release has already been made, we need to manually trigger the `CD` workflow in GitHub.
        - [ ] At [this link](https://github.com/organizationname/samplepackagename/actions/workflows/cd.yml), click the `Run workflow` button.
        - [ ] This should build the package and release it to PyPI.
- [ ] 11) Set up publishing on Conda-Forge
    - [ ] create a [conda-forge recipe and feedstock](https://conda-forge.org/docs/maintainer/adding_pkgs/#creating-recipes) with the below instructions:
        - [ ] Create a new environment using : `mamba create --name grayskull grayskull`
        - [ ] Activate this new environment : `conda activate MY_ENV`
        - [ ] Generate the recipe : `grayskull pypi --strict-conda-forge https://github.com/organizationname/samplepackagename`
        - [ ] this should create a `meta.yaml` file, which is the recipe.
    - [ ] fork and clone the [stage-recipes](https://github.com/conda-forge/staged-recipes) repository on conda-forge.
    - [ ] checkout a new branch : `git checkout -b samplepackagename`
    - [ ] create a new folder : `staged-recipes/recipes/samplepackagename`
    - [ ] copy your `meta.yaml` file into this folder, remove the `meta.yaml` from wherever it was generated.
    - [ ] commit and push your changes to your fork, and in the main [repository](https://github.com/conda-forge/staged-recipes) open a PR.
- [ ] 12) Set up GitHub Pages to host the documentation website
    - this will only work for a public repository
    - [ ] On the GitHub repository, click the `main` button in the upper left
    - [ ] type "gh-pages" and create a new branch
    - [ ] go to your repositories settings -> `Pages`
    - [ ] for `Source`, choose "Deploy from a branch"
    - [ ] for `Branch`, choose "gh-pages"
    - [ ] for `Select folder`, choose "root"
    - [ ] optionally choose a custom URL and hit save
    - [ ] in each PR there should be a preview for the new site
    - [ ] for every push to `main` (from a PR), the site will be updated
- [ ] 13) Finalize
    - [ ] remove all the above instructions and raise any issues in this template's repository if you have any suggestion or found any errors in this template!
    - [ ] replace all instances of "zenodo_DOI" in this repository with your new Zenodo DOI.
        - if your Zenodo link is https://doi.org/10.5281/zenodo.15863068,, your DOI is just 15863068
        - make sure to use the DOI from the `Cite all versions?` portion of your Zenodo page.
    - [ ] if you want, please submit a PR to this repository to add your package to this list at the top of this README to showcase it!

# AirborneGeo
Tools for processing airborne geophysics data.

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
[actions-badge]:            https://github.com/mdtanker/airbornegeo/workflows/CI/badge.svg
[actions-link]:             https://github.com/mdtanker/airbornegeo/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/airbornegeo
[conda-link]:               https://github.com/conda-forge/airbornegeo-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/mdtanker/airbornegeo/discussions
[pypi-link]:                https://pypi.org/project/airbornegeo/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/airbornegeo
[pypi-version]:             https://img.shields.io/pypi/v/airbornegeo
[website-badge]:                https://readthedocs.org/projects/airbornegeo/badge/?version=latest
[website-link]:                 https://airbornegeo.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

<!-- SPHINX-END-badges -->
