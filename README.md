# Global Outage Detection with OWL-I

Code for "Global Power Outage Detection at the Kilometer Scale from Satellite Nighttime Lights". Preprint available on EarthArXiv: Rice et al. 2026 (https://doi.org/10.31223/X5B507)

OWL-I (the **O**utage Frame**w**ork for **L**uminosity **I**nterruptions) is the first 1-km resolution, standardized, calibrated, and reproducible global outage detection method derived from satellite nighttime light observations, described in an upcoming publication Rice et al. (in review).

The OWL-I USA dataset, produced with this model, is available on Zenodo [here](https://doi.org/10.5281/zenodo.20433557)

If you are interested in additional OWL-I data covering other regions, please contact me!  julian.rice@pnnl.gov

## Usage

This codebase is designed to be run on the NERSC supercomputer, but will likely run on other systems.


1. Create the appropriate conda environment using the environment specs files in `env`
2. Set (export) the `HIGHRES_OUTAGE_DATA_DIR` and `HIGHRES_OUTAGE_OUTPUT_DIR` environment variables to be paths pointing to where you want to load data and save outputs to, respectively. On NERSC, this perhaps could be `$SCRATCH/data` and `$SCRATCH/output/highres_outage/`
3. src/download_nasa_nightlight.py: Use to download nightlight data of your choice
    1. NOTE: Full CONUS nightlight 2012-2024 is at least 700 GB of data
4. compute_linear_correction.py: De-trend seasonal/angular/lunar effects from nasa data
5. preprocess_eaglei.py: Pre-compute nightly eaglei samples
6. outage_nightlight_align.py: Create a dataset of aligned outage and nightlight samples to evaluate the relation between nightlight and eagle-i
7. nl_to_outage_model.py: Defines model architecture and trains model

