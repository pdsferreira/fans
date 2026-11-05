# FANS: cosmic-web spirality and handedness

Python analysis of **Filament-Arm Node Systems (FANS)**, including projected spirality and handedness, preferred-axis tests, results by arm multiplicity, and a 27-configuration robustness grid. **v88.15 (5 November 2026)** updates the v88.14 defaults to 1000 bootstraps and Planck18 auxiliary node geometry.

## Installation

Use Python 3.10+ and run from this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Input catalogues

Place the inputs in `data/`, or supply their paths through the command-line options.

| Catalogue | Files |
|---|---|
| [Chen: SDSS DR12](https://sites.google.com/site/yenchicr/catalogue-data) | `dr12_FMaps_full.txt` (filament points) and `dr12_Fint.txt` (intersections/nodes). |
| [Carrón Duque: SDSS DR16](https://www.javiercarron.com/catalogue) | Download and extract the three CSV blocks; use `Block1.csv`, `Block2.csv`, `Block3.csv`, or pass their actual paths with `--carron-files`. Keep the original columns, including density. |

**Carrón also requires an externally generated SCMS node reconstruction**, supplied with `--sconce-fits` (default: `data/nodes_scms.fits`). Reconstruct directional density ridges in each redshift shell, then identify junctions with a metric-graph knot detector and label the corresponding ridge points `Knot`. Use **Planck18 consistently, with the same sky selection and thin-shell boundaries as the filament reconstruction used for the association**. The default recomputes auxiliary node and Carrón shell distances with Astropy Planck18 from `z_low/z_high`; see the [input guide](docs/INPUTS.md).

Store the reconstructed points, including knot-labelled points, in a **FITS binary table in HDU 1** with these exact column names:

| Columns | Contents |
|---|---|
| `RA`, `DEC` | Equatorial coordinates in degrees. |
| `z_low`, `z_high` | Reconstruction-shell redshift boundaries. |
| `comov_dist_low`, `comov_dist_high` | Comoving boundaries in Mpc, used in explicit FITS-distance mode. |
| `bw`, `unc_meas`, `density` | SCMS bandwidth, reconstruction uncertainty, and density. |
| `grad_Dir1`, `grad_Dir2`, `grad_Dir3` | Cartesian components of the tangential gradient/direction on the sphere. |
| `knot_label` | `Knot` for junction candidates; `Non-Knot` otherwise. |

The pipeline clusters nearby knot points into nodes. It needs the reconstruction fields above; a three-column node-position list is insufficient. SCMS reconstruction and input data are not bundled.

## Run

```bash
python scripts/run_referee_revision.py fiducial \
  --data-dir data --sconce-fits data/nodes_scms.fits \
  --outroot results/fiducial --stage all --n-jobs 8 \
  --bootstrap-B 1000 --null-B 1000
```

Use `--stage projected` for projected statistics and multiplicity controls only, or add `--dry-run` to inspect commands without processing data. Outputs include the full sample, individual multiplicity classes, and fixed-composition checks for two- and three-arm systems. The four statistical redshift bins are distinct from the reconstruction shells.

See [running and outputs](docs/RUNNING.md) for the robustness grid and [release notes](docs/RELEASE_NOTES.md) for provenance and statistical settings.

## Citation

Please cite da Silveira Ferreira & Cen (2026), *Filament-Arm Node Systems (FANS): A New Probe of the Winding and Chirality of the Cosmic Web*: [ApJ Letters, 1011, L47](https://iopscience.iop.org/article/10.3847/2041-8213/aeb802) | [arXiv:2609.07792](https://arxiv.org/abs/2609.07792). Also cite the catalogue/method papers corresponding to your inputs.
