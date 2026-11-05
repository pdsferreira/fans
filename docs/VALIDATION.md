# Validation of v88.15

Date: **5 November 2026**.

## Checks completed

- Editable installation succeeded and reports package version `0.88.15`.
- **42 tests passed, with no failures or skips**, in 4.43 seconds: nine new v88.15 cases and the existing scientific revision tests.
- The new numerical tests verify that default node distances use Astropy Planck18 even when native FITS distance fields contain Planck15 or deliberately displaced values.
- Auxiliary and Carrón geometry agree for identical shells, including projected coordinates, comoving centres, and accepted shell matches.
- Explicit native FITS mode respects supplied distances and converts Mpc/h and Mpc consistently. A displaced native shell is rejected by the matching criterion as expected.
- Redshift boundaries and shell identifiers remain unchanged across distance modes.
- Command-line defaults and dry runs propagate 1000 bootstraps to the fiducial, multiplicity, and extrinsic stages. The grid retains its intentionally disabled redundant preliminary bootstrap, then runs the multiplicity bootstrap with 1000.
- The existing revision tests cover accepted-parent integrity, multiplicity/composition controls, resampling, HEALPix partitioning, FITS export, and extrinsic conventions.
- Source inventory, documentation links, ZIP integrity, and bundled checksums were checked when building the release.

## Tests run

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_defaults_v88_15.py \
  tests/test_revision.py
```

## Environment

| Component | Version |
|---|---|
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| Matplotlib | 3.10.8 |
| SciPy | 1.17.0 |
| Astropy | 8.0.2 |
| healpy | 1.20.1 |
| PyYAML | 6.0.3 |
| pytest | 9.2.0 |

## Scope

No survey-catalogue reconstruction or scientific inference was rerun. These checks validate the changed defaults, distance calculations, interfaces, and selected scientific regressions. They do not establish that a new external node reconstruction reproduces the published measurements. Other existing tests remain available but were not rerun for this update.
