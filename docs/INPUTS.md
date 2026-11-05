# Input catalogues and the SCMS node reconstruction

This guide describes the readers shipped with v88.15. The inputs are external to this repository. Preserve the catalogue selections, density weights, distance conventions, and thin reconstruction shells when preparing an analysis.

## Public filament catalogues

### Chen

Download **Filament catalogue** (`dr12_FMaps_full.txt`) and **Filament intersections** (`dr12_Fint.txt`) from the [authors' catalogue page](https://sites.google.com/site/yenchicr/catalogue-data).

The readers accept the native whitespace-separated files. The ridge reader recognises the original coordinate/direction headers and maps them to `ra, dec, z_low, v_ra, v_dec`. Headerless ridge data use the order `ra dec z_low density H UM v_ra v_dec`. Headerless intersections use `ra dec z_low` as their first three columns. Keep the native density and direction fields.

Nodes and ridges are associated by the original slice's lower-redshift boundary, rounded to five decimal places. Node IDs are assigned internally.

### Carrón Duque

Download the three compressed CSV blocks from the [authors' catalogue page](https://www.javiercarron.com/catalogue), extract them, and use `Block1.csv`, `Block2.csv`, and `Block3.csv`, or supply their actual paths:

```bash
--carron-files /path/to/block1.csv /path/to/block2.csv /path/to/block3.csv
```

The reader requires `ra, dec, angle, z_low, z_high` and normalises header capitalisation. Preserve `dens` for the density-weighted analysis and `unc` for reconstruction uncertainties; other original columns may remain. If `dens` is absent, the reader falls back to unit weights, which changes the intended analysis.

The files contain ridge points; the auxiliary node reconstruction described below is still required. The fiducial redshift range is 0.1–0.6, but the launcher expects all three input paths. The CDS/VizieR fixed-width `.dat` exports are not direct replacements for these CSV inputs.

## Preparing the auxiliary node reconstruction

This is an external preprocessing step. The FANS code reads a completed reconstruction and clusters its labelled knot points; it does not run SCMS on a galaxy catalogue.

1. Select the galaxy sample and sky mask for the intended reconstruction. Use Planck18 consistently and keep the original thin shell boundaries used for the node–ridge association. Reconstruct each shell separately with directional SCMS (`d=1`) on the sphere, retaining the ridge coordinates, directional density estimates, tangential gradient vectors, bandwidths, and reconstruction uncertainties.
2. Identify junction candidates on these ridge points using a metric-graph knot detector. In SCONCE, `DetectKnot` examines the ridge points in a neighbourhood annulus and returns `Knot` or `Non-Knot`. This classification is distinct from selecting local density maxima. Use a documented and fixed bandwidth/knot-detection prescription for the reconstruction.
3. Write the reconstructed ridge points and their knot labels to the FITS format below. Retain non-knot ridge points as well. The FANS pipeline selects the knot-labelled rows and combines nearby candidates within each shell using a 5 h⁻¹ Mpc linking length and density-weighted positions in the fiducial configuration.

The algorithm interfaces are documented in the [official SCONCE API](https://sconce-scms.readthedocs.io/en/latest/api_reference.html), including `DirSCMS`, `DirKDEGradLog`, and `DetectKnot`. This is a method reference, not a download of a node catalogue.

### Required FITS structure

Use a primary HDU followed by a binary table at **HDU index 1**. Column names are case-sensitive:

| Column | Type | Meaning/convention |
|---|---|---|
| `RA`, `DEC` | Floating point | Equatorial longitude/latitude, degrees. |
| `z_low`, `z_high` | Floating point | Lower/upper redshift boundary of the original reconstruction shell. |
| `comov_dist_low`, `comov_dist_high` | Floating point | Comoving shell boundaries in **Mpc**. Required by the FITS schema, but used only when native FITS distances are explicitly selected. |
| `bw` | Floating point | Native directional-reconstruction bandwidth. |
| `unc_meas` | Floating point | Native reconstruction uncertainty. |
| `density` | Floating point | Reconstructed density; used for weighting. |
| `grad_Dir1`, `grad_Dir2`, `grad_Dir3` | Floating point | Cartesian XYZ components of the Riemannian gradient/tangent direction on the unit sphere. |
| `knot_label` | FITS character string | `Knot` for candidates; `Non-Knot` for other ridge points. |

No external `node_id` or `shell_id` is required. Label matching strips whitespace and ignores capitalisation. Supply genuine reconstruction values and finite, usable tangent vectors: zero vectors do not define tangents and are excluded when slices are built.

All points in a shell must share consistent boundary and distance metadata. The auxiliary reader groups by `round(z_low, 5)`; do not combine distinct shells under the same rounded lower boundary.

## Cosmology and shell compatibility

A newly generated node reconstruction must be internally consistent with the cosmology, units, galaxy selection, and shell definitions used to generate its ridges and distances. Matching these choices is necessary for a reproducible node–filament association. Recreating the exact historical nodes also requires the original reconstruction settings; matching column names alone does not reproduce them.

**The default now uses the same Astropy Planck18 distance calculation for the auxiliary node geometry and Carrón ridge shells:**

| Part of the pipeline | Current convention |
|---|---|
| Auxiliary SCMS node geometry | Recomputes shell boundaries from `z_low/z_high` with Astropy `Planck18`; uses `h = 0.6766` and the mean of the two boundary distances for projection. |
| Carrón ridge shells | Uses the same Astropy `Planck18` boundary distances and `h = 0.6766`. |
| Chen geometry | Flat ΛCDM comoving distances in h⁻¹ Mpc with `Omega_m = 0.3153`. |

The default auxiliary mode is `distance_source=planck18`. Both shell matching and node geometry use distances computed from the redshift boundaries, so old distance columns in a FITS file cannot silently retain another cosmology. In h⁻¹ Mpc, the projected shell scale is `Planck18.h * (chi(z_low) + chi(z_high)) / 2`. This uses the full Astropy cosmology, including its radiation/neutrino treatment, rather than a flat matter–Lambda approximation.

The lower-level SCMS and Carrón adapters retain explicit `--distance-source fits` and `--distance-source zmid` options for legacy work. FITS mode reads native distances, converting h⁻¹ Mpc to Mpc for matching if `--fits-distance-units mpc_h` is selected. The legacy `zmid` mode uses the configurable flat matter–Lambda approximation. The fixed `planck18` mode ignores `--h` and `--omega-m` overrides. The high-level launcher uses the new Planck18 default.

Recalculating these distances does not regenerate the SCMS ridge coordinates or knot labels. Prepare any new auxiliary reconstruction with matching cosmology, selection, and shell definitions from the start. Chen's separate catalogue geometry retains its original `Omega_m = 0.3153` convention.

The Carrón adapter pairs each auxiliary shell with the Carrón shell nearest in comoving centre. The default acceptance requires an overlap of at least 50% of the narrower shell and a centre offset of at most 5 Mpc. It records the association in `sconce_carron_shell_matches.csv`; check its `accepted` column. This overlap check is not a test that the two reconstructions have identical shell boundaries or cosmological assumptions.

The broad analysis bins `[0.1, 0.225, 0.35, 0.475, 0.6]` are applied **after** reconstruction. They must not be substituted for the original thin SCMS shells.

## Reader locations

- `spirality/chen_node_spirality.py`: `read_chen_fmaps` and `read_intersections`.
- `spirality/carron_proxy_spirality.py`: `read_carron_blocks`, shell construction and matching.
- `spirality/sconce_node_spirality.py`: `read_sconce_fits`, `build_sconce_slices`, `build_nodes_from_sconce_knots`.

