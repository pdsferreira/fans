# v88.1 selection clarification

The original compact NODES/SEGMENTS layout below is retained. QUALITY=1 requires an accepted multi-arm parent and is propagated by a validated parent join. NODES.SPIRALITY remains the absolute NODE-level signed average; it is NOT the arm-population S_proj. To reproduce Eq. (2)/(3), group SEGMENTS by FANS_ID and ARM_ID, combine T_PHI using NPOINTS, then weight arms by their valid segment counts. Main analysis CSVs store the arm quantities directly. Stage/selection audits are distributed as CSV, not silently folded into the FITS science columns.

# Public FANS FITS data model

`export_fans_fits.py` writes one binary FITS file per catalogue.  Each file has
three HDUs:

1. `PRIMARY`: run-level metadata;
2. `NODES`: one row per FANS node/candidate;
3. `SEGMENTS`: one row per measured arm segment.

The catalogue key is encoded in the filename, e.g. `carron_fans_all.fits`,
`chen_fans_all.fits`, or `sconce_fans_all.fits`, rather than repeated as a
string column.  The full export keeps all reconstructed candidates and uses
`QUALITY == 1` to mark the final accepted FANS sample.  The quality-only public
release files are named `*_fans_quality.fits` and contain only accepted systems.
Use `--write-quality-copy` to write both full and quality-only products in one
command, or `--accepted-only` to write only the quality file.

## `NODES` extension

| Column | Meaning | Units / convention |
|---|---|---|
| `FANS_ID` | Public string identifier for a FANS candidate. The catalogue prefix is included in the identifier. | String |
| `RA` | Right ascension of the node/FANS centre. | degrees |
| `DEC` | Declination of the node/FANS centre. | degrees |
| `Z` | Effective redshift used for binning and analysis. | dimensionless |
| `NARMS` | Number of valid filament arms linked to the node after the arm-quality criteria. | Integer |
| `NSEGS` | Total number of valid arm segments contributing to that FANS object. | Integer |
| `NPOINTS` | Total number of ridge/catalogue points contributing to the accepted segments. | Integer |
| `HANDEDNESS` | Signed node-level winding statistic. It is the weighted mean tangential component of outward-oriented filament tangents around the node. Positive/negative signs follow the pipeline handedness convention. | dimensionless |
| `SPIRALITY` | Winding amplitude, defined as `abs(HANDEDNESS)`. | dimensionless |
| `QUALITY` | Integer flag selecting the final accepted sample. `1` is equivalent to the internal pipeline cut `status == "ok"`; `0` means rejected candidate. | unsigned byte integer |

## `SEGMENTS` extension

| Column | Meaning | Units / convention |
|---|---|---|
| `SEG_ID` | Public string identifier for the segment. | String |
| `FANS_ID` | Identifier of the parent FANS node/candidate. | String |
| `ARM_ID` | Local arm identifier within the parent FANS. | Integer |
| `SEG_INDEX` | Radial segment index along the arm, starting from the node. | Integer |
| `RA` | Approximate right ascension of the segment centroid. It is reconstructed from node position plus local tangent-plane offset. | degrees |
| `DEC` | Approximate declination of the segment centroid. It is reconstructed from node position plus local tangent-plane offset. | degrees |
| `Z` | Parent-node/effective redshift used for the segment. | dimensionless |
| `NPOINTS` | Number of ridge/catalogue points used to estimate this segment tangent. | Integer |
| `X_MPC_H` | Local tangent-plane east/RA offset of the segment centroid relative to the node. This is the native pipeline coordinate. | h^-1 Mpc |
| `Y_MPC_H` | Local tangent-plane north/Dec offset of the segment centroid relative to the node. This is the native pipeline coordinate. | h^-1 Mpc |
| `RHO_MPC_H` | Projected radial distance from the node in the local tangent plane. | h^-1 Mpc |
| `PHI_DEG` | Polar angle of the segment centroid in the local tangent plane. | degrees |
| `T_PHI` | Segment-level signed tangential component of the outward-oriented filament tangent. This is the elementary winding contribution. | dimensionless |
| `T_RADIAL` | Segment-level radial component of the outward-oriented filament tangent. | dimensionless |
| `WEIGHT` | Segment weight used by the pipeline, typically the sum of point/density weights; when no separate weights exist it tracks `NPOINTS`. | dimensionless |
| `QUALITY` | Integer flag inherited from the parent node: `1` if the parent FANS is accepted, `0` otherwise. | unsigned byte integer |

## Notes

- `QUALITY` is stored as FITS unsigned byte (`TFORM=B`) rather than FITS logical (`TFORM=L`), because some tools display logical T/F as ASCII byte values 84/70.
- The public FITS products intentionally omit repeated catalogue strings,
  internal node IDs, raw status labels, redshift-slice bounds, radial-coherence
  diagnostics, and object-level uncertainty placeholders.  This keeps the files
  compact and focused on the quantities needed to reproduce the paper plots.
- We do not export a node-level `HANDEDNESS_ERR`: the paper uncertainties are
  catalogue/bin-level bootstrap and sign-flip/null uncertainties, not independent
  per-object measurement errors.
- `RA`/`DEC` in `SEGMENTS` are convenience coordinates for plotting.  The most
  faithful representation of the measured geometry is the local tangent-plane
  coordinate set `X_MPC_H`, `Y_MPC_H`, `RHO_MPC_H`, `PHI_DEG`, `T_PHI`, and
  `T_RADIAL`.
- The public FITS products are intended to reconstruct the central projected and
  extrinsic measurements.  Bootstrap/null resampling products are still stored
  in the pipeline CSV outputs.
