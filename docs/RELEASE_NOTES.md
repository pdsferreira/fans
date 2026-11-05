# Release notes and distribution provenance

## v88.15 — 5 November 2026

The author confirmed that the final published analysis used **1000 bootstrap resamples**. This release sets 1000 as the default in the fiducial/grid launcher, multiplicity analysis, direct extrinsic CLI, configuration example, and auxiliary bootstrap calculations used by legacy plotting paths. Existing explicit overrides and disabled-bootstrap stages retain their meaning.

Auxiliary node geometry now defaults to **Astropy Planck18**, matching the Carrón shell-distance convention. The code recomputes the comoving distances to the original `z_low` and `z_high` boundaries, uses their mean for the projected shell scale, and applies `Planck18.h`. The same distance policy is used for shell association. Native FITS distances are used only when explicitly requested; Mpc and h⁻¹ Mpc are handled consistently in that mode. The original redshift shells are preserved. Chen's separate catalogue geometry is unchanged.

The README and GitHub Description now link to both the [ApJ Letters publication](https://iopscience.iop.org/article/10.3847/2041-8213/aeb802) and [arXiv:2609.07792](https://arxiv.org/abs/2609.07792).

Nine new regression cases exercise Planck18 distances, auxiliary/Carrón geometry and shell matching, native-distance unit conversion, unchanged redshift boundaries, and propagation of the 1000-bootstrap default. See [validation](VALIDATION.md).

These are updated scientific run settings. Use a new output root when moving from the earlier distribution; changing defaults does not update old result files or regenerate an external SCMS reconstruction.

## Recovered source

This distribution is based on `spirality_v88_14_matrix_format.zip`, saved on **18 October 2026 at 13:13:21 Asia/Shanghai (05:13:21 UTC)**. The preceding `spirality_v88_13_colourbar.zip` was saved the same day at **12:31:12 Asia/Shanghai (04:31:12 UTC)**. The v88.13-to-v88.14 changes concern matrix presentation and version metadata; v88.15 adds the scientific defaults described above.

The first GitHub preparation replaced personal absolute input paths with `data/` defaults, used a generic auxiliary FITS filename, added consistent editable-install metadata, retained scientific modules/scripts/tests, and replaced accumulated working documentation with concise public instructions. Historical logs, captions, comparison outputs, and obsolete manifests were removed.

Earlier recovered multiplicity metadata recorded `B=500`, `null_B=1000`, and `nside=16`. Those earlier metadata have not been rewritten. The current 1000-bootstrap default follows the author's confirmation of the final paper run.

Input catalogues, an auxiliary reconstruction, and complete survey-run products are external. No software licence was declared in the recovered archives, and this distribution does not introduce one on the authors' behalf.

The source fingerprint changes with this release, so the launcher's compatibility check may reject a resume of older runs. [SOURCE_PROVENANCE.json](../SOURCE_PROVENANCE.json) records the source archives and file-level changes; `SHA256SUMS.txt` covers the files actually bundled in this release.
