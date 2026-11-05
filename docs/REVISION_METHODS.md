# Statistical definitions and interpretation — v88.1

This document describes implemented choices, not measured SDSS results. The numerical results must be regenerated before the manuscript is revised.

## 1. Analysis population and unchanged geometry

Let j identify a parent node and i one of its surviving arms. A scientific arm enters the population statistics if and only if its parent has `status == 'ok'` and at least two valid arms. All arms of an accepted parent must be present. Duplicate parents/arms, orphan arms and incomplete accepted systems are errors, not silently repairable cases.

The local segment construction remains density-weighted. The arm winding is h_i = sum_k N_ridge,ik s_ik / sum_k N_ridge,ik. The population statistics use W_i = N_valid_segments,i:

    H_proj = sum_i W_i h_i / sum_i W_i
    S_proj = sum_i W_i |h_i| / sum_i W_i
    A_sign = sum_i sign(h_i) / number(h_i != 0).

These are arm-level statistics. In particular, abs(mean of the signed arms of a node) is not the population spirality. The legacy `NODES.SPIRALITY` FITS column retains its node-level meaning and is explicitly labelled as such. Segment tables permit reconstruction of the arm statistic; the analysis CSVs store it directly.

## 2. Observed multiplicity and selection audit

The audit distinguishes raw seed angle groups, groups after the initial tangent-orientation filter, the capped candidate list and arms surviving contiguous radial tracing. These are operational stages, not a ground-truth graph connectivity. Even the raw-seed grouping imposes the linking radius, angular-gap and minimum seed-point rules. The input catalogues do not provide a common complete physical graph through this code path.

Thinning seed points can split an angular chain, so intermediate group counts need not decrease monotonically. Radial tracing stops at the first failed segment. A counterfactual trace is also recorded for edge-rejected nodes and for nodes that would otherwise exit before tracing. These diagnostics do not promote rejected parents or their arms into the science sample.

Unmatched/missing ridge slices are explicitly unassessed; their multiplicity is unknown, not zero. Input redshift selection, initial node clustering and the existence of suitable reconstructed slices remain prerequisites. The audit does not reconstruct missing physical filaments or undo selection already performed when producing the public catalogues.

Stage distributions have explicit populations: all processed/recorded nodes, edge-safe nodes and accepted FANS. Counts and both evaluated/population denominators are provided. The transition table tracks the same parents through candidate and survivor multiplicities; its rows are not comparisons of unrelated histograms.

## 3. Multiplicity-conditioned and composition-standardized statistics

The statistic S_m(z_b) is computed by selecting arms whose parent has observed multiplicity m, retaining the original arm weights. Node fractions f_m = N_FANS,m / N_FANS and estimator-weight fractions pi_m = sum_m W_i / sum_all W_i are separately reported.

The following identity is checked over all observed classes, including rare tails:

    S_all(z_b) = sum_m pi_m(z_b) S_m(z_b).

Fixed-composition standardization uses the predefined target classes m=2 and m=3. A pooled full-range reference pi_m* is the fraction of arm weight in each class within that target population. The standardized statistic is

    S_fixed(z_b) = pi_2* S_2(z_b) + pi_3* S_3(z_b).

The directly comparable observed-mixture statistic uses the same 2+3-arm target population. Its difference from S_fixed is reported. The unrestricted S_all is also saved; it is not silently equated with S_2+3 when rare higher-multiplicity systems exist.

The standardization is considered supported only if each target class has at least 10 FANS in at least 5 occupied spatial blocks in every bin (defaults recorded in metadata). These are diagnostic support thresholds, not a theorem about coverage. If support fails, standardized values are unavailable; no missing class is imputed, merged or selected after viewing its trend. Conditional class curves and counts are still supplied.

The reference fractions are re-estimated in every bootstrap draw and shared across all its bins, propagating estimation of the reference composition. A second saved bootstrap fixes the reference to its observed value, providing conditional-on-reference uncertainty as a diagnostic.

## 4. Joint spatial bootstrap and projected nulls

Production sky blocks are actual HEALPix RING pixels, NSIDE=16 by default. Occupied blocks are sampled uniformly with replacement. A selected block retains all its nodes and all their arms, and the same multiplicities are used across every redshift bin and arm class. Repeated blocks count repeatedly in sign-only as well as weighted statistics.

Each catalogue is processed separately; no independence between the two SDSS reconstructions is asserted. The grid is a set of correlated construction alternatives, not independent observations.

Reported bootstrap errors are sample standard deviations (ddof=1). Percentile summaries are also saved. Missing/zero-denominator realizations are NaN rather than zero; intervals/errors are suppressed when fewer than two valid draws or fewer than 90% valid draws are available. Conditional means can be reported even when their uncertainty is unsupported. Covariance matrices use one common complete set of draws per matrix, not pairwise deletion; matrix indexing and valid-draw counts are exported.

High-minus-low-redshift contrasts and OLS slopes over fixed bin centres summarize the shape of S(z). Their errors use the paired draws. Slope/error is labelled descriptive and is not a calibrated detection p-value. The test does not require strict monotonicity in every bin.

Projected sign-flip nulls preserve h_i amplitudes, positions, weights and membership, flipping signs coherently per occupied block. One draw is shared across bins/classes. H and A_sign have empirical two-sided absolute-value p-values with the +1 convention. S has no sign-flip p-value because |h_i| is unchanged by sign flips. Null resolution is 1/(B_null+1); an empirical p-value is never zero. Nonzero spatial correlations across different chosen blocks and survey systematics remain assumptions/limitations of this resampling scheme; unit tests do not establish their physical adequacy.

## 5. Extrinsic dipole and the monopole choice

For each trial direction d, mu_i = n_i dot d. The continuous fit always includes an intercept:

    h_i = H0(d) + A(d) mu_i + residual_i
    A = sum W(mu-mu_bar)(h-h_bar) / sum W(mu-mu_bar)^2
    H0 = h_bar - A mu_bar.

Thus adding a constant to all h_i changes H0, not A. Do not drop this intercept on an incomplete footprint.

The primary axial sign statistic follows the submitted Equation (6):

    q_i = sign(h_i mu_i),   A_ax = (N_q+ - N_q-) / (N_q+ + N_q-).

It is literally a folded count of the original signs. It can respond to a projected sign monopole combined with unequal hemispheric sampling; it should be interpreted alongside the continuous intercept-corrected dipole, not as an isolated pure dipole coefficient.

A separate diagnostic retains the previous residual definition:

    q_i,res = sign((h_i - H0(d)) mu_i).

The point statistic is invariant to an added constant offset when H0 is refitted. However, it is no longer strictly sign-only in the original h_i: its threshold depends on a continuous-amplitude fit and need not preserve each original CW/CCW label. Both definitions are saved; the primary is not switched after comparing significances. Nulls for the residual diagnostic refit H0 and reselect the dipole axis on each realization. They remain the stated sign-flip null of the observed amplitudes, not a separate conditional-nonzero-monopole null model.

The axis is chosen by max_d |A(d)|, as in the paper, over the positive-declination hemisphere. The same positive-declination representative is used for data, bootstrap and nulls; at the equator, the representative with RA<180 degrees removes the exact antipodal duplication. A and A_ax remain signed in this convention. Neither the full-sample sign nor every redshift-bin sign is forced positive. The absolute amplitude is used only where the selection/test statistic requires it.

## 6. Extrinsic uncertainties, axis selection and significance

The data bins use the observed full-range best axis. Each spatial bootstrap realization repeats the full-range directional search, refits the monopole and then evaluates all bins on that realization's single selected global axis. This propagates global axis estimation without introducing an independent per-bin optimization. In parallel, the same draws are evaluated on the observed fixed axis to provide explicitly conditional errors/covariances.

Formal extrinsic errors come from the actual bootstrap samples. Analytic independent-sign noise scales are retained only as diagnostics, not relabelled bootstrap errors. Empirical 68%/95% angular radii of bootstrap axes use headless angular distances, arccos(abs(d_boot dot d_obs)). These describe bootstrap directional spread; in a weak or absent signal they must not be advertised as a well-localized detected physical axis or as a proof of frequentist coverage. The former atan(sigma_A/A) angular surrogate is not the default.

Every sign-flip null repeats the complete global dipole search. The dipole p-value compares max|A| to its corresponding null maximum. Both axial sign definitions are evaluated only on each null's dipole-selected axis and calibrated separately. There is no second sign-optimized search in the supported revision launcher. Plot significance labels use the unbinned full-range optimized calibration; bins are correlated and not combined as independent detections.

Symmetric signed null envelopes use the 68.27% and 95.45% quantiles of absolute null values. This is a coverage convention, distinct from a one-sided or two-sided Gaussian-equivalent conversion of the global empirical p-value; both Z conversions remain available in the tables. Formal conclusions should refer to the stated global p-values. Dipole-fit coefficients are not mathematically bounded by unity on a partial footprint; angular displays return unavailable rather than saturating at 90 degrees if an arcsin argument exceeds its domain. The raw dimensionless estimates remain available.

The 27-point construction grid performs no extrinsic search, bootstrap axis scan or extrinsic look-elsewhere calibration. It tests projected quantities and their multiplicity/composition dependence using the same corrected construction stage.

## 7. Interpretation after rerunning

Persistence of the redshift trend at fixed observed multiplicity and fixed observed composition rules out a changing mixture of those observed classes as its sole explanation. It does not alone establish evolution at fixed physical graph connectivity, fix node mass/environment, measure selection completeness or identify a unique gravitational mechanism. If the control weakens/removes the trend, the abstract, results and discussion must acknowledge the composition contribution. No new numerical conclusion is assumed by this code release.
