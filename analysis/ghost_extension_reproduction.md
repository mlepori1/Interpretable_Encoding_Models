# Ghost extension reproduction

The [dated notebook](ghost_spotlight_extension_2026-09-22.ipynb) reproduces
selection, anatomical quantification, and surface figures for the additional
Ghost voxel search. Its [HTML export](ghost_spotlight_extension_2026-09-22.html)
contains the saved tables and figures. This is postprocessing of the tracked
regression results, not a new regression run. The notebook uses this repository
alone, plus the external surface assets listed below; no private participant
metadata or sibling analysis repository is needed.

## Analysis definitions

The fixed signature is `40+, 94+, 79-, 44+, 49+, 64+, 199+, 16-, 669-, 884+`.
The upstream spotlight fits use these ten Matryoshka predictors in
unconstrained ridge regression, without surprisal or feature selection.
Directions are evaluated after fitting. This differs from the sign-constrained
source-to-target generalization analysis used to discover the signature.

The saved results use five shuffled sentence folds (`random_state=19`), with
leave-one-out ridge parameter selection within each training fold. The `raw`
results used here do not standardize the responses. Predictivity is
`tanh(mean(arctanh(fold_r))) / sqrt(nc / 100)`, where metadata `nc` is the
noise ceiling on the variance scale. A normalized correlation of 0.5 does not
mean that half of the explainable variance is explained.

Well-predicted voxels have normalized correlation >= 0.5. Feature
direction-aligned voxels additionally have >= 7 matching coefficient signs,
averaged across five folds; the same seven signs need not match in every fold.
Anatomical quantification retains participants with more than three aligned
voxels: P1, P2, P3, P5, and P8.

| Set | Count |
| --- | ---: |
| Additional Ghost candidates, eight participants | 1,490 |
| Well predicted | 275 |
| Well predicted and direction-aligned | 181 |
| Direction-aligned, five retained participants | 178 |
| Well predicted but non-aligned, same five participants | 86 |

Within those five participants, functional LTPJ contains 143/178 aligned
voxels (80.3%) and 20/86 non-aligned voxels (23.3%). The Glasser PGi/TPOJ1/2/3
supplement contains 138/178 (77.5%) and 10/86 (11.6%), respectively.
Denominators count participant-vertex observations, not unique mesh vertices.
None of the 320 Abstract/Concrete comparison voxels reaches 7/10 mean agreement.

The group map pools observations on the common fsaverage7 mesh; it is not a
participant-average activation map. Point color is the fold-averaged direction
count. Repeated vertex IDs are overplotted, with higher agreement drawn last.
Across eight participants, 181 aligned observations occupy 178 unique vertex
IDs. The notebook also reports pairwise overlaps and unique-vertex counts.

## Inputs and environment

The following inputs are already tracked in this repository:

- `results_anon/categories/spotlight/sae/gemma-2-2b-res-matryoshka-dc/mean/12/raw/regressions/p*/results.csv`
  for the additional Ghost search and comparison populations.
- `results_anon/categories/full_features/generalization_8000/sae/gemma-2-2b-res-matryoshka-dc/mean/12/standardized_betas/generalization/ghost.csv`
  and `data/processed_csvs_anon/p*/ghost_meta.csv` for the original 160 voxels
  and discovery-cluster supplement.
- `data/processed_csvs_anon/p2/ghost_additional_betas.csv` for the strict
  10/10 subset's response-correlation check.
- `data/processed_csvs_anon/p*/langfroi12345_meta.csv` for validation of language
  parcel assignments within the publicly released subset. This is not a
  validation against every language-network vertex.

The upstream fitting implementation is `src/spotlight_analysis.py`, supported
by `src/regression_utils.py`. The notebook uses the tracked
`analysis/generalization_utils.py` and `analysis/helpers.py`.

Install the CPU-only postprocessing environment with Python 3.10:

```bash
python -m venv .venv-ghost
source .venv-ghost/bin/activate
python -m pip install -r analysis/requirements-ghost.txt
```

Place the external masks and mesh files in `analysis/ghost_surface_assets/`,
or set `GHOST_SURFACE_ASSET_DIR` to their directory. The exact filenames and
reference SHA-256 hashes are in
[`ghost_surface_assets/manifest.json`](ghost_surface_assets/manifest.json).
These external assets are not bundled in this commit.

- The primary LTPJ mask is required for both numerical anatomy and surface
  figures. Obtain the source MNI volume from the Saxe Lab group maps and follow
  the [canonical neuromaps transform instructions](saxe_ltpj_surface_transform/README.md).
- The Glasser label volume and lookup table provide the anatomical supplement.
- The SN8 frontal language surface supplies parcels 1, 2, and 3, and the SN220
  surface supplies parcels 4 and 5, as encoded
  and checked in the notebook. Use the exact reference files in the manifest.
- Missing `lh.inflated` and `lh.sulc` are downloaded from the NSD fsaverage
  release by the notebook. The other assets must be provided beforehand.

## Run the notebook

Open the dated notebook from the repository root or `analysis/` and run its
cells in order. To execute and export it from the repository root:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 analysis/ghost_spotlight_extension_2026-09-22.ipynb
python -m jupyter nbconvert --to html --embed-images analysis/ghost_spotlight_extension_2026-09-22.ipynb
```

The saved notebook contains its analysis code; rebuilding is optional. After
editing the builder scripts, regenerate its cells with:

```bash
python analysis/build_ghost_spotlight_extension_notebook.py
python analysis/update_ghost_overlap_notebook_2026_09_22.py
python analysis/update_ghost_surface_notebook_2026_09_22.py
```

Rebuilding replaces the dated notebook and clears its outputs. Execute and
export again afterward. Figures and anatomical CSVs are written to
`analysis/figures/`; dated PDFs under `analysis/figures/preprint/` are paper-layout
copies and must be refreshed explicitly after changing figures. The date in
the filenames identifies this analysis snapshot, not the execution date.

The public port preserves the selection thresholds and fitted models. Its
verification re-executes postprocessing against saved regressions rather than
refitting all 1,490 voxels.

Run `python -m pytest -q tests/test_ghost_extension.py` for the offline checks
of surface-array formats, binary masks, vertex-index bounds, and inline
rendering setup. These checks do not need the external surface files.
