# Natural Stories replication

This package reproduces the sparse-encoding analysis and its four appendix
panels from released Natural Stories responses. It needs no other repository
checkout. Run commands from the `Interpretable_Encoding_Models` root.
Generated data, features, fits, and figures stay in ignored directories
beneath this package.

## What comes from the original pipeline

We use Andrea de Varda's released nine timestamped transcripts and
group-averaged shift-3 responses for IFGorb, IFG, MFG, AntTemp, and PostTemp.
The network-average response is also available, but is not an additional
anatomical region. Inputs are downloaded from immutable upstream commit
`a8a4729c11c5846c22453455d9c2fd885f77ea18` and verified against
[`data_manifest.json`](data_manifest.json), rather than copied into this Git
repository. The 15 required files total 247,002 bytes.

The original [response preparation](https://github.com/Andrea-de-Varda/multilingual-models-brain/blob/a8a4729c11c5846c22453455d9c2fd885f77ea18/additional_analyses/NaturalStories/natstor_fmri_timeseries.py)
and [encoding code](https://github.com/Andrea-de-Varda/multilingual-models-brain/blob/a8a4729c11c5846c22453455d9c2fd885f77ea18/additional_analyses/NaturalStories/fit_encoding_natstor.py)
define the response series, word-end-time assignment to two-second bins,
mean pooling, empty-bin imputation, and leave-one-story-out evaluation.
We retain those conventions and use the released shift directly, without
adding another hemodynamic shift.

Our additions are Gemma layer-12 SAE features, an explicit surprisal
predictor, F-screening and L1 selection followed by L2 refitting, a
100-second cutoff, and story-wise inner parameter selection. These are
extensions of the upstream pipeline, not its unchanged dense RidgeCV model.

## Environment and inputs

Use Python 3.10. Fitting and plotting need only the CPU requirements:

```bash
python -m venv replications/natural_stories/.venv
source replications/natural_stories/.venv/bin/activate
python -m pip install -r replications/natural_stories/requirements.txt
python -m replications.natural_stories.prepare_data
```

`prepare_data` validates existing files and never overwrites mismatches.
Use `--verify-only` to check a downloaded dataset, or `--source-dir PATH`
to import the same hash-verified inputs from an existing NaturalStories
directory. Neither option changes the input definitions.

## Feature extraction

Fresh extraction additionally requires a CUDA-capable machine, the optional
dependencies below, and Hugging Face access to Gemma. No cluster scheduler
or lab-specific account is required by these scripts.

```bash
python -m pip install -r replications/natural_stories/requirements-features.txt
NSTORIES_DIR=replications/natural_stories
for representation in surprisal matryoshka jumprelu; do
  python -m replications.natural_stories.extract \
    --data-dir "$NSTORIES_DIR/data" \
    --output-dir "$NSTORIES_DIR/runs/$representation" \
    --representation "$representation"
done
```

Matryoshka and JumpReLU extraction also save the corresponding residual
stream, so no separate residual extraction is needed. Features are saved
under each output directory's `features/` subdirectory. Use
`--stories 1` for an extraction smoke test and `--local-files-only` to
prohibit checkpoint downloads.

An optional whole-sentence consistency check is available for SAE extraction
with `--validation-reference-dir PATH`. That directory must contain
`stimset.csv` (a `sentence` column), `residual.pkl`, and `sae.pkl`: the latter
two are trusted NumPy-array pickles with matching sentence rows, from Gemma
layer 12 and the selected SAE release. The first five sentences are checked
against their reference vectors. These reference features are not included
in this package and are not required for extraction; missing files are
reported before loading a model. Reference hashes are recorded and checked
when resuming. This check covers whole-sentence features, not the batched
word-offset path.

Checkpoint defaults are pinned in [`extract.py`](extract.py): Gemma revision
`c5ebcd40d208330abc697524c919956e692655cf`, Matryoshka revision
`4c28e790b0213430d52dd8e5a09f681a53f5b79d`, and JumpReLU revision
`fd571b47c1c64851e9b1989792367b9babb4af63`. The historical LM revision was
recorded; the SAE revisions were recovered from the available cached
snapshots, not from historical extraction manifests. New manifests record
the actual revisions, model-artifact hashes, input/output hashes, extraction
settings, and software versions. The pinned requirements capture the tested
core versions, not a complete transitive environment lock.

## Fitting the models for the four panels

The following uses only the five individual fROIs. It omits the unused
finer-bin union and fits only the residual-stream circular control shown in
panel A. Each output subdirectory contains per-story correlations, selected
features and coefficients, held-out predictions, and provenance records.

```bash
NSTORIES_DIR=replications/natural_stories
for roi in IFGorb IFG MFG AntTemp PostTemp; do
  for model in surprisal residual sae_full sae_0_128 sae_128_512 sae_512_2048 sae_2048_8192 sae_8192_32768; do
    python -m replications.natural_stories.fit \
      --data-dir "$NSTORIES_DIR/data" \
      --feature-dir "$NSTORIES_DIR/runs/matryoshka/features" \
      --surprisal-dir "$NSTORIES_DIR/runs/surprisal/features" \
      --output-dir "$NSTORIES_DIR/runs/encoding_actual/$model" \
      --roi "$roi" --models "$model" --lasso-jobs 1
  done
  python -m replications.natural_stories.fit \
    --data-dir "$NSTORIES_DIR/data" \
    --feature-dir "$NSTORIES_DIR/runs/jumprelu/features" \
    --surprisal-dir "$NSTORIES_DIR/runs/surprisal/features" \
    --output-dir "$NSTORIES_DIR/runs/encoding_actual/jumprelu" \
    --roi "$roi" --models jumprelu --lasso-jobs 1
  python -m replications.natural_stories.fit \
    --data-dir "$NSTORIES_DIR/data" \
    --feature-dir "$NSTORIES_DIR/runs/matryoshka/features" \
    --surprisal-dir "$NSTORIES_DIR/runs/surprisal/features" \
    --output-dir "$NSTORIES_DIR/runs/encoding_circular_shift/residual" \
    --roi "$roi" --models residual --response-shifts 26 52 78 104 --lasso-jobs 1
done
```

For a one-fold smoke test, add `--held-out-stories 1` and use a separate
output directory. All nine input stories are still needed to form the
training and test sets. Resume is allowed only for identical input contents,
settings, code, and library versions. Use a new output directory after
changing a configuration; do not resume a partial smoke test as a full run.
Legacy result CSVs can be plotted directly but are not silently adopted as
new fitting checkpoints.

## Figures

```bash
python -m replications.natural_stories.plot \
  --actual-dir replications/natural_stories/runs/encoding_actual \
  --control-dir replications/natural_stories/runs/encoding_circular_shift \
  --output-dir replications/natural_stories/runs/figures
```

This saves only the four manuscript panels, as PDF/PNG/SVG files with
editable PDF fonts, and the numerical tables underlying them. Paths can
point to existing saved fits; plotting does not require extraction or refitting.

| Panel | Definition |
| --- | --- |
| A | Mean raw Pearson correlation and SEM across nine held-out stories. Control points are four shift-specific nine-story means, with SEM across those four values. Delta compares surprisal with the mean of JumpReLU, Matryoshka, and residual models. |
| B | Number of features selected in each Matryoshka bin by the full model, averaged across nine fits per fROI, with SEM. These are counts, not coefficient magnitudes. |
| C | Separate restricted-bin models, each including surprisal. Each dot first averages the five fROI scores within one held-out story; bars and SEM summarize those nine story means. |
| D | Top 15 signed features across five fROIs × nine folds: maximum 45 fits. Inclusion is determined by L1 selection; direction is the final ridge coefficient's sign. |

Panel D now excludes the network average. To reproduce the former 54-fit
version explicitly, use `--feature-count-targets all` and provide the nine
`Lang` full-Matryoshka fits as well. No regression refit is needed when those
fits already exist. In the saved results, 40+ and 79− remain selected in all
45 regional fits; 14 of the former top 15 remain, with 194+ replacing 29743−.

## Exact methods and manuscript clarifications

- Context contains at most 100 words **including the target**. SAE encoding
  is applied to target-subtoken states before averaging within words.
  Word surprisal instead **sums** subtoken negative log-probabilities, then
  is averaged across words in each two-second bin. Word count is excluded.
- Word-end timestamps define bin membership; a word ending exactly at a
  bin boundary enters the preceding bin. Empty bins use the same story's
  mean nonempty-bin predictors. The released inputs contain five such
  trailing bins, across four stories.
- Only the eight outer-training stories affect selection and tuning.
  Screening retains **at most** 8,000 candidates, including surprisal when
  it survives screening; smaller feature spaces are not expanded. Surprisal
  is always restored in the final ridge design.
- The first 50 bins of each story are omitted from feature-selection rows
  and inner/outer scoring. Inner and final ridge training use all training
  bins. Surprisal scaling also uses all outer-training bins, so the cutoff
  does not remove their indirect influence through this scaling step.
- This preserves the original extension's tuning procedure: F-screening
  is done once per outer fold before LassoCV, and the selected support is
  held fixed during inner ridge validation. It is **not** a fully nested
  end-to-end feature-selection pipeline, although the outer test story
  remains excluded. LM/SAE features retain native scales; surprisal and
  response scaling use training data. L1 and L2 parameters minimize inner
  validation MSE; reported predictivity is held-out Pearson correlation.
- Averages in these figures use raw correlations, not Fisher averaging or
  noise-ceiling normalization. No new significance tests are plotted.
- In the results text, say that **several prominent signed features
  overlap with the 7T results**, rather than implying all top features do:
  40+, 79−, 44−, and 82+ occur in both signed top-15 lists.
- The manuscript's blanket TitanRTX featurization statement needs an
  exception: the saved Natural Stories extraction manifests record an
  NVIDIA H100 80GB GPU.

## Verification

```bash
python -m pytest -q tests/test_natural_stories_data.py tests/test_natural_stories_features.py tests/test_natural_stories_fit.py tests/test_natural_stories_plot.py
```

Tests cover immutable input validation, context/token masks, pooling,
surprisal aggregation, cutoff bookkeeping, forced surprisal, cache safety,
and panel denominators. The saved historical correlations were independently
recomputed from their predictions (maximum difference below 9e-8), and the
new plotting summaries were checked against those saved results. These
checks are not a fresh GPU extraction or a complete rerun of all fits.

A representative AntTemp full-SAE fit (held-out story 1) was also rerun
against the downloaded responses and existing feature cache: it reproduced
`r = 0.4329907408241524`, the same 19 selected features, and both selected
regularization parameters, with coefficient and prediction differences below
9e-17. Optional comparison tests use `NATURAL_STORIES_REFERENCE_FIT` (the
original fitting script) and `NATURAL_STORIES_RESULTS_ROOT` (an existing run
directory); these four checks skip when those inputs are not supplied. The
remaining tests run without another checkout or model weights.
