# LTPJ MNI to fsaverage transformation

The canonical transformation is maintained in
[`gretatuckute/neuromaps`](https://github.com/gretatuckute/neuromaps), file
`examples/perform_transforms.py`, commit
`dc81cfdafd7687de5019c7b8b4f8badb22e75bc6`. This repository keeps a pointer to
that implementation rather than a second transformation script.

## Source volume

Download `LTPJ_xyz.nii` from the
[Saxe Lab Theory of Mind group maps](https://saxelab.mit.edu/use-our-theory-mind-group-maps/).
The exact source used here has SHA-256
`10c398d744b6831cb9c0f4ac24529afe8b029d98cddbdd611f530e97e47cfd5b`.
It is a binary 2-mm MNI volume, not an unthresholded probability map:

| Property | Value |
| --- | --- |
| Shape | 79 × 95 × 68 |
| Orientation | LAS |
| Number of values | 510,340 |
| Nonzero voxels | 3,322 |
| Unique values | 0, 1 |

## Run the canonical example

Clone the repository into a separate directory and use the pinned commit:

```bash
git clone https://github.com/gretatuckute/neuromaps.git
cd neuromaps
git checkout dc81cfdafd7687de5019c7b8b4f8badb22e75bc6
python -m venv .venv
source .venv/bin/activate
python -m pip install 'setuptools<81' 'nilearn==0.9.2' 'nibabel==5.3.2' -e .
```

Use Python 3.10. In your local copy of `examples/perform_transforms.py`, set
`speech_parcel_NQ36_to_fsaverage = False` and `ltpj_to_fsaverage = True`, leaving
the other transform flags false. Set `path` in the LTPJ block to the directory
containing `LTPJ_xyz.nii`. The historical example also loads an unrelated
`mni_lang_parcel` before checking any flags: comment out that one load when
running only the LTPJ block, or supply its language-volume path. Then run:

```bash
python examples/perform_transforms.py
```

The primary operation is
`mni152_to_fsaverage(img=ltpj_path, method='nearest', fsavg_density='164k')`,
using neuromaps' registration-fusion mapping at fsaverage7 density. The example
prints the input shape, total values, nonzero count, and unique values. It
also produces a linear-interpolation sensitivity check. It writes four files
beside the source volume:

```text
lh.LTPJ_xyz_fsaverage_nearest_164k.gii
rh.LTPJ_xyz_fsaverage_nearest_164k.gii
lh.LTPJ_xyz_fsaverage_linear_164k.gii
rh.LTPJ_xyz_fsaverage_linear_164k.gii
```

## Use the surface mask in the Ghost notebook

The exact nearest-neighbor left-hemisphere output used by the figure is now
included as
`analysis/ghost_surface_assets/lh.Saxe_Dufour2013_LTPJ_fsaverage7_nearest.gii`.
No transformation is needed to execute the notebook. When independently
regenerating the parcel, use that filename in your own asset directory and
point `GHOST_SURFACE_ASSET_DIR` to it for comparison.
Each hemisphere has 163,842 vertices. The primary left mask has 6,587 nonzero
vertices; the right mask has none. An independent rerun of this transformation
matched all four saved surface arrays exactly.

The reference left-hemisphere file has SHA-256
`add401f8c4c4b05e58b94ac86f1654b9c7d3699800ff4eef0e2118d65d6cc263`.
GIFTI serialization can change file hashes without changing the arrays, so
compare loaded arrays when checking a newly generated mask. The Ghost
notebook checks vertex count and parcel size before indexing the mask.
