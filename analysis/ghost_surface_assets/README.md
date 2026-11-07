# Ghost figure surface parcels

This directory includes the three parcel files used for the main Ghost
surface figure. Each contains 163,842 values for the left fsaverage7
hemisphere. They are group reference parcels, not individual participants'
brain responses. Exact file hashes are recorded in [manifest.json](manifest.json).

| File | Role |
| --- | --- |
| `lh.Saxe_Dufour2013_LTPJ_fsaverage7_nearest.gii` | Functional LTPJ outline and anatomical quantification |
| `lh.parcels_language_fsaverage_frontal_SN8.gii` | Language outlines: IFGorb, IFG, MFG (labels 1–3) |
| `lh.parcels_language_fsaverage_REGULARIZED.gii` | Language outlines: AntTemp and PostTemp (labels 4–5) |

## LTPJ source and transformation

The LTPJ mask is derived from `LTPJ_xyz.nii`, distributed on the
[Saxe Lab group maps page](https://saxelab.mit.edu/use-our-theory-mind-group-maps/).
The source page requests credit to Dufour et al. (2013). This is a binary
thresholded group mask, not an unthresholded probability map. The included
surface derivative uses neuromaps' nearest-neighbor MNI-to-fsaverage7
transformation and contains 6,587 nonzero vertices.

The [transformation record](../saxe_ltpj_surface_transform/README.md) provides
the source checksum, pinned implementation, environment, and reproduction
instructions. Re-running the transformation is optional: the exact surface
mask used by the figure is included here.

## Language parcel provenance

These are the study's surface reference maps, not unchanged files downloaded
from the [EvLab parcel collection](https://www.evlab.mit.edu/resources-all/download-parcels).
The frontal file contains the three left frontal parcels drawn by Kendrick
Kay and Greta Tuckute on June 15, 2024, as documented by the study's original
data-packaging code. Despite its historical `SN8` filename, it should not be
described as an untouched public SN8 release.

The regularized file derives from the EvLab SN220 language parcels, with
subsequent surface cleanup. Only its temporal labels 4 and 5 are used in
these figures. The notebook checks the combined five-parcel definition
against the released language-fROI metadata. Retain attribution to the study
and the upstream parcel sources when reusing these derivatives. No new
license for upstream resources is conferred by this repository.

## Other surface files

The notebook downloads `lh.inflated` and `lh.sulc` from the NSD fsaverage
release when they are absent. An internet connection is therefore needed
on first execution unless those two files are supplied locally.

Glasser is not used in the main figure. Its optional supplementary surface
map requires `lh.HCP_MMP1_fsaverage.mgz` and `HCP_MMP1_fsaverage.txt`; these
are not bundled. Without them, the notebook skips the Glasser surface
supplement and continues with the LTPJ figures. Metadata-derived anatomical
tables remain available without the Glasser surface files.
