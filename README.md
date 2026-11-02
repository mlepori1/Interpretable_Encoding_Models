# Augmented Sparse Encoding Models

Code and data for "Interpreting Brain Responses to Language with Sparse Features from Language Models".

The respository is structured as follows:

- `src/regression_analysis.py` contains code for the quantitative regression analyses in Study 1 and Study 3.
- `src/qualitative_analysis.py` contains code for the qualitative/feature-based analyses in Study 1, Study 2, and Study 3.
- `src/generalization_analysis.py` contains code for the generalization analyses in Study 2 and Study 3.


Examples of calling these scripts to generate results for subtypes or fROIs is found in `src/bash_scripts`

Code for generating figures in the preprint are found in the `analysis` directory.

The [QA-Emb-29 reproduction instructions](features/benara2024/README.md)
describe how to rebuild the existing Benara et al. comparison features from
saved answers or regenerate them using pinned model revisions. The generator
is `src/featurize_qa_emb.py`; its default output matches the existing
`--use_benara2024` regression input without changing that regression code.

The [Ghost extension reproduction instructions](analysis/ghost_extension_reproduction.md)
cover the additional-voxel search, participant overlap, anatomical quantification,
and surface figures, with a standalone notebook and rendered HTML. The notebook
uses tracked anonymized results and separately supplied surface assets.
