# Simple script to concatenate the participant names and neuroids from the deduped set into one file to feed to generaliation script
import pandas as pd

df1 = pd.read_csv("../../results/fROI/full_features/regressions/sae/gemma-2-2b-res-matryoshka-dc/mean/12/standardized_betas/regressions/cvn7002_cvn7007_cvn7011_cvn7012/results_langfroi12345_dedup.csv")
df2 = pd.read_csv("../../results/fROI/full_features/regressions/sae/gemma-2-2b-res-matryoshka-dc/mean/12/standardized_betas/regressions/cvn7006_cvn7009_cvn7013_cvn7016/results_langfroi12345_dedup.csv")

df1 = df1[["neuroid", "Participant"]]
df2 = df2[["neuroid", "Participant"]]

pd.concat([df1, df2], axis=0).to_csv("filter_set.csv", index=False)
