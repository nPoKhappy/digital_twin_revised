# Figure preparation and provenance

The colour PDFs are vector figures with embedded fonts and a 190 x 133 mm canvas. The PNG and grayscale PNG files are previews; PNG is not listed as a final format in the JTICE author-guide text supplied by the user. See the two separately supplied caption files for figure descriptions, definitions and error-bar details. The 190 mm width is an Elsevier general sizing example, not a confirmed JTICE-specific width. Confirm the current journal instructions and production dimensions before submission.

These test probes were used in prior analyses and are not a fresh blind holdout. KCI is ANN-reference directional agreement and does not alone prove Aspen physical gain direction. Some historical validation files were deleted; consult source READMEs and provenance for limitations.

Rebuild from the repository root with `python -m scripts.analysis.plot_jtice_teacher_gain_kci`. The script checks per-seed counts and reconstructs the reported means and sample SD; it does not retrain or re-evaluate models.
