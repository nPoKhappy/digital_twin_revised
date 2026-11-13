# 63-model validation: expanded Y scale

These figures retain all 126 distinct regional coordinates. The y axis is piecewise: 0.35–1.35 is given 2.6 times the display height of 1.35–2.75. The ratio of display distance to numeric difference therefore changes at 1.35; the dotted reference line and figure heading disclose this. No numeric range is skipped and no points are jittered. All panels share the same x and y scales. No-Gain references are repeated across model families.

KCI is agreement with an ANN reference, not a direct measurement of the Aspen gain. AvgRelRMSE uses each seed’s matching No-Gain as denominator; R5-5 partly crosses the core boundary. Full constant-scale figures remain in the adjacent publication_figure folder.

Rebuild: `.\.venv\Scripts\python.exe -m scripts.plotting.plot_retrain_63_expanded_y --overwrite`.
