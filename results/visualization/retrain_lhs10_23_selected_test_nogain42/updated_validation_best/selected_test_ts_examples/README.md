# Exploratory test time-series examples

Cases are selected by test performance, so these examples are illustrative, not a representative performance estimate. The ranking and all case/seed RMSE values are provided in the accompanying CSV files.

Selected: interpolation ['R5-2', 'R5-1', 'R5-6']; extrapolation ['R5-23', 'R5-20']. These are the five cases illustrated in the previous requested figure, including R5-6. Current ranks use the average of H2S and SO2 percentage improvements in mean individual-seed TS RMSE relative to the mean No-Gain seed RMSE. The largest current internal deterioration is R5-6 with gain25_ss25 (mean improvement -31.6%). Three seeds: 42, 43, 44.

The y-axis is mole fraction in units of 10^-3 kmol/kmol. Curves show the arithmetic mean of seed-wise predictions, without error bars. The RMSE of a mean prediction curve is not the same as the mean of individual seed RMSEs. TS predictions use 18 Aspen reference history steps followed by recursive model-output feedback, with the recorded decoder inputs supplied at each step; these are not H1 or H18 results. The test cases were previously examined. See provenance.json for source hashes.

Recreate from repository root: `python -m scripts.plotting.plot_selected_test_improvement_cases`
