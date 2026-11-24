Q11 filtered trial dataset
==========================

Files
-----
Source dataset: E:\con2phys_submission\dataset\trials_info.csv
Filtered dataset: trials_info_filtered.csv

The source contained 10,373 trial records from 18 mice. The filtered dataset
contains 10,350 records. All original columns and trial identifiers were
preserved.

Filtering criteria
------------------
Each retained record met all three criteria:

1. Complete timing information: trial_start, stim_start, outcome, and
   trial_end were all present.
2. Strict event order:
   trial_start < stim_start < outcome < trial_end
3. The record was not the final recorded trial for its mouse. The final trial
   was defined as the record with the maximum trial_id within each mouse.

The first two criteria retained 10,361 complete, correctly ordered records.
Seven of the 18 final records had already failed those criteria, so excluding
the remaining 11 final records produced the final 10,350-record dataset.

No maximum-duration threshold, statistical outlier rule, variable A/B/C
filter, or trial-count balancing was applied. Long but correctly ordered
non-final trials were retained.

Validation checks
-----------------
- No missing values remain in the four timing columns.
- Every retained record satisfies the required event order.
- No retained record is the final trial of its mouse.
- No duplicate (mouse_id, trial_id) combinations are present.
- trial_duration equals trial_end - trial_start within floating-point
  precision.

All durations below are in seconds. SD denotes standard deviation, IQR is the
interval from the first to the third quartile, and values in square brackets
are Q1-Q3. The full-trial median is calculated directly and is not expected to
equal the sum of the three portion medians.

Overall trial portions
----------------------
| Portion                  | N      | Mean +/- SD  | Median [IQR]       | 5th-95th percentile | Range          |
|--------------------------|--------|--------------|--------------------|----------------------|----------------|
| Full trial               | 10,350 | 4.133 +/- 5.438 | 2.673 [2.072-3.921] | 1.855-10.224         | 1.596-68.565   |
| Trial start -> stimulus  | 10,350 | 1.137 +/- 1.158 | 0.684 [0.562-1.271] | 0.482-3.239          | 0.011-15.382   |
| Stimulus -> outcome      | 10,350 | 1.734 +/- 5.291 | 0.382 [0.277-0.705] | 0.203-7.360          | 0.057-60.692   |
| Outcome -> trial end     | 10,350 | 1.262 +/- 0.399 | 1.069 [1.058-1.112] | 1.046-2.077          | 0.046-3.414    |

Per-mouse durations
-------------------
Values are median [Q1-Q3] in seconds. "Original" is the number of records in
the source dataset; "Filtered" is the number retained in
trials_info_filtered.csv.

| Mouse | Original | Filtered | Full trial          | Start -> stimulus   | Stimulus -> outcome  | Outcome -> end       |
|-------|----------|----------|---------------------|---------------------|----------------------|----------------------|
| 1     | 408      | 407      | 2.581 [2.147-3.355] | 0.710 [0.591-1.063] | 0.407 [0.286-0.709]  | 1.110 [1.101-1.128]  |
| 2     | 850      | 849      | 2.330 [2.043-3.243] | 0.698 [0.561-1.256] | 0.378 [0.308-0.519]  | 1.066 [1.059-1.075]  |
| 3     | 478      | 473      | 2.741 [2.168-4.286] | 0.648 [0.547-0.894] | 0.634 [0.390-1.609]  | 1.064 [1.058-1.076]  |
| 4     | 505      | 504      | 3.311 [2.279-4.992] | 1.888 [0.847-3.587] | 0.281 [0.238-0.353]  | 1.058 [1.051-1.064]  |
| 5     | 424      | 423      | 3.479 [2.382-5.629] | 0.738 [0.598-1.455] | 0.603 [0.331-2.617]  | 1.111 [1.101-1.128]  |
| 6     | 471      | 470      | 2.055 [1.925-2.956] | 0.606 [0.530-0.705] | 0.299 [0.250-0.441]  | 1.066 [1.057-1.079]  |
| 7     | 476      | 475      | 3.098 [2.188-7.317] | 0.612 [0.523-0.745] | 0.841 [0.456-4.862]  | 1.059 [1.052-1.069]  |
| 8     | 663      | 662      | 2.246 [2.008-3.161] | 0.615 [0.533-0.722] | 0.392 [0.307-0.577]  | 1.062 [1.055-1.076]  |
| 9     | 580      | 579      | 2.405 [2.032-3.208] | 0.638 [0.535-0.811] | 0.401 [0.288-0.715]  | 1.061 [1.054-1.069]  |
| 10    | 663      | 662      | 2.535 [2.043-4.360] | 0.634 [0.533-1.153] | 0.464 [0.300-1.415]  | 1.057 [1.050-1.064]  |
| 11    | 641      | 640      | 2.299 [2.070-4.134] | 0.618 [0.541-0.721] | 0.502 [0.348-1.516]  | 1.086 [1.066-1.101]  |
| 12    | 437      | 436      | 4.396 [2.924-8.154] | 0.894 [0.604-1.628] | 1.224 [0.473-4.979]  | 1.063 [1.055-2.051]  |
| 13    | 571      | 570      | 2.689 [1.997-3.419] | 0.687 [0.550-1.332] | 0.312 [0.246-0.468]  | 1.063 [1.055-1.077]  |
| 14    | 505      | 504      | 2.868 [2.118-3.868] | 0.721 [0.565-1.749] | 0.352 [0.272-0.484]  | 1.079 [1.067-2.050]  |
| 15    | 458      | 457      | 2.967 [2.157-4.372] | 0.799 [0.587-1.498] | 0.295 [0.236-0.526]  | 1.067 [1.057-2.055]  |
| 16    | 626      | 625      | 2.664 [2.022-3.767] | 0.694 [0.559-1.469] | 0.372 [0.261-0.751]  | 1.072 [1.064-1.083]  |
| 17    | 994      | 992      | 2.762 [2.112-3.448] | 0.945 [0.644-1.620] | 0.279 [0.232-0.365]  | 1.114 [1.106-1.127]  |
| 18    | 623      | 622      | 2.656 [2.011-3.611] | 0.656 [0.548-1.153] | 0.364 [0.232-0.703]  | 1.067 [1.059-1.081]  |
