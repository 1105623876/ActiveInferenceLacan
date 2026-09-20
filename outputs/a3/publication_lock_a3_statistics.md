# A3 publication-lock statistics

n_boot=10000. Seeds 1000..1199.

## Primary: Delta_memory = occ(λ=0)−occ(λ=1), schedule-averaged

- mean = 0.659900
- bootstrap 95% CI = [0.627874, 0.690551]
- sign-flip p = 0.000100
- sync only: 0.691900 [0.659549,0.721751] p=0.000100
- seq_ABC only: 0.627900 [0.587399,0.668151] p=0.000100

## Secondary (raw then Holm)

- hard occupancy seq-sync: p_raw=0.000100 p_Holm=0.000300
- lambda trend: p_raw=0.000100 p_Holm=0.000300
- observer x schedule: p_raw=0.000100 p_Holm=0.000300
- final20 hard sync vs seq: p_raw=0.000000 p_Holm=0.000000
- final20 lam0 vs lam1 sync: p_raw=0.000000 p_Holm=0.000000
- final20 lam0 vs lam1 seq: p_raw=0.000000 p_Holm=0.000000

hard seq−sync occupancy mean=0.763900 CI=[0.700000,0.823900] p=0.000100
interaction var=0.058288 p=0.000100; without hard p=0.000100
trend mean Spearman(−λ, occ)=0.900931 p=0.000100

## lambda × schedule (primary env)

| schedule | observer | occupancy | final20 | misID | NLL | Brier |
|----------|----------|-----------|---------|-------|-----|-------|
| sync | hard | 0.2011 | 0.205 | 0.000 | 0.000 | 0.000 |
| sync | smooth | 0.3924 | 0.005 | 0.000 | 0.806 | 0.345 |
| sync | filter_l0.00 | 0.9083 | 0.895 | 0.101 | 0.730 | 0.177 |
| sync | filter_l0.02 | 0.8669 | 0.100 | 0.089 | 0.338 | 0.155 |
| sync | filter_l0.05 | 0.8028 | 0.035 | 0.149 | 0.509 | 0.246 |
| sync | filter_l0.10 | 0.7087 | 0.015 | 0.244 | 0.730 | 0.362 |
| sync | filter_l0.20 | 0.5872 | 0.005 | 0.352 | 0.979 | 0.480 |
| sync | filter_l0.50 | 0.3873 | 0.000 | 0.504 | 1.339 | 0.623 |
| sync | filter_l1.00 | 0.2165 | 0.000 | 0.642 | 1.701 | 0.737 |
| seq_ABC | hard | 0.9650 | 0.965 | 0.000 | 0.000 | 0.000 |
| seq_ABC | smooth | 0.4045 | 0.000 | 0.000 | 0.806 | 0.345 |
| seq_ABC | filter_l0.00 | 0.8504 | 0.830 | 0.140 | 1.113 | 0.246 |
| seq_ABC | filter_l0.02 | 0.8667 | 0.095 | 0.091 | 0.340 | 0.156 |
| seq_ABC | filter_l0.05 | 0.8037 | 0.030 | 0.153 | 0.514 | 0.249 |
| seq_ABC | filter_l0.10 | 0.7170 | 0.015 | 0.239 | 0.721 | 0.358 |
| seq_ABC | filter_l0.20 | 0.5958 | 0.005 | 0.352 | 0.979 | 0.480 |
| seq_ABC | filter_l0.50 | 0.3969 | 0.000 | 0.506 | 1.341 | 0.624 |
| seq_ABC | filter_l1.00 | 0.2225 | 0.000 | 0.640 | 1.702 | 0.737 |

## robustness env

| schedule | observer | occupancy | final20 | misID | NLL | Brier |
|----------|----------|-----------|---------|-------|-----|-------|
| sync | hard | 1.0000 | 1.000 | 0.000 | 0.000 | 0.000 |
| sync | filter_l0.00 | 0.9990 | 1.000 | 0.009 | 0.043 | 0.016 |
| sync | filter_l0.20 | 0.6008 | 0.000 | 0.341 | 0.955 | 0.470 |
| sync | filter_l1.00 | 0.2248 | 0.000 | 0.636 | 1.691 | 0.734 |
| seq_ABC | hard | 1.0000 | 1.000 | 0.000 | 0.000 | 0.000 |
| seq_ABC | filter_l0.00 | 0.9994 | 1.000 | 0.004 | 0.014 | 0.007 |
| seq_ABC | filter_l0.20 | 0.6102 | 0.005 | 0.340 | 0.954 | 0.469 |
| seq_ABC | filter_l1.00 | 0.2455 | 0.000 | 0.633 | 1.685 | 0.731 |
