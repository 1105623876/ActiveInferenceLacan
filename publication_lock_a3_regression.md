# A3 Stage 2 regression

1. A col norm: 2.220e-16 → PASS
2. isolated env ids unique 9: True
3. q/C/d simplex: True
4. lam0==static: True
5. lam.20==leaky: True
6. lam1==memoryless: True
7. sigma=1 + lam=1 vs hard S_post: True

## Legacy vs A2 (seeds 0-19, T=50)

| env | sched | cond | n_match_Spre | n | note |
|-----|-------|------|--------------|---|------|
| 5_2_6 | sync | hard | 20 | 20 | must_exact |
| 5_2_6 | sync | smooth | 20 | 20 | must_exact |
| 5_2_6 | sync | static | 20 | 20 | must_exact |
| 5_2_6 | sync | leaky | 20 | 20 | must_exact |
| 5_2_6 | sync | memoryless | 20 | 20 | must_exact |
| 5_2_6 | seq_ABC | hard | 20 | 20 | must_exact |
| 5_2_6 | seq_ABC | smooth | 20 | 20 | must_exact |
| 5_2_6 | seq_ABC | static | 0 | 20 | seq_infer_may_differ |
| 5_2_6 | seq_ABC | leaky | 0 | 20 | seq_infer_may_differ |
| 5_2_6 | seq_ABC | memoryless | 0 | 20 | seq_infer_may_differ |
| 8_8_8 | sync | hard | 20 | 20 | must_exact |
| 8_8_8 | sync | smooth | 20 | 20 | must_exact |
| 8_8_8 | sync | static | 20 | 20 | must_exact |
| 8_8_8 | sync | leaky | 20 | 20 | must_exact |
| 8_8_8 | sync | memoryless | 20 | 20 | must_exact |
| 8_8_8 | seq_ABC | hard | 20 | 20 | must_exact |
| 8_8_8 | seq_ABC | smooth | 20 | 20 | must_exact |
| 8_8_8 | seq_ABC | static | 0 | 20 | seq_infer_may_differ |
| 8_8_8 | seq_ABC | leaky | 0 | 20 | seq_infer_may_differ |
| 8_8_8 | seq_ABC | memoryless | 0 | 20 | seq_infer_may_differ |

8. required exact cells: PASS
9. regression md sha repeat: True (143e4259ed77d836)
