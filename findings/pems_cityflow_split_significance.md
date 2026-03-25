# Findings: PEMS-Derived Demand Split Significance (CityFlow RL)

## Experiment Summary

This report records the final trained-vs-untrained comparison using PEMS-derived demand splits in CityFlow.

- Base config: `configs/cityflow.more_cycles.yaml`
- Comparison command: `traffic_rl.cli.compare_splits`
- Flow inputs:
  - `outputs/pems04/flow_train.json`
  - `outputs/pems04/flow_val.json`
  - `outputs/pems04/flow_test.json`
- Checkpoint: `outputs/agent_checkpoint_cityflow.npz`
- Protocol:
  - Episodes per seed: `10`
  - Number of seeds: `5`
  - Samples per policy per split: `50`
- Aggregate report source: `outputs/compare_splits_big/aggregate_report.json`

## Split-Wise Results

| Split | Trained Mean Reward | Untrained Mean Reward | Mean Diff (Trained - Untrained) | 95% CI | p-value | Cohen's d | Significant (alpha=0.05) |
|---|---:|---:|---:|---|---:|---:|---|
| Train | -2507.0 | -6847.6 | +4340.6 | [3815.96, 4869.10] | 0.00019996 | 3.1571 | ✅ Yes |
| Val | -3129.0 | -7983.8 | +4854.8 | [4536.1955, 5136.16] | 0.00019996 | 6.2182 | ✅ Yes |
| Test | -2398.0 | -8079.2 | +5681.2 | [5460.98, 5888.8985] | 0.00019996 | 10.1607 | ✅ Yes |

## Interpretation

- The trained policy consistently outperforms the untrained policy on all splits.
- Improvements are statistically significant for train, validation, and test splits.
- Positive mean differences across train/val/test indicate generalization beyond the training split.

## Methodological Notes

- Reward/observation evaluation is localized to incoming lanes of the controlled intersection to better reflect policy impact.
- PEMS route mapping was expanded to multiple valid CityFlow routes to avoid trivial single-phase behavior.
- Results remain simulation-based and scenario-specific; they support algorithmic effectiveness within this setup.

## One-Line Conclusion

The RL policy demonstrates strong and statistically significant improvements over the untrained baseline across PEMS-derived train/val/test demand splits in the CityFlow evaluation environment.
