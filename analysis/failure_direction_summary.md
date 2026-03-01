# Failure Direction Analysis

## Overall Failure Direction

| Direction | Count | Percentage |
|-----------|-------|------------|
| Failed on TRUE  | 450 | 73.6% |
| Failed on FALSE | 161 | 26.4% |
| **Total** | 611 | 100% |

## Failure Direction by Domain

| Domain | Failed on TRUE | Failed on FALSE | Total | % TRUE |
|--------|----------------|-----------------|-------|--------|
| physical | 125 | 44 | 169 | 74.0% |
| social | 162 | 46 | 208 | 77.9% |
| temporal | 52 | 25 | 77 | 67.5% |
| time | 111 | 46 | 157 | 70.7% |

## Failure Direction by Domain+Scenario

| Domain_Scenario | Failed on TRUE | Failed on FALSE | Total | % TRUE |
|-----------------|----------------|-----------------|-------|--------|
| social_causal | 87 | 13 | 100 | 87.0% |
| physical_causal | 73 | 22 | 95 | 76.8% |
| time_causal | 57 | 23 | 80 | 71.2% |
| time_comparison | 54 | 23 | 77 | 70.1% |
| social_comparison | 54 | 22 | 76 | 71.1% |
| physical_comparison | 37 | 13 | 50 | 74.0% |
| temporal_causal | 29 | 11 | 40 | 72.5% |
| temporal_comparative | 22 | 14 | 36 | 61.1% |
| social_comparative | 21 | 11 | 32 | 65.6% |
| physical_comparative | 15 | 9 | 24 | 62.5% |
| temporal_physical | 1 | 0 | 1 | 100.0% |

## Selected Pairs for Phase 2

- **Total selected:** 200
- **Avg confidence on wrong answer:** 0.896
- **Avg logit gap:** 2.534

### By Domain:

- time: 103
- temporal: 50
- social: 47

### By Failure Direction:

- Failed on TRUE: 153
- Failed on FALSE: 47