# CSS sampler experiment — analysis scripts

Formal experiment for the CSS hierarchical substructure samplers
(`TP_FREE_CSS_SAMPLER` = random | paircov | fullcov | bondcov | triplecov).

- `test_css_samplers.py` — smoke battery (5 modes; no model load).
- `css_offline_eval.py` — offline truth bond-breaking evaluation, 190-target
  track: does an arm produce a fragment that fully contains the true
  first-step reaction centre? Per-arm hit rate + Wilson 95% CI, stratified
  by rxnmapper atom-map confidence and reaction-centre size; strict
  changed-atom centre (primary) and changed-atom + 1-bond shell (secondary).

Run with the worktree package shadowing the editable install:
  PYTHONPATH=retro_star/packages/mlp_retrosyn <retropro python> \\
    analysis/css_offline_eval.py --pkl retro_star/dataset/routes_possible_test_hard.pkl \\
    --out analysis/css_offline_eval_targets.json
