#!/usr/bin/env bash
# Submit the strict-confirmation matrix: 2 modes x 2 datasets x 2 seeds = 8 L40 jobs.
# Usage: bash submit_strict_confirm_<ts>.sh   (run on the BJMU login node)
set -euo pipefail

ROOT=/lustre1/liuzm/liuzm_chenqx
WT="$ROOT/RetroPRO_strict_confirm"
RUNNER_TS="2026_09_03_102748"
RUNNER="$WT/analysis/run_strict_confirm_arm_${RUNNER_TS}.sh"
OUT="$WT/analysis_outputs/strict_confirm_${RUNNER_TS}"
mkdir -p "$OUT/jobs"

submitted="$OUT/submitted_jobs.tsv"
printf 'job_id\tname\tmode\ttest_routes\tseed\tresult_dir\n' >"$submitted"

for mode in nonstrict strict; do
  for routes in pth_hard uspto190; do
    for seed in 42 20260903; do
      name="sc_${mode}_${routes}_s${seed}"
      jobfile="$OUT/jobs/${name}.sbatch"
      cat > "$jobfile" <<SB
#!/bin/bash
#SBATCH -J ${name}
#SBATCH -p gpu_l40
#SBATCH -N 1
#SBATCH -o ${OUT}/jobs/${name}_%j.out
#SBATCH -e ${OUT}/jobs/${name}_%j.err
#SBATCH --no-requeue
#SBATCH -A liuzm_g1
#SBATCH --qos=liuzml40
#SBATCH --gres=gpu:1
#SBATCH --overcommit
#SBATCH --mincpus=12
pkurun bash ${RUNNER} ${mode} ${routes} ${seed} ${OUT}/${name}
SB
      job_id="$(sbatch --parsable "$jobfile")"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$job_id" "$name" "$mode" "$routes" "$seed" "$OUT/$name" >>"$submitted"
      echo "submitted $name as $job_id"
    done
  done
done
echo "all 8 jobs submitted; manifest: $submitted"
