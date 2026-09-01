#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="$repo_root/analysis/run_anchor8_pistachio_hard_2026_09_02_003500.sh"
out_base="$repo_root/analysis_outputs/anchor8_overnight_2026_09_02_003500"
mkdir -p "$out_base"
chain_log="$out_base/chain.log"
stage_status="$out_base/stage_status.tsv"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >>"$chain_log"
}

check_artifacts() {
  local dir="$1"
  [[ -s "$dir/plan.pkl" ]] \
    && [[ -s "$dir/fragment_yield.jsonl" ]] \
    && [[ -s "$dir/retro_candidates.jsonl" ]] \
    && grep -q '"grain": "retro_candidate"' "$dir/retro_candidates.jsonl" \
    && grep -q '"grain": "reaction"' "$dir/retro_candidates.jsonl"
}

record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$(date '+%Y-%m-%d %H:%M:%S')" >>"$stage_status"
}

log "chain start"
printf 'stage\tsampler\titerations\troute_limit\tstatus\tfinished_at\n' >"$stage_status"
anchor_ok=0

log "S0 smoke anchor8 2x3 start"
if bash "$runner" "$out_base/s0_smoke_anchor8_i3_r2" anchor8 3 2 >>"$chain_log" 2>&1 \
  && check_artifacts "$out_base/s0_smoke_anchor8_i3_r2"; then
  log "S0 PASS"
  record s0_smoke anchor8 3 2 pass
  anchor_ok=1
else
  log "S0 FAIL"
  record s0_smoke anchor8 3 2 fail
fi

if [[ "$anchor_ok" -eq 1 ]]; then
  log "S1 gate anchor8 20x50 start"
  if bash "$runner" "$out_base/s1_gate_anchor8_i50_r20" anchor8 50 20 >>"$chain_log" 2>&1 \
    && check_artifacts "$out_base/s1_gate_anchor8_i50_r20"; then
    log "S1 PASS"
    record s1_gate anchor8 50 20 pass
  else
    log "S1 FAIL"
    record s1_gate anchor8 50 20 fail
    anchor_ok=0
  fi
fi

if [[ "$anchor_ok" -eq 1 ]]; then
  log "S2 main anchor8 100x1000 start"
  if bash "$runner" "$out_base/s2_main_anchor8_i1000" anchor8 1000 0 >>"$chain_log" 2>&1 \
    && check_artifacts "$out_base/s2_main_anchor8_i1000"; then
    log "S2 PASS"
    record s2_main anchor8 1000 0 pass
  else
    log "S2 FAIL"
    record s2_main anchor8 1000 0 fail
  fi
else
  log "S2 skipped (anchor8 gate failed)"
  record s2_main anchor8 1000 0 skipped
fi

log "S3 control random 100x1000 start"
if bash "$runner" "$out_base/s3_control_random_i1000" random 1000 0 >>"$chain_log" 2>&1 \
  && check_artifacts "$out_base/s3_control_random_i1000"; then
  log "S3 PASS"
  record s3_control random 1000 0 pass
else
  log "S3 FAIL"
  record s3_control random 1000 0 fail
fi

log "chain end"
