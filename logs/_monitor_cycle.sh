#!/bin/bash
# One inspection cycle for the two GPU training runs. Appends to logs/monitor.log.
cd /c/Softwares/code/Nuka-Physics || exit 1
PY="C:/Users/Nidho/AppData/Local/Programs/Python/Python313/python.exe"
PS="C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
MON=logs/monitor.log
STATE=logs/_monitor_state
BUDGET=40000000

if [ ! -f "$STATE" ]; then
cat > "$STATE" <<EOF
hs_frames=0
bf_frames=0
hs_guard=0
bf_guard=0
hs_restarts=0
bf_restarts=0
hs_status=RUNNING
bf_status=RUNNING
start_ts=$(date +%s)
EOF
fi
source "$STATE"

now=$(date +%s)

# ---- inspect_task tag log cfg nn_dir prefix ----
# echoes: alive|epoch|frames|guard_total|rew
inspect_task() {
  local tag=$1 log=$2 cfg=$3
  local alive=0 line epoch frames guard rew
  grep -Fq -- "--config $cfg" /tmp/_procs.txt && alive=1
  line=$(grep -o "epoch: [0-9]* frames: [0-9]*/$BUDGET" "$log" 2>/dev/null | tail -1)
  epoch=$(echo "$line" | sed -n 's/epoch: \([0-9]*\) .*/\1/p')
  frames=$(echo "$line" | sed -n "s/.*frames: \([0-9]*\)\/$BUDGET/\1/p")
  guard=$(grep -c "\[nuka-guard\] skipped batch" "$log"); [ -z "$guard" ] && guard=0
  rew=$(grep -o "rewards: *\[[-0-9.e]*\]" "$log" 2>/dev/null | tail -1 | tr -d '[]' | awk '{print $NF}')
  echo "${alive}|${epoch:-NA}|${frames:-0}|${guard}|${rew:-NA}"
}

"$PS" -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine" > /tmp/_procs.txt 2>/dev/null

hs_res=$(inspect_task hs logs/train_front_handstand.log configs/rl_games/go2_front_handstand.yaml)
bf_res=$(inspect_task bf logs/train_backflip.log configs/rl_games/go2_backflip.yaml)

handle() {
  # $1=tag $2=res $3=prev_frames_var_val $4=prev_guard $5=restarts $6=status
  # sets globals: R_alive R_epoch R_frames R_guard R_rew R_note R_newstatus R_restart_cmd
  local tag=$1 res=$2 prev_f=$3 prev_g=$4 restarts=$5 status=$6
  local IFS='|'
  set -- $res
  local alive=$1 epoch=$2 frames=$3 guard=$4 rew=$5
  R_alive=$alive; R_epoch=$epoch; R_frames=$frames; R_guard=$guard; R_rew=$rew
  R_note=""; R_newstatus=$status; R_restart_cmd=""
  local gdelta=$((guard - prev_g))
  [ "$gdelta" -gt 200 ] && R_note="guard-skip-delta=$gdelta(>200)"
  case $status in
    COMPLETED|FAILED) return ;;
  esac
  if [ "$alive" = "1" ]; then
    if [ "$frames" -gt "$prev_f" ]; then
      R_newstatus=OK
    else
      local logf=logs/train_$([ "$tag" = hs ] && echo front_handstand || echo backflip).log
      R_newstatus=STALLED
      R_note="last-err: $(tail -c 400 "$logf" 2>/dev/null | tr '\r\n' ' ' | tail -c 200)"
    fi
  else
    local logf=logs/train_$([ "$tag" = hs ] && echo front_handstand || echo backflip).log
    if grep -q "Maximum reward achieved" "$logf" || grep -q "MAX FRAMES NUM" "$logf" || [ "$frames" -ge "$BUDGET" ]; then
      R_newstatus=COMPLETED
    else
      R_newstatus=CRASHED_PENDING_RESTART
    fi
  fi
}

# ---- handle handstand ----
handle hs "$hs_res" "$hs_frames" "$hs_guard" "$hs_restarts" "$hs_status"
hs_alive=$R_alive; hs_epoch=$R_epoch; hs_frames_now=$R_frames; hs_guard_now=$R_guard; hs_rew=$R_rew
if [ "$R_newstatus" = "CRASHED_PENDING_RESTART" ]; then
  tsx=$(date +%Y%m%d_%H%M%S)
  tail -200 logs/train_front_handstand.log > "logs/crash_handstand_${tsx}.log"
  if [ "$hs_restarts" -lt 3 ]; then
    ck=$(ls -t runs/go2_front_handstand/nn/last_go2_front_handstand_ep_*.pth 2>/dev/null | head -1)
    PYTHONPATH=python "$PY" tools/train/train_skill.py --config configs/rl_games/go2_front_handstand.yaml --max-frames $BUDGET --checkpoint "$ck" >> logs/train_front_handstand.log 2>&1 &
    hs_restarts=$((hs_restarts+1))
    hs_note_restart="RESTARTED#$hs_restarts ck=$(basename "$ck")"
    hs_status=RESTARTED
  else
    hs_status=FAILED
  fi
elif [ "$R_newstatus" = "OK" ] && [ "$hs_status" = "RESTARTED" ]; then
  hs_status=OK_RECOVERED
elif [ "$hs_status" != "FAILED" ] && [ "$hs_status" != "COMPLETED" ]; then
  hs_status=$R_newstatus
fi

# ---- handle backflip ----
handle bf "$bf_res" "$bf_frames" "$bf_guard" "$bf_restarts" "$bf_status"
bf_alive=$R_alive; bf_epoch=$R_epoch; bf_frames_now=$R_frames; bf_guard_now=$R_guard; bf_rew=$R_rew
if [ "$R_newstatus" = "CRASHED_PENDING_RESTART" ]; then
  tsx=$(date +%Y%m%d_%H%M%S)
  tail -200 logs/train_backflip.log > "logs/crash_backflip_${tsx}.log"
  if [ "$bf_restarts" -lt 3 ]; then
    ck=""
    for c in $(ls -t runs/go2_backflip/nn/last_go2_backflip_ep_*.pth 2>/dev/null); do
      r=$(echo "$c" | sed -n 's/.*_rew_\([0-9.e]*\)\.pth/\1/p')
      ok=$(awk -v a="$r" 'BEGIN{print (a<1000000 && a!="") ? 1 : 0}')
      [ "$ok" = "1" ] && ck="$c" && break
    done
    if [ -n "$ck" ]; then
      PYTHONPATH=python "$PY" tools/train/train_skill.py --config configs/rl_games/go2_backflip.yaml --max-frames $BUDGET --checkpoint "$ck" >> logs/train_backflip.log 2>&1 &
    else
      PYTHONPATH=python "$PY" tools/train/train_skill.py --config configs/rl_games/go2_backflip.yaml --max-frames $BUDGET >> logs/train_backflip.log 2>&1 &
    fi
    bf_restarts=$((bf_restarts+1))
    bf_note_restart="RESTARTED#$bf_restarts ck=$(basename "${ck:-none}")"
    bf_status=RESTARTED
  else
    bf_status=FAILED
  fi
elif [ "$R_newstatus" = "OK" ] && [ "$bf_status" = "RESTARTED" ]; then
  bf_status=OK_RECOVERED
elif [ "$bf_status" != "FAILED" ] && [ "$bf_status" != "COMPLETED" ]; then
  bf_status=$R_newstatus
fi

ts=$(date "+%Y-%m-%d %H:%M:%S")
echo "$ts HS[$hs_status] proc=$hs_alive ep=$hs_epoch frames=$hs_frames_now/$BUDGET rew=$hs_rew guard=$hs_guard_now(+$((hs_guard_now-hs_guard))) restarts=$hs_restarts $hs_note_restart" >> "$MON"
echo "$ts BF[$bf_status] proc=$bf_alive ep=$bf_epoch frames=$bf_frames_now/$BUDGET rew=$bf_rew guard=$bf_guard_now(+$((bf_guard_now-bf_guard))) restarts=$bf_restarts $bf_note_restart" >> "$MON"

cat > "$STATE" <<EOF
hs_frames=$hs_frames_now
bf_frames=$bf_frames_now
hs_guard=$hs_guard_now
bf_guard=$bf_guard_now
hs_restarts=$hs_restarts
bf_restarts=$bf_restarts
hs_status=$hs_status
bf_status=$bf_status
start_ts=$start_ts
EOF

elapsed=$((now - start_ts))
done_flag=0
case "$hs_status" in OK|OK_RECOVERED|STALLED|RESTARTED|RUNNING) ;; *) done_hs=1 ;; esac
case "$bf_status" in OK|OK_RECOVERED|STALLED|RESTARTED|RUNNING) ;; *) done_bf=1 ;; esac
if [ "${done_hs:-0}" = "1" ] && [ "${done_bf:-0}" = "1" ]; then done_flag=1; fi
if [ "$elapsed" -ge 14400 ]; then done_flag=1; fi

echo "CYCLE hs=$hs_status($hs_frames_now) bf=$bf_status($bf_frames_now) elapsed=${elapsed}s done=$done_flag"
