#!/usr/bin/env bash
# Wait for the live demo to reach its goal, then report.
#
#     bash setup/wait_for_goal.sh [logfile] [max_minutes]
LOG="${1:-/tmp/demo.log}"
MAX_MIN="${2:-10}"

for _ in $(seq 1 $((MAX_MIN * 4))); do
    if grep -aq "GOAL REACHED" "$LOG"; then break; fi
    if grep -aq "Traceback" "$LOG"; then echo "node crashed"; break; fi
    sleep 15
done

echo "=== mission log ==="
grep -aE "SLAM started|plan #|GOAL REACHED|blocked|urgent|WARN|ERROR" "$LOG" | tail -25
echo
echo "=== final robot pose (Gazebo) ==="
timeout 10 ign model -m roboto_bot -p 2>/dev/null | tail -4
