#!/usr/bin/env bash
# GATE A -- can Gazebo Fortress produce real lidar data under WSL2?
#
# Fortress lidar is `gpu_lidar`: it RENDERS the scene through ogre2 to
# produce ranges. gz-sim 6 has documented OGRE2 initialisation failures under
# WSL2 (gazebosim/gz-sim#2502, #1116). If that hits, /lidar goes silent and
# SLAM has no input -- so this is a project-defining gate, not a cosmetic one.
#
# Evidence printed here belongs in the report appendix whichever way it goes.
#
#     bash setup/gate_a_check.sh [--gui]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORLD="$HERE/gate_a_lidar.sdf"
RENDER_ENGINE="${RENDER_ENGINE:-ogre2}"

# Applies the rendering configuration established by this gate. See that
# file for why software rendering is the correct choice on this machine.
# shellcheck source=/dev/null
[ -f "$HERE/gazebo_env.sh" ] && source "$HERE/gazebo_env.sh"

pass() { printf '\033[1;32m  PASS\033[0m  %s\n' "$*"; }
fail() { printf '\033[1;31m  FAIL\033[0m  %s\n' "$*"; }
info() { printf '\033[1;36m==> %s\033[0m\n' "$*"; }

info "Environment"
printf '  %-16s %s\n' "ubuntu"   "$(lsb_release -ds 2>/dev/null)"
printf '  %-16s %s\n' "kernel"   "$(uname -r)"
for p in ignition-fortress ros-humble-desktop ros-humble-ros-gz; do
    printf '  %-16s %s\n' "${p#ros-humble-}" \
        "$(dpkg-query -W -f'${Version}' "$p" 2>/dev/null || echo MISSING)"
done

info "OpenGL"
GL_REND=$(glxinfo -B 2>/dev/null | sed -n 's/^OpenGL renderer string: //p')
GL_VER=$(glxinfo -B 2>/dev/null | sed -n 's/^OpenGL core profile version string: //p')
printf '  %-16s %s\n' "renderer" "${GL_REND:-<none>}"
printf '  %-16s %s\n' "version"  "${GL_VER:-<none>}"
if [[ "$GL_REND" == *llvmpipe* && "${GPU:-0}" == "1" ]]; then
    fail "asked for hardware (GPU=1) but got llvmpipe"
elif [[ "$GL_REND" == *llvmpipe* ]]; then
    # Deliberate on this machine: hardware GL crashes OGRE2 (see
    # setup/gazebo_env.sh). Measured RTF is ~1.0, so this is not a problem.
    pass "software rendering, by choice -- see the RTF measured below"
elif [[ -n "$GL_REND" ]]; then
    pass "hardware accelerated"
fi

info "Starting gz-sim server (headless rendering, engine=$RENDER_ENGINE)"
LOG=$(mktemp)
ign gazebo -s -r -v4 --headless-rendering --render-engine "$RENDER_ENGINE" \
    "$WORLD" >"$LOG" 2>&1 &
SIM_PID=$!
trap 'kill $SIM_PID 2>/dev/null; wait $SIM_PID 2>/dev/null' EXIT

for _ in $(seq 1 30); do
    sleep 1
    ign topic -l 2>/dev/null | grep -q '^/lidar$' && break
done

if ! kill -0 $SIM_PID 2>/dev/null; then
    fail "server exited early -- likely an OGRE2 init crash"
    echo "--- last 25 log lines ---"; tail -25 "$LOG"
    exit 1
fi
if ! ign topic -l 2>/dev/null | grep -q '^/lidar$'; then
    fail "/lidar topic never appeared after 30 s"
    echo "--- last 25 log lines ---"; tail -25 "$LOG"
    exit 1
fi
pass "server running, /lidar advertised"

info "Sampling /lidar"
SAMPLE=$(timeout 20 ign topic -e -t /lidar -n 2 2>/dev/null)
if [[ -z "$SAMPLE" ]]; then
    fail "no messages received on /lidar"
    echo "--- last 25 log lines ---"; tail -25 "$LOG"
    exit 1
fi

# Expected by construction: wall at +5 m ahead, wall at 8 m to the right.
# The sample goes via a file, not stdin: a heredoc and a here-string both bind
# stdin, and the here-string would win -- feeding message text to the parser.
SAMPLE_FILE=$(mktemp)
printf '%s' "$SAMPLE" > "$SAMPLE_FILE"

python3 - "$SAMPLE_FILE" <<'PY'
import re, sys
txt = open(sys.argv[1]).read()
vals = [float(v) for v in re.findall(r'^\s*ranges:\s*(-?[\d.einf]+)', txt, re.M)]
if not vals:
    # Fall back to the flow-style encoding some builds emit.
    vals = [float(v) for v in re.findall(r'ranges:\s*\[([^\]]*)\]', txt)[0].split(',')] \
        if re.search(r'ranges:\s*\[', txt) else []

finite = [v for v in vals if v == v and v not in (float('inf'), float('-inf'))]
print(f"  beams received : {len(vals)}")
print(f"  finite ranges  : {len(finite)}")
if finite:
    print(f"  range span     : {min(finite):.2f} .. {max(finite):.2f} m")

ok = len(finite) >= 20 and 4.0 < min(finite) < 6.0
print()
if ok:
    print("\033[1;32m  PASS\033[0m  gpu_lidar returns real geometry "
          "(nearest ~5 m == the wall we placed)")
else:
    print("\033[1;31m  FAIL\033[0m  ranges are missing or do not match the known world")
sys.exit(0 if ok else 1)
PY
RC=$?

info "Real-time factor"
RTF=$(timeout 15 ign topic -e -t /stats -n 5 2>/dev/null \
      | grep -A3 real_time_factor | grep -oE '[0-9]+\.[0-9]+' | tail -3 | tr '\n' ' ')
echo "  ${RTF:-<not reported>}"

echo
if [[ $RC -eq 0 ]]; then
    printf '\033[1;32m==> GATE A PASSED -- proceed with Gazebo Fortress\033[0m\n'
else
    printf '\033[1;31m==> GATE A FAILED\033[0m\n'
    echo "Fallback ladder:"
    echo "  F1  MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA, recheck glxinfo"
    echo "  F2  RENDER_ENGINE=ogre bash setup/gate_a_check.sh   (OGRE 1.x)"
    echo "  F3  LIBGL_ALWAYS_SOFTWARE=1, fewer beams, accept low RTF"
    echo "  F4  Gazebo Classic 11 (CPU 'ray' sensor, no rendering needed)"
fi
exit $RC
