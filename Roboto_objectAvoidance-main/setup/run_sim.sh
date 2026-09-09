#!/usr/bin/env bash
# Launch the generated site world with the robot in it.
#
#     bash setup/run_sim.sh [--gui] [extra ign args...]
#
# Resolves the two model paths (generated site geometry, and the robot) and
# applies the rendering configuration established by Gate A.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
SITE_NAME="${SITE_NAME:-pittsburgh_greenfield}"

WORLD="$ROOT/data/site_$SITE_NAME/world/site.sdf"
[ -f "$WORLD" ] || {
    echo "missing $WORLD"
    echo "run: python -m tools.gis_pipeline.build_world"
    exit 1
}

# shellcheck source=/dev/null
source "$HERE/gazebo_env.sh"

export IGN_GAZEBO_RESOURCE_PATH="$ROOT/data/site_$SITE_NAME/world/models:$ROOT/ros2_ws/src/roboto_ros/models${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"

MODE="-s -r --headless-rendering"
if [ "${1:-}" = "--gui" ]; then
    MODE="-r"
    shift
fi

echo "world : $WORLD"
echo "models: $IGN_GAZEBO_RESOURCE_PATH"
# shellcheck disable=SC2086
exec ign gazebo $MODE --render-engine ogre2 "$@" "$WORLD"
