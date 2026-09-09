# Gazebo environment for this machine. Source before any `ign gazebo` run:
#     source setup/gazebo_env.sh
#
# ---------------------------------------------------------------------------
# WHY SOFTWARE RENDERING, ON A MACHINE WITH AN RTX 4070
# ---------------------------------------------------------------------------
# This looks wrong and is not. Measured on this system (Gate A, 2026-08-24):
#
#   ogre2 + D3D12 (hardware)  -> SIGABRT during scene init.
#         Ogre2Material::SetTextureMapImpl -> TextureGpuManager::_waitFor
#         -> GenerateHwMipmaps::_executeSerial -> RenderSystem_GL3Plus
#         -> __cxa_throw -> std::terminate -> abort
#         i.e. WSLg's D3D12 GL mapping layer cannot service OGRE2's
#         hardware mipmap generation. Matches gz-sim#2502 / #1116.
#
#   ogre1 (hardware)          -> server runs, /lidar publishes, but EVERY
#         beam returns exactly range_min. Verified not a self-hit: the
#         result is identical with the sensor raised 5 m clear of all
#         geometry. The OGRE 1.x depth path in ign-rendering6 simply does
#         not produce valid depth here.
#
#   ogre2 + llvmpipe (software) -> CORRECT ranges (4.87-4.96 m against a
#         wall whose near face is at 4.90 m) at real-time factor ~1.00.
#
# The bug is in the GPU mapping layer, not in Gazebo and not in OGRE2.
# llvmpipe implements glGenerateMipmap correctly, so the crash disappears.
#
# The cost is far lower than it sounds: gpu_lidar renders a 180x1 depth
# strip, which is a trivial workload. Sim time is driven by /clock, so a
# lower real-time factor lengthens wall-clock runs without altering results
# -- and from Day 9 all tuning runs against recorded bags anyway.
#
# RE-TEST THIS after any WSL, Mesa, or NVIDIA driver update:
#     bash setup/gate_a_check.sh          # software (current default)
#     RENDER_ENGINE=ogre2 GPU=1 bash setup/gate_a_check.sh
# If hardware ever starts working, drop LIBGL_ALWAYS_SOFTWARE for the speed.
# ---------------------------------------------------------------------------

if [ "${GPU:-0}" != "1" ]; then
    export LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe
fi

# Fortress renders even in headless mode (gpu_lidar IS a render), so this
# only skips the window -- it does not skip the render engine.
export IGN_HEADLESS_RENDERING="${IGN_HEADLESS_RENDERING:-1}"

# Keep sim deterministic and quiet by default.
export IGN_VERBOSE="${IGN_VERBOSE:-1}"
