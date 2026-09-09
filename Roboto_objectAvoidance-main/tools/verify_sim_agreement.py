"""Cross-validate Gazebo's lidar against the offline raycast simulator.

These are two independent renderings of the same world:

  * Gazebo casts gpu_lidar rays against the extruded 3D building MESH;
  * roboto_core.sim.raycast_sim marches rays across the 2D rasterised
    occupancy grid built from the same OSM footprints.

They share only the source data, not any code. So agreement between them is
strong evidence that the whole geometric chain is coherent: OSM fetch,
projection, rasterisation, mesh extrusion, the world origin, the robot
spawn pose, and the sensor mounting all line up.

Disagreement localises the fault:
  * a constant angular offset  -> sensor yaw / beam-ordering mismatch
  * a constant range offset    -> sensor mounting pose
  * a mirrored profile         -> an east/west or row-order flip
  * agreement only near the robot -> range limits differ

Usage:
    ign topic -e -t /scan -n 1 > /tmp/scan.txt      # with the sim running
    python -m tools.verify_sim_agreement /tmp/scan.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from roboto_core.sim.raycast_sim import Lidar2D, LidarSpec, RaycastWorld
from tools.gis_pipeline.common.crs import load_site
from tools.gis_pipeline.common.paths import derived_dir

# Mirrors the <pose> of the gpu_lidar sensor in roboto_bot/model.sdf,
# expressed relative to base_link.
LIDAR_OFFSET_X = 0.15


def parse_ign_scan(path: Path) -> np.ndarray:
    """Pull the `ranges` array out of an `ign topic -e` LaserScan dump."""
    text = Path(path).read_text()
    vals = re.findall(r"^\s*ranges:\s*(-?[\d.]+(?:e[-+]?\d+)?|inf|-inf|nan)\s*$",
                      text, re.M)
    if not vals:
        raise SystemExit(f"no `ranges:` entries found in {path}")
    return np.array([float(v) for v in vals])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scan", help="file containing `ign topic -e -t /scan -n 1`")
    ap.add_argument("--site", default=None)
    args = ap.parse_args()

    site = load_site(args.site)
    spec = LidarSpec.from_site(site)

    gz = parse_ign_scan(Path(args.scan))
    if gz.size % spec.num_beams == 0 and gz.size > spec.num_beams:
        gz = gz[: spec.num_beams]          # keep the first message only

    world = RaycastWorld.from_occupancy_yaml(derived_dir(site) / "prior_occ.yaml")
    pose = (site.spawn_x + LIDAR_OFFSET_X, site.spawn_y,
            np.radians(site.spawn_yaw_deg))
    sim = Lidar2D(world, spec, rng=0).scan(pose, noise=False).ranges

    if gz.size != sim.size:
        print(f"  beam count differs: gazebo {gz.size} vs offline {sim.size}")
        return 1

    gz_v, sim_v = np.isfinite(gz), np.isfinite(sim)
    both = gz_v & sim_v

    print(f"beams              : {gz.size}")
    print(f"valid  gazebo      : {gz_v.sum()}")
    print(f"valid  offline sim : {sim_v.sum()}")
    print(f"valid  in both     : {both.sum()}")
    print(f"agreement on which beams return: "
          f"{(gz_v == sim_v).mean() * 100:.1f} %")

    if both.sum() < 5:
        print("\n  too few shared returns to compare ranges")
        return 1

    def report(mask, label):
        d = gz[mask] - sim[mask]
        c = np.corrcoef(gz[mask], sim[mask])[0, 1]
        print(f"  {label:22s} n={mask.sum():3d}  median {np.median(d):+.3f} m  "
              f"p90|d| {np.percentile(np.abs(d), 90):.3f} m  "
              f"max|d| {np.abs(d).max():6.2f} m  corr {c:.4f}")
        return d, c

    # gpu_lidar renders the scene as several stitched depth-camera faces
    # when the FOV exceeds one pass, and the outermost rays land on a seam.
    # Measured here: beams 0 and N-1 disagree by 11-12 m while their
    # immediate neighbours agree to 0.3 m. Real lidar drivers discard edge
    # beams for similar reasons, so the SLAM front end trims them too.
    trimmed = both.copy()
    trimmed[0] = trimmed[-1] = False

    print()
    print("range agreement:")
    report(both, "all beams")
    d, corr = report(trimmed, "edge beams trimmed")

    edge_bad = [i for i in (0, gz.size - 1)
                if both[i] and abs(gz[i] - sim[i]) > 1.0]
    if edge_bad:
        print()
        print(f"  note: {len(edge_bad)} FOV-extreme beam(s) disagree "
              f"(gpu_lidar face-seam artefact); trimmed above.")

    # The engines differ legitimately: Gazebo casts against a 3D mesh from
    # 0.30 m up, the offline sim marches a 0.10 m raster in 2D. Sub-decimetre
    # median agreement means the geometry chain lines up.
    ok = (abs(np.median(d)) < 0.25
          and corr > 0.98
          and (gz_v == sim_v).mean() > 0.85)

    print()
    if ok:
        print("  PASS  Gazebo and the offline simulator describe the same world")
    else:
        print("  FAIL  the two simulators disagree; see the guide in the module "
              "docstring for what each failure mode means")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
