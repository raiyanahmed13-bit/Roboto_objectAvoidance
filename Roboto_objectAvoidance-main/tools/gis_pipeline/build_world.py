"""Turn OSM building footprints into a Gazebo world.

Three design decisions worth stating, because two of them look wrong:

**Extruded meshes, not SDF `<polyline>`.** SDF has a polyline geometry that
extrudes a 2D outline, which is exactly what this needs -- but Ignition
never finished implementing it (gazebosim/gz-sim#186), so it silently
produces nothing. trimesh extrusion into a single OBJ is boring and works.

**Flat ground, even though we fetched a DEM.** The DEM drives *planning
cost*, not simulated terrain. A 2D lidar on real 3D terrain pitches out of
plane and feeds garbage into a 2D SLAM front end -- that would corrupt the
graded core of the project to gain a nicer screenshot. The GIS requirement
is slope-aware planning, which is fully satisfied without terrain physics.
A heightmap world remains an optional extra for one figure.

**One merged mesh, not 236 models.** Gazebo's per-model overhead dominates
at this count, and nothing needs buildings individually addressable --
discrepancy objects are separate models by design.

Usage:
    python -m tools.gis_pipeline.build_world [--site PATH]
"""

from __future__ import annotations

import argparse

from .common.crs import SiteSpec, load_site
from .common.paths import load_vector, world_dir

MODEL_NAME = "site_buildings"


def display_heights(site: SiteSpec, gdf):
    """Per-building heights, with untagged ones varied for appearance.

    OSM gave this site exactly ONE real height tag out of 236 footprints,
    so every other building falls back to `default_height_m`. Extruded flat,
    that renders as a field of identical slabs, which reads as broken rather
    than as a neighbourhood.

    Where `buildings.untagged_height_range_m` is set, untagged buildings
    (those sitting exactly on the default) get a height drawn from that
    range instead. Buildings that DO carry a real tag are never touched.

    THIS IS COSMETIC AND CANNOT MOVE ANY MEASUREMENT. The prior costmap and
    occupancy grid are built from 2D footprints and never read a height;
    the lidar plane sits at 0.30 m, so anything above roughly a metre is
    indistinguishable to it. Visual and collision geometry both come
    through here, so they cannot disagree with each other.

    The draw is seeded per building from its own centroid, so the world is
    reproducible: rebuilding gives byte-identical output.
    """
    import numpy as np

    cfg = site.raw["buildings"]
    rng_range = cfg.get("untagged_height_range_m")
    default_h = float(cfg["default_height_m"])
    heights = gdf["height_m"].astype(float)
    if not rng_range:
        return heights

    lo, hi = float(rng_range[0]), float(rng_range[1])
    out = []
    for geom, h in zip(gdf.geometry, heights):
        if not np.isclose(h, default_h) or geom.is_empty:
            out.append(h)                     # a real OSM tag; leave it
            continue
        # Seeded by position, so it is stable across rebuilds.
        seed = abs(hash((round(geom.centroid.x, 3),
                         round(geom.centroid.y, 3)))) % (2 ** 32)
        out.append(float(np.random.default_rng(seed).uniform(lo, hi)))
    return np.asarray(out)


def build_building_mesh(site: SiteSpec, margin_m: float = 0.0):
    """Extrude every footprint to its height and merge into one mesh."""
    import trimesh
    from shapely.geometry import box as shapely_box

    gdf = load_vector(site, "buildings")

    # Keep the buffer ring: the robot can see past the working extent, and
    # clipping at the boundary would leave a void where buildings should be.
    clip = shapely_box(*site.bbox_local(site.fetch_buffer_m + margin_m))
    gdf = gdf[gdf.geometry.intersects(clip)]
    heights = display_heights(site, gdf)

    meshes, skipped, first_error = [], 0, None
    for geom, height in zip(gdf.geometry, heights):
        if geom.is_empty or not geom.is_valid:
            skipped += 1
            continue
        try:
            meshes.append(trimesh.creation.extrude_polygon(geom, float(height)))
        except Exception as exc:
            # Degenerate footprints (slivers, self-touching rings) fail
            # triangulation. Dropping a handful is fine; silently emitting a
            # broken mesh is not. Keep the first reason -- if EVERY polygon
            # fails it is an environment problem (a missing triangulation
            # backend, say), not bad data, and the message must say so
            # instead of drowning in one warning per building.
            skipped += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"

    if not meshes:
        raise RuntimeError(
            f"no buildings could be extruded ({skipped} attempted).\n"
            f"first failure was -- {first_error}\n"
            f"If every polygon failed, trimesh most likely has no "
            f"triangulation backend: pip install mapbox-earcut"
        )
    if skipped:
        print(f"  note: {skipped} footprint(s) skipped; first was {first_error}")

    return trimesh.util.concatenate(meshes), len(meshes), skipped


def footprint_boxes(site: SiteSpec, margin_m: float = 0.0):
    """Oriented bounding box per building, for collision geometry.

    WHY NOT THE MESH ITSELF: DART's ODE collision detector segfaults in
    OdeMesh::fillArrays on the concatenated building mesh (236 disjoint
    components, non-manifold as a whole). Boxes are numerically robust and
    DART handles hundreds of them without complaint.

    The approximation is confined to PHYSICS. `gpu_lidar` renders the scene,
    so it reads the *visual* geometry -- the true extruded footprint -- and
    the SLAM input is therefore unaffected. Only contact response uses these
    boxes, and an oriented bounding box is conservative (never smaller than
    the building), which is the safe direction for a collision-count metric.
    """
    import numpy as np
    from shapely.geometry import box as shapely_box

    gdf = load_vector(site, "buildings")
    clip = shapely_box(*site.bbox_local(site.fetch_buffer_m + margin_m))
    gdf = gdf[gdf.geometry.intersects(clip)]
    # Same heights the visual mesh uses, so the two cannot drift apart.
    heights = display_heights(site, gdf)

    boxes, area_true, area_box = [], 0.0, 0.0
    for geom, height in zip(gdf.geometry, heights):
        if geom.is_empty or not geom.is_valid:
            continue
        rect = geom.minimum_rotated_rectangle
        if rect.geom_type != "Polygon":
            continue
        xs, ys = rect.exterior.coords.xy
        pts = np.column_stack([xs[:4], ys[:4]])

        e0 = pts[1] - pts[0]
        e1 = pts[2] - pts[1]
        w, l = float(np.hypot(*e0)), float(np.hypot(*e1))
        if w < 0.5 or l < 0.5:
            continue
        yaw = float(np.arctan2(e0[1], e0[0]))
        cx, cy = float(rect.centroid.x), float(rect.centroid.y)

        boxes.append((cx, cy, float(height), w, l, yaw))
        area_true += geom.area
        area_box += w * l

    inflation = (area_box / area_true) if area_true else float("nan")
    return boxes, inflation


MTL_NAME = "buildings.mtl"

# Matches the <material> in the model SDF. Both are needed: the SDF colour
# applies to primitives, the MTL to the mesh, and only the MTL reaches the
# renderer for mesh geometry.
_MTL = """# GENERATED by tools/gis_pipeline/build_world.py
newmtl building
Ka 0.62 0.60 0.56
Kd 0.72 0.70 0.66
Ks 0.10 0.10 0.10
Ns 8.0
d 1.0
illum 2
"""


def _export_obj(mesh, path) -> None:
    """Write the merged mesh as an OBJ with FLAT normals and a material.

    trimesh's default export writes bare `v` and `f` lines: no normals and
    no material. Both omissions are invisible in the file and unmistakable
    on screen --

      * with no material the OBJ loader warns `Missing material for
        shape[]` and Ogre falls back to its default WHITE. The tan colour
        in the model SDF never applies, because mesh geometry takes its
        material from the mesh, not from the SDF.
      * with no normals there is nothing to light against, so every face
        returns the same shade and the buildings read as flat white
        cutouts with no volume at all.

    `unmerge_vertices` splits shared vertices so each face gets its own,
    which makes the exported normals per-face rather than averaged across
    edges. That matters here: these are extruded prisms, and smooth-shading
    a building corner rounds off exactly the hard edge that tells the eye
    it is a solid box.
    """
    import trimesh

    mesh = mesh.copy()
    mesh.unmerge_vertices()             # flat shading, crisp edges
    mesh.fix_normals()                  # consistent outward winding

    obj = trimesh.exchange.obj.export_obj(
        mesh, include_normals=True, include_texture=False)

    # trimesh does not emit the material reference, so add it. `usemtl`
    # must precede the first face for the loader to bind it.
    lines = [ln for ln in obj.splitlines()
             if not ln.startswith(("mtllib", "usemtl"))]
    for i, ln in enumerate(lines):
        if ln.startswith("f "):
            lines.insert(i, "usemtl building")
            break
    obj = f"mtllib {MTL_NAME}\n" + "\n".join(lines) + "\n"

    path.write_text(obj, encoding="utf-8")
    (path.parent / MTL_NAME).write_text(_MTL, encoding="utf-8")


def _model_sdf(mesh_rel: str, boxes) -> str:
    collisions = []
    for i, (cx, cy, h, w, l, yaw) in enumerate(boxes):
        collisions.append(
            f"""      <collision name="c{i}">
        <pose>{cx:.3f} {cy:.3f} {h / 2:.3f} 0 0 {yaw:.6f}</pose>
        <geometry><box><size>{w:.3f} {l:.3f} {h:.3f}</size></box></geometry>
      </collision>"""
        )
    collision_xml = "\n".join(collisions)

    return f'''<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{MODEL_NAME}">
    <static>true</static>
    <link name="link">
      <!-- Visual is the true extruded footprint mesh. gpu_lidar renders the
           scene, so this is what the lidar actually measures against.
           Collision is one oriented box per building: DART's ODE mesh path
           segfaults on the concatenated mesh, and physics only needs
           contact response. See footprint_boxes() for the reasoning. -->
      <visual name="visual">
        <geometry><mesh><uri>{mesh_rel}</uri></mesh></geometry>
        <material>
          <ambient>0.62 0.60 0.56 1</ambient>
          <diffuse>0.72 0.70 0.66 1</diffuse>
          <specular>0.10 0.10 0.10 1</specular>
        </material>
      </visual>
{collision_xml}
    </link>
  </model>
</sdf>
'''


def _model_config() -> str:
    return f'''<?xml version="1.0"?>
<model>
  <name>{MODEL_NAME}</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <description>
    OSM building footprints extruded to their tagged heights, in the
    site-local metric frame. Generated by tools/gis_pipeline/build_world.py
    -- edit the pipeline, not this file.
  </description>
</model>
'''


def _world_sdf(site: SiteSpec) -> str:
    import math

    span = 2 * (max(site.width_m, site.height_m) + 2 * site.fetch_buffer_m)
    yaw = math.radians(site.spawn_yaw_deg)
    return f'''<?xml version="1.0" ?>
<!-- GENERATED by tools/gis_pipeline/build_world.py - do not edit by hand.
     Site: {site.name}   origin: ({site.lat0}, {site.lon0})

     The world origin IS the ROS `map` origin IS the local TM origin.
     x = East, y = North, z = Up. No model may introduce an extra offset. -->
<sdf version="1.8">
  <world name="{site.name}">

    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="ignition-gazebo-physics-system"
            name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system"
            name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-user-commands-system"
            name="ignition::gazebo::systems::UserCommands"/>
    <!-- gpu_lidar produces nothing at all without this plugin. -->
    <plugin filename="ignition-gazebo-sensors-system"
            name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <!-- Documentary: lets a reader check the georeferencing claim against
         the OSM source. Nothing in the stack depends on it. -->
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>{site.lat0}</latitude_deg>
      <longitude_deg>{site.lon0}</longitude_deg>
      <elevation>{site.alt0}</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>

    <!-- Ambient is deliberately LOW. At 0.6 it swamped the directional
         light, so every face returned nearly the same shade and the
         buildings looked like flat cutouts even once the mesh carried
         normals. 0.35 leaves enough fill to read shaded sides while
         letting the sun actually model the geometry.

         Shadows stay off: this machine renders through llvmpipe on the
         CPU, and shadow maps there cost more than they are worth. Face
         shading from the directional light supplies the depth cue. -->
    <scene>
      <ambient>0.35 0.35 0.38 1</ambient>
      <background>0.62 0.72 0.85 1</background>
      <shadows>false</shadows>
    </scene>

    <!-- Low and oblique, so walls facing the sun and walls facing away
         differ strongly. A near-overhead sun lights every roof equally and
         leaves the walls flat, which is what makes an extruded-footprint
         city look like paper. -->
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 50 0 0 0</pose>
      <diffuse>0.95 0.93 0.88 1</diffuse>
      <specular>0.15 0.15 0.15 1</specular>
      <direction>-0.6 0.45 -0.65</direction>
    </light>

    <!-- A dim fill from the opposite side so shaded walls stay legible
         rather than going to flat black. Cheap: no shadows, no extra
         geometry pass. -->
    <light type="directional" name="fill">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 50 0 0 0</pose>
      <diffuse>0.25 0.27 0.32 1</diffuse>
      <specular>0 0 0 1</specular>
      <direction>0.6 -0.45 -0.5</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal>
            <size>{span:.0f} {span:.0f}</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal>
            <size>{span:.0f} {span:.0f}</size></plane></geometry>
          <material>
            <ambient>0.35 0.37 0.33 1</ambient>
            <diffuse>0.45 0.48 0.42 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <include>
      <uri>model://{MODEL_NAME}</uri>
    </include>

    <!-- Spawn pose comes from site.yaml, so the robot starts at the same
         place in Gazebo and in the offline raycast simulator. z=0 because
         the model already places its own links at the right heights. -->
    <include>
      <uri>model://roboto_bot</uri>
      <name>roboto_bot</name>
      <pose>{site.spawn_x} {site.spawn_y} 0 0 0 {yaw:.6f}</pose>
    </include>

  </world>
</sdf>
'''


def build(site: SiteSpec) -> dict:
    wdir = world_dir(site)
    mdir = wdir / "models" / MODEL_NAME
    (mdir / "meshes").mkdir(parents=True, exist_ok=True)

    mesh, n_built, n_skipped = build_building_mesh(site)
    mesh_path = mdir / "meshes" / "buildings.obj"
    _export_obj(mesh, mesh_path)

    boxes, inflation = footprint_boxes(site)
    (mdir / "model.sdf").write_text(
        _model_sdf("meshes/buildings.obj", boxes), encoding="utf-8")
    (mdir / "model.config").write_text(_model_config(), encoding="utf-8")
    world_path = wdir / "site.sdf"
    world_path.write_text(_world_sdf(site), encoding="utf-8")

    return {
        "world": world_path,
        "mesh": mesh_path,
        "models_dir": wdir / "models",
        "n_built": n_built,
        "n_skipped": n_skipped,
        "n_faces": len(mesh.faces),
        "bounds": mesh.bounds,
        "mesh_mb": mesh_path.stat().st_size / 1e6,
        "n_boxes": len(boxes),
        "box_inflation": inflation,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None)
    args = ap.parse_args()

    site = load_site(args.site)
    print(f"site '{site.name}'")
    r = build(site)

    lo, hi = r["bounds"]
    print(f"  buildings  : {r['n_built']} extruded, {r['n_skipped']} skipped")
    print(f"  mesh       : {r['n_faces']} faces, {r['mesh_mb']:.1f} MB")
    print(f"  bounds x   : {lo[0]:8.1f} .. {hi[0]:8.1f} m")
    print(f"  bounds y   : {lo[1]:8.1f} .. {hi[1]:8.1f} m")
    print(f"  bounds z   : {lo[2]:8.1f} .. {hi[2]:8.1f} m")
    print(f"  collision  : {r['n_boxes']} oriented boxes "
          f"(area x{r['box_inflation']:.2f} vs true footprints)")
    print(f"  world      : {r['world']}")
    print()
    print("  run with:")
    print(f"    export IGN_GAZEBO_RESOURCE_PATH={r['models_dir'].resolve()}")
    print(f"    ign gazebo -s -r --headless-rendering {r['world']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
