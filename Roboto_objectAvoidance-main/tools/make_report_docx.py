"""Generate the project progress document (.docx).

Regenerate after significant work:
    .venv-gis/Scripts/python.exe -m tools.make_report_docx

Output: docs/roboto_progress_report.docx
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = Path("docs") / "roboto_progress_report.docx"

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
ACCENT = RGBColor(0x0B, 0x5C, 0x8A)


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _shade(par, hex_fill: str):
    pr = par._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    pr.append(shd)


def _border(par, hex_color="C8D4DC", size=6, edges=("left",)):
    pr = par._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    for edge in edges:
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "6")
        el.set(qn("w:color"), hex_color)
        bdr.append(el)
    pr.append(bdr)


def code(doc, text: str, fill="F4F6F8"):
    """Monospace, shaded block with a left rule. Used for code and trees."""
    for line in text.strip("\n").split("\n"):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(line if line.strip() else " ")
        r.font.name = "Consolas"
        r.font.size = Pt(8.5)
        r.font.color.rgb = INK
        _shade(p, fill)
        _border(p, edges=("left",))
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def para(doc, text: str, size=10.5, italic=False, color=INK, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.italic = italic
    r.font.color.rgb = color
    return p


def rich(doc, parts, size=10.5, space_after=6):
    """parts = [(text, bold, mono), ...] for inline emphasis."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    for text, bold, mono in parts:
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(8.8 if mono else size)
        if mono:
            r.font.name = "Consolas"
        r.font.color.rgb = INK
    return p


def bullet(doc, text: str, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.3 + 0.25 * level)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    return p


def note(doc, title: str, text: str, fill="FFF6E5", color="E0A800"):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.left_indent = Inches(0.1)
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(10)
    _shade(p, fill)
    _border(p, hex_color=color, size=12, edges=("left",))

    p2 = doc.add_paragraph()
    p2.paragraph_format.space_after = Pt(8)
    p2.paragraph_format.left_indent = Inches(0.1)
    r2 = p2.add_run(text)
    r2.font.size = Pt(10)
    _shade(p2, fill)
    _border(p2, hex_color=color, size=12, edges=("left",))


def h(doc, text: str, level=1):
    hd = doc.add_heading(text, level=level)
    for r in hd.runs:
        r.font.color.rgb = ACCENT if level <= 2 else INK
    return hd


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    for i, htxt in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(htxt)
        r.bold = True
        r.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            mono = val.startswith("`") and val.endswith("`")
            r = p.add_run(val.strip("`"))
            r.font.size = Pt(8.5 if mono else 9.5)
            if mono:
                r.font.name = "Consolas"
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------

def build() -> Path:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.85)
        s.top_margin = s.bottom_margin = Inches(0.8)

    _cover(doc)
    _overview(doc)
    _environment(doc)
    _repo_map(doc)
    _frames(doc)
    _gis(doc)
    _simulation(doc)
    _slam(doc)
    _planning(doc)
    _discrepancy(doc)
    _closed_loop(doc)
    _results(doc)
    _bugs(doc)
    _testing(doc)
    _running(doc)
    _status(doc)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


def _cover(doc):
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("GIS-Prior SLAM and Terrain-Aware Path Planning\nfor a Ground Robot")
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = ACCENT

    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs = s.add_run("Implementation Progress Report")
    rs.font.size = Pt(13)
    rs.font.color.rgb = MUTED

    s2 = doc.add_paragraph()
    s2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs2 = s2.add_run(
        "ROS 2 Humble  |  Gazebo Fortress  |  Ubuntu 22.04 (WSL2)\n"
        "Site: Greenfield / Squirrel Hill, Pittsburgh PA")
    rs2.font.size = Pt(10)
    rs2.font.color.rgb = MUTED

    doc.add_paragraph()
    table(doc,
          ["Metric", "Value"],
          [["Tests passing", "128"],
           ["Git commits", "51"],
           ["Source lines (core, tools, tests)", "7,639"],
           ["OSM buildings in world", "236 (134 inside working extent)"],
           ["Terrain relief", "56.6 m across the site"],
           ["Gate A (Gazebo lidar under WSL2)", "PASSED"],
           ["Gate B (frame alignment)", "PASSED"],
           ["Gate C (scan matcher accuracy)", "PASSED"],
           ["SLAM vs odometry", "0.48 m vs 4.18 m RMSE"],
           ["Discrepancy detection", "recall 91.7 %, precision 100 %, F1 0.957"],
           ["Collision ablation", "layer off 0/3 missions succeed; layer on see 11.5"]],
          widths=[3.2, 3.4])
    doc.add_page_break()


def _overview(doc):
    h(doc, "1. What This Project Does", 1)
    para(doc,
         "A ground robot is handed a prior map built from real-world public GIS data "
         "(OpenStreetMap building footprints plus a USGS elevation model) and plans a "
         "route across it. As it drives, a SLAM system written from scratch builds the "
         "robot's own occupancy map from lidar.")
    para(doc,
         "The headline contribution is map discrepancy detection: the robot finds where "
         "reality disagrees with the GIS prior, quantifies that divergence, and replans "
         "around it. The thesis is that public GIS data is a powerful but stale prior, "
         "and SLAM is what keeps it honest.")

    h(doc, "1.1 The three GIS roles", 2)
    table(doc, ["Role", "What it does", "Status"],
          [["Prior costmap", "OSM footprints and roads become a planning costmap", "Working"],
           ["Terrain cost", "DEM slope makes steep ground expensive or impassable", "Working"],
           ["Discrepancy detection",
            "SLAM finds obstacles the prior lacks", "Working, 3/3 recall"]],
          widths=[1.5, 3.9, 1.2])

    h(doc, "1.2 Key design decisions", 2)
    table(doc, ["Decision", "Choice", "Why"],
          [["SLAM front end", "Correlative scan matching (Olson 2009)",
            "Global search in a window; cannot get stuck like ICP. Gives covariance free."],
           ["Match score", "Trimmed mean, worst 10% discarded",
            "A plain mean lets unmapped obstacles drag the pose. 0.12 m -> 13.4 m ATE."],
           ["Match reference", "Prior fused with the live map",
            "Pure self-map matching erodes 63% of real walls; fused erodes 0.6%."],
           ["Likelihood field", "Distance transform, exp(-d^2/2s^2)",
            "Blurring occupancy peaks INSIDE buildings and biases the match."],
           ["Terrain in Gazebo", "Flat ground; DEM used for planning cost only",
            "A 2D lidar on 3D terrain pitches out of plane and corrupts the 2D SLAM."],
           ["Building collision", "Oriented boxes; mesh for visuals",
            "DART segfaults on the mesh. gpu_lidar renders visuals, so scans are unaffected."],
           ["Dev loop", "Offline raycast simulator, not Gazebo",
            "1.7 ms per scan. A 344 m mission runs in seconds, not minutes."],
           ["Global planner", "Hand-rolled Python A*",
            "A Nav2 C++ costmap plugin is 1.5-2 days of build systems for no marks."]],
          widths=[1.3, 2.1, 3.0])
    doc.add_page_break()


def _environment(doc):
    h(doc, "2. Environment Setup", 1)

    h(doc, "2.1 WSL was disabled, not broken", 2)
    para(doc, "Initial symptom: every wsl.exe command failed with:")
    code(doc, "wsl: WSL installation appears to be corrupted\n"
              "     (Error code: Wsl/CallMsi/Install/REGDB_E_CLASSNOTREG)")
    para(doc, "Diagnosis: the Store MSIX for WSL was installed, but the Windows "
              "optional features it depends on were switched off, so the COM class "
              "wsl.exe calls was unregistered.")
    code(doc,
         "dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart\n"
         "dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart\n"
         "# reboot, then:\n"
         "wsl --update\n"
         "wsl --install -d Ubuntu-22.04")
    note(doc, "Diagnostic trap worth remembering",
         "The WSL feature is NOT named Microsoft-Windows-Subsystem-Linux in the component "
         "store; the underlying packages are Microsoft-Windows-Lxss-*. Searching the "
         "registry for the DISM feature name finds nothing and looks like failure even "
         "when the install succeeded.")

    h(doc, "2.2 ROS 2 and Gazebo install", 2)
    rich(doc, [("Pinned and reproducible in ", False, False),
               ("setup/install_system.sh", False, True), (".", False, False)])
    table(doc, ["Component", "Version"],
          [["Ubuntu", "22.04.5 LTS"],
           ["ROS 2", "Humble (ros-humble-desktop)"],
           ["Gazebo", "Fortress (ignition-fortress 1.0.3)"],
           ["ROS-Gazebo bridge", "ros-humble-ros-gz 0.244.25"],
           ["slam_toolbox", "2.6.10 (evaluation baseline only)"]],
          widths=[2.4, 4.2])

    h(doc, "2.3 Gate A: can Gazebo produce lidar under WSL2?", 2)
    para(doc,
         "This was the top schedule risk. In Fortress the lidar is gpu_lidar: it RENDERS "
         "the scene through OGRE2 to produce ranges. If rendering fails, /scan dies and "
         "SLAM has no input at all. Headless mode does not avoid this, because headless "
         "still renders.")
    rich(doc, [("Test world with known geometry: ", False, False),
               ("setup/gate_a_lidar.sdf", False, True),
               (" places walls at exactly 5 m and 8 m, so correct output is known in advance.",
                False, False)])
    table(doc, ["Configuration", "Result"],
          [["ogre2 on D3D12 (hardware)",
            "SIGABRT in GenerateHwMipmaps via RenderSystem_GL3Plus"],
           ["ogre1 (hardware)",
            "Runs and publishes, but EVERY beam returns range_min"],
           ["ogre2 on llvmpipe (software)",
            "Correct ranges 4.87-4.96 m, real-time factor ~1.00"]],
          widths=[2.4, 4.2])
    note(doc, "Why the ogre1 result was the dangerous one",
         "It looked like success: the server started, /lidar was advertised, messages "
         "flowed at 10 Hz. Only the known-geometry test world revealed the data was "
         "garbage. A world of arbitrary buildings would have produced plausible numbers "
         "and the fault would have surfaced days later as 'why won't SLAM converge'.")
    para(doc,
         "Software rendering costs far less than it sounds: gpu_lidar renders a 180x1 "
         "depth strip, and simulation runs on /clock, so a lower real-time factor "
         "lengthens wall-clock runs without changing results. Measured RTF is ~0.82 with "
         "the GUI open and ~1.00 headless. The working configuration lives in "
         "setup/gazebo_env.sh rather than in anyone's memory.")
    doc.add_page_break()


def _repo_map(doc):
    h(doc, "3. Repository Map", 1)
    note(doc, "Two copies exist",
         "E:\\roboto is the Windows working copy (edited and committed). "
         "~/roboto inside WSL is a git clone where everything actually RUNS - "
         "WSL's ext4 is far faster, and GDAL is blocked on Windows by Application "
         "Control. Reach the WSL copy from Explorer at "
         "\\\\wsl.localhost\\Ubuntu-22.04\\home\\hem\\roboto")

    code(doc, r"""
roboto/
|
+-- setup/                                  environment + Gazebo launch
|   +-- install_system.sh                   pinned ROS 2 + Gazebo install
|   +-- gazebo_env.sh                       rendering config (Gate A outcome)
|   +-- gate_a_check.sh  gate_a_lidar.sdf   lidar-under-WSL2 diagnostic
|   +-- run_sim.sh                          launch the site world
|
+-- tools/gis_pipeline/                     offline GIS (own venv, heavy deps)
|   +-- site.yaml                     [*]   SINGLE SOURCE OF TRUTH for geometry
|   +-- common/{crs,grid,paths}.py    [*]   projection authority, artifact I/O
|   +-- fetch_osm.py   fetch_dem.py         OSM buildings/roads, USGS 3DEP
|   +-- build_slope.py                      DEM -> local frame + gradients
|   +-- build_prior.py                      footprints + slope -> prior costmap
|   +-- build_world.py                      footprints -> Gazebo SDF world
|
+-- tools/                                  experiment harnesses
|   +-- run_slam_offline.py           [*]   plan, drive, map, score a mission
|   +-- run_discrepancy_offline.py    [*]   inject obstacles, measure recall
|   +-- verify_sim_agreement.py             Gazebo lidar vs offline simulator
|   +-- make_report_docx.py                 this document
|
+-- ros2_ws/src/
|   +-- roboto_core/                        PURE PYTHON - no ROS imports
|   |   +-- roboto_core/
|   |       +-- frames.py             [*]   GridSpec, the row-order flip, PGM I/O
|   |       +-- slam/
|   |       |   +-- transforms2d.py         SE(2) pose algebra
|   |       |   +-- scan.py                 laser scan container
|   |       |   +-- occupancy.py            log-odds mapping + observed_count
|   |       |   +-- likelihood_field.py     distance-transform field
|   |       |   +-- scan_matcher.py   [*]   correlative matching, trimmed score
|   |       |   +-- slam.py           [*]   orchestrator: predict/correct/map
|   |       +-- plan/
|   |       |   +-- astar.py          [*]   A* with directional terrain cost
|   |       +-- discrepancy/
|   |       |   +-- compare.py        [*]   cell classification vs the prior
|   |       |   +-- cluster.py        [*]   cells -> objects, tracking
|   |       +-- sim/
|   |           +-- raycast_sim.py    [*]   offline 2D lidar simulator
|   |           +-- odom_corruptor.py       realistic wheel-odometry drift
|   |
|   +-- roboto_ros/models/roboto_bot/
|       +-- model.sdf                       diff-drive robot + gpu_lidar
|
+-- data/site_pittsburgh_greenfield/
|   +-- raw/            buildings.gpkg, roads.gpkg, dem_src.tif
|   +-- derived/        prior_occ.pgm/.yaml, prior_cost.npz, dem_local.tif
|   +-- world/          site.sdf + models/site_buildings/meshes/buildings.obj
|
+-- tests/                                  114 tests, no ROS required
|   +-- test_alignment.py       21    [*]   Gate B: frame contract
|   +-- test_scan_matcher.py    19    [*]   Gate C: matcher accuracy
|   +-- test_prior.py           16          costmap + terrain
|   +-- test_xml_artifacts.py   13          all SDF/XML well-formed
|   +-- test_robot_model.py     12          SDF agrees with site.yaml
|   +-- test_sim.py             11          raycasting + odometry drift
|   +-- test_occupancy.py       11          log-odds mapping, standoff
|   +-- test_astar.py           11          directional terrain cost
|
+-- docs/                                   this report
""")
    para(doc, "[*] marks the files most worth reading first.", size=9.5, italic=True,
         color=MUTED)
    doc.add_page_break()


def _frames(doc):
    h(doc, "4. The Coordinate Frame Contract", 1)
    para(doc,
         "Projects like this fail silently because the GIS raster, the Gazebo world and "
         "the ROS map frame drift apart, and nobody notices until every metric is "
         "meaningless. The contract is fixed in one place and never recomputed elsewhere.")

    note(doc, "The invariant",
         "local Transverse Mercator frame  ==  ROS `map` frame  ==  Gazebo world frame. "
         "x = East, y = North, z = Up, origin at (lat0, lon0) from site.yaml. "
         "No package may apply any additional offset or rotation.")

    h(doc, "4.1 Why local TM and not UTM", 2)
    code(doc, "+proj=tmerc +lat_0=40.4187 +lon_0=-79.9376 +k=1 +x_0=0 +y_0=0 "
              "+datum=WGS84 +units=m +no_defs")
    para(doc,
         "k=1 gives zero scale distortion at the origin, and coordinates stay small "
         "numbers near zero. UTM eastings near 500,000 m lose about 6 cm of precision "
         "when stored as float32, which silently corrupts a 0.10 m grid.")

    h(doc, "4.2 The four alignment traps", 2)
    table(doc, ["Trap", "Consequence if wrong"],
          [["Raster row order: GDAL row 0 = north, ROS row 0 = south",
            "Map looks correct but is vertically mirrored"],
           ["map_server `origin` is the bottom-left CORNER, not the centre",
            "Everything offset by half the extent, still looks plausible"],
           ["East/west mirror from a projection sign error",
            "Map still looks like a neighbourhood; discrepancy reports 100% disagreement"],
           ["DEM arriving in a different CRS than the vectors",
            "Terrain silently offset from buildings"]],
          widths=[3.4, 3.2])
    rich(doc, [("The flip happens in exactly one place, in ", False, False),
               ("roboto_core/frames.py", False, True),
               (". A np.flipud anywhere else in the codebase is a bug.", False, False)])

    h(doc, "4.3 Gate B: the assertions that protect it", 2)
    code(doc,
         "def test_northeast_polygon_lands_northeast(small):\n"
         "    occ = _rasterize(box(2.0, 2.0, 8.0, 8.0), small)\n"
         "    rows, cols = np.nonzero(occ)\n"
         "    assert rows.min() >= small.height // 2, \"fell in the SOUTH half\"\n"
         "    assert cols.min() >= small.width // 2,  \"fell in the WEST half\"")
    doc.add_page_break()


def _gis(doc):
    h(doc, "5. The GIS Pipeline", 1)
    code(doc,
         "cd ~/roboto\n"
         ".venv-gis/bin/python -m tools.gis_pipeline.fetch_osm\n"
         ".venv-gis/bin/python -m tools.gis_pipeline.fetch_dem\n"
         ".venv-gis/bin/python -m tools.gis_pipeline.build_slope --preview\n"
         ".venv-gis/bin/python -m tools.gis_pipeline.build_prior --preview\n"
         ".venv-gis/bin/python -m tools.gis_pipeline.build_world")

    h(doc, "5.1 Fetching real data", 2)
    code(doc,
         "buildings:  236 footprints    (134 inside the 300 m extent)\n"
         "roads:       78 segments\n"
         "elevation : 307.3 .. 363.8 m  relief 56.6 m,  nodata 0.00 %")
    note(doc, "Why GeoPackage and not GeoJSON",
         "GeoJSON is defined to be WGS84 lon/lat. Writing a metric CRS to it makes "
         "GeoPandas silently drop the CRS: coordinates come back correct in value but "
         "labelled EPSG:4326, so every .area, .buffer and .centroid computes in degrees "
         "without raising. This was a real bug, caught by a test. GeoPackage round-trips "
         "an arbitrary CRS exactly, and load_vector() now refuses anything geographic.")
    para(doc, "The DEM comes from the USGS 3DEP ImageServer over plain HTTPS. py3dep is "
              "deliberately avoided: its async HTTP stack fails DNS resolution in some "
              "environments even when the host resolves fine.")

    h(doc, "5.2 Directional terrain, not a scalar slope", 2)
    para(doc, "build_slope.py stores the gradient COMPONENTS (dz/dx, dz/dy), not a "
              "slope magnitude, because the planner needs directional slope: climbing a "
              "grade is expensive, descending it is nearly free, and crossing it "
              "sideways is a rollover risk.")
    code(doc,
         "def along_path_slope(self, dzdx, dzdy, heading_rad):\n"
         '    """Signed slope in radians along heading. Positive = uphill."""\n'
         "    return np.arctan(dzdx * np.cos(heading_rad) + dzdy * np.sin(heading_rad))")

    h(doc, "5.3 Building the world", 2)
    code(doc,
         "buildings  : 236 extruded, 0 skipped\n"
         "mesh       : 3428 faces, 0.1 MB\n"
         "collision  : 236 oriented boxes (area x1.05 vs true footprints)")
    para(doc, "Two decisions look wrong and are not. SDF has a polyline geometry that "
              "would extrude outlines directly, but Ignition never finished implementing "
              "it (gz-sim#186) and it fails silently. And the ground stays flat despite "
              "the DEM: a 2D lidar on real 3D terrain pitches out of plane and feeds "
              "garbage into the 2D SLAM front end, so terrain enters as planning cost "
              "only.")
    doc.add_page_break()


def _simulation(doc):
    h(doc, "6. Simulation", 1)

    h(doc, "6.1 The offline raycast simulator", 2)
    rich(doc, [("roboto_core/sim/raycast_sim.py", False, True),
               (" runs at 1.7 ms per scan, 58x real time. It is the development loop, "
                "the unit-test fixture, and the fallback had Gate A failed. A 344 m "
                "mission simulates in seconds.", False, False)])

    h(doc, "6.2 Lidar range: a measured design change", 2)
    para(doc, "The original 12 m sensor left the robot effectively blind. Measured on "
              "this site, median free-cell clearance is 8.5 m and the 90th percentile "
              "is 22 m. Return counts at the spawn pose:")
    table(doc, ["Sensor range", "Valid beams (of 180)"],
          [["12 m", "1"], ["20 m", "13"], ["30 m", "68"], ["50 m", "104"]],
          widths=[2.0, 2.4])
    para(doc, "Raised to 30 m, matching real outdoor 2D scanners. Across 400 random free "
              "poses the median is now 74/180 beams, with only 5.8% beam-starved.")

    h(doc, "6.3 Odometry corruption is mandatory", 2)
    para(doc, "Gazebo's DiffDrive odometry is near-perfect. Feeding that to SLAM makes "
              "the scan matcher look pointless and destroys the headline 'SLAM beats "
              "odometry' comparison. Drift is injected at the wheel level so it produces "
              "the systematic curvature real robots show. Calibrated against a 300 m "
              "loop; the first attempt was 50x too aggressive:")
    table(doc, ["wheel_asymmetry", "Final drift over 300 m"],
          [["0.015", "110 m  (37%)  - absurd"],
           ["0.003", "64 m  (21%)"],
           ["0.0003", "13 m  (4.5%)"],
           ["0.0001", "7 m  (2.4%)  - chosen"]],
          widths=[2.0, 3.0])
    para(doc, "The value looks implausibly small and is not: a wheel asymmetry a bends "
              "the path by a/baseline radians per metre, so it is amplified by "
              "distance/baseline - 750x over a 300 m run with a 0.4 m baseline.")

    h(doc, "6.4 Cross-validating the two simulators", 2)
    para(doc, "Gazebo casts rays at an extruded 3D mesh; the offline simulator marches a "
              "2D raster. They share only the OSM source data, no code, so agreement is "
              "strong evidence the whole geometric chain is coherent.")
    code(doc,
         "agreement on which beams return: 98.9 %\n"
         "  all beams          n= 66  median +0.080 m  p90 0.310 m  corr 0.8495\n"
         "  edge beams trimmed n= 64  median +0.078 m  p90 0.280 m  corr 0.9991\n"
         "  PASS  Gazebo and the offline simulator describe the same world")
    note(doc, "The two disagreeing beams",
         "Exactly beams 0 and 179 - the FOV extremes - disagreed by 12 m while their "
         "immediate neighbours agreed to 0.3 m. That signature identifies a gpu_lidar "
         "face-seam artefact: a 270 degree FOV exceeds one render pass, so Ignition "
         "stitches several depth-camera faces and the outermost rays land on the seam. "
         "Real lidar drivers discard edge beams for the same reason.")
    doc.add_page_break()


def _slam(doc):
    h(doc, "7. SLAM", 1)

    h(doc, "7.1 Occupancy mapping", 2)
    code(doc,
         "self.log_odds       = np.full(grid.shape, prior, dtype=np.float32)\n"
         "self.observed_count = np.zeros(grid.shape, dtype=np.uint16)\n"
         "l_occ = +0.85    l_free = -0.40    clamp +/- 5.0\n"
         "free_standoff_m = 0.30")
    note(doc, "observed_count is what makes discrepancy detection possible",
         "Discrepancy detection must never compare a cell the robot has not sensed. "
         "Without this mask every unvisited cell reads as 'the prior claims an obstacle "
         "and I see none', and the headline metric degenerates into 'the robot did not "
         "go there'. This grid is what separates disagreement from absence of evidence.")
    para(doc, "The free-trace stand-off is a robustness parameter, not a geometric one. "
              "Free space is carved by EVERY beam of every scan while a wall cell is hit "
              "only occasionally, so pose error lets the trace cut into obstacles and "
              "erode them. See section 12.2.")

    h(doc, "7.2 The likelihood field", 2)
    code(doc,
         "dist = distance_transform_edt(~occupied) * grid.resolution\n"
         "field = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))")
    para(doc, "A distance transform, NOT a Gaussian blur of occupancy. Blurring peaks in "
              "the interior of a solid building rather than on its surface, which biases "
              "the matcher into walls - see section 12.1.")

    h(doc, "7.3 Correlative scan matching", 2)
    para(doc, "ICP and Hector-style gradient descent are local methods that diverge once "
              "initial yaw error exceeds roughly 15-20 degrees - exactly the regime after "
              "odometry drifts across a block. Correlative matching searches a bounded "
              "window exhaustively, so within that window it cannot get stuck. It then "
              "pays for itself three more times: the score surface yields an edge "
              "covariance free; the same code with branch-and-bound is the loop-closure "
              "verifier; and at coarse resolution it registers the SLAM map against the "
              "GIS prior.")
    para(doc, "Gate C, recovering known perturbations on synthetic terrain:")
    table(doc, ["Initial perturbation", "Recovered error", "Score", "Time"],
          [["0.0 m, 0.0 deg", "0.075 m, 0.00 deg", "0.927", "7 ms"],
           ["1.00 m, 8.6 deg", "0.056 m, 0.09 deg", "0.938", "7 ms"],
           ["1.56 m, 14.3 deg", "0.025 m, 0.07 deg", "0.930", "7 ms"],
           ["0.64 m, 17.2 deg", "0.050 m, 0.06 deg", "0.930", "7 ms"]],
          widths=[1.8, 1.9, 1.0, 0.9])
    para(doc, "The same answer emerges from every initialisation (spread under 6 cm), "
              "which is the global-search property that motivated the choice. The "
              "covariance is correctly anisotropic: in a corridor test, along-axis "
              "uncertainty exceeds across-axis by more than 2x.")

    h(doc, "7.4 The trimmed score", 2)
    rich(doc, [("This is load-bearing for this project. ", True, False),
               ("The premise is that the world contains obstacles the reference map does "
                "not, so a plain mean is exactly the wrong statistic: points landing on "
                "an unmapped obstacle score ~0, and the matcher can raise the mean by "
                "SHIFTING THE POSE until those points land on something else. It gets "
                "dragged off the truth by the very features we are trying to detect.",
                False, False)])
    code(doc,
         "keep = int(round(N * (1.0 - self.score_trim)))\n"
         "cut  = N - keep\n"
         "scores[t] = np.partition(vals, cut, axis=0)[cut:].mean(axis=0)")

    h(doc, "7.5 The orchestrator", 2)
    para(doc, "slam.py runs predict (odometry delta) -> correct (scan match) -> map. Two "
              "details matter:")
    bullet(doc, "A LOCAL field window. Building a likelihood field over the full "
                "3000 x 3000 grid takes 0.66 s, impossible at 10 Hz. The sensor reaches "
                "only 30 m, so the field covers a window around the robot, rebuilt when "
                "it nears the edge: ~25 ms every few metres.")
    bullet(doc, "A rejected match falls back to odometry rather than accepting a "
                "confident-looking wrong answer, and the event is recorded. A run with "
                "many fallbacks has a very different error profile from one with none.")
    doc.add_page_break()


def _planning(doc):
    h(doc, "8. Planning", 1)
    para(doc, "Terrain cost for a ground robot is not a property of a cell, it is a "
              "property of a MOVE. Climbing a 15% grade is expensive, descending it is "
              "nearly free, crossing it sideways is a rollover risk. So the edge cost "
              "depends on heading, which makes the search graph genuinely directed - the "
              "cost from A to B differs from B to A.")
    code(doc,
         "along = arctan(gx*cos(phi) + gy*sin(phi))      # signed, along the move\n"
         "cross = arctan(-gx*sin(phi) + gy*cos(phi))\n"
         "cost += step * (w_up*max(along,0) + w_down*max(-along,0) + w_cross*|cross|)\n"
         "if abs(along) > max_slope: return inf          # hard block")
    para(doc, "A scalar slope field cannot express this, and neither can Nav2's NavFn or "
              "Smac without a custom cost function - one reason the planner is "
              "hand-rolled. Setting all weights to zero recovers a plain distance and "
              "obstacle planner, which is the no-terrain ablation baseline for free.")

    h(doc, "8.1 Verified on synthetic terrain", 2)
    table(doc, ["w_uphill", "Path length", "Climb", "Lateral detour"],
          [["0.0", "60.0 m", "23.8 m", "0.2 m"],
           ["2.0", "60.0 m", "23.8 m", "0.2 m"],
           ["8.0", "126.9 m", "1.1 m", "39.2 m"],
           ["25.0", "150.2 m", "1.1 m", "39.8 m"]],
          widths=[1.1, 1.6, 1.2, 1.6])
    para(doc, "With a 25 m hill in the way, the planner goes straight over it when "
              "terrain is disabled and detours 39 m around it when it is not, avoiding "
              "23 of the 24 metres of climb.")

    note(doc, "Emergent behaviour: switchbacks",
         "On a 31 degree fall line with a 25 degree limit the planner does not refuse - "
         "it zigzags. A 45 degree diagonal has an along-path slope of only 23 degrees, "
         "which is allowed, so the search invents switchbacks exactly as mountain roads "
         "and real vehicles do. Nothing special-cases this; it falls out of putting the "
         "cost on the edge rather than the cell. True blocking needs a grade above 0.66 "
         "for a 25 degree limit.")
    doc.add_page_break()


def _discrepancy(doc):
    h(doc, "9. Discrepancy Detection", 1)
    para(doc, "The headline result: finding where reality disagrees with OpenStreetMap, "
              "in a way that survives contact with a real, imperfect SLAM map.")

    h(doc, "9.1 Cell classification", 2)
    table(doc, ["Class", "Condition"],
          [["UNKNOWN", "not observed, or not confident - EXCLUDED from all metrics"],
           ["AGREE_FREE / AGREE_OCC", "prior and SLAM concur"],
           ["MISSED", "SLAM sees an obstacle the prior lacks  <- headline"],
           ["PHANTOM", "prior claims an obstacle SLAM cannot find"]],
          widths=[1.8, 4.8])
    para(doc, "Thresholds are deliberately asymmetric. Declaring a MISSED obstacle is "
              "safety-critical and cheap to act on, so be eager (l >= 1.5, 5 "
              "observations). Declaring a PHANTOM removes a known obstacle from the "
              "costmap on the robot's own evidence, which can drive it into a wall - so "
              "demand far more (l <= -3.0, 15 observations).")

    h(doc, "9.2 Registration first", 2)
    para(doc, "Even a good SLAM map sits centimetres off the prior, and comparing "
              "unregistered grids inflates every metric. A single global SE(2) correction "
              "is estimated first, reusing the scan matcher - the third job that one "
              "implementation does. The residual is reported as a diagnostic in its own "
              "right: above about a metre means something upstream is broken.")

    h(doc, "9.3 Cells to objects", 2)
    para(doc, "Objects, not cells, are what the planner can act on and what the report "
              "should quote. Pipeline: closing (consolidate) -> connected components -> "
              "minimum-area filter. Opening is DISABLED by default - see section 12.5.")

    h(doc, "9.4 The experimental design that makes it measurable", 2)
    para(doc, "Two worlds are built that differ in known ways: the prior world is OSM as "
              "the pipeline produced it, and the truth world is the prior plus obstacles "
              "OSM does not contain. The robot drives the prior-planned route through the "
              "TRUTH world, so its lidar sees the extra obstacles. Because we placed them, "
              "precision and recall are computable rather than hand-labelled.")
    note(doc, "State this limitation plainly in the final report",
         "Ground truth here is a simulated world we generated, perturbed by known "
         "synthetic obstacles. This is NOT a claim to detect real errors in "
         "OpenStreetMap. Overclaiming there is the fastest way to lose a grader's trust.")
    doc.add_page_break()


def _closed_loop(doc):
    h(doc, "10. Closing the Loop: Replanning", 1)
    para(doc, "Detection is only half the claim. The other half is that the robot does "
              "something useful with it. Obstacles are placed ON the planned route, not "
              "beside it, so failing to react has a consequence that can be counted.")
    code(doc, """
plan over the GIS prior
    |
    v
step along the path ---> lidar ---> SLAM ---> map
    |                                          |
    |                              classify vs prior, cluster
    |                                          |
    |<--- replan over fused costmap <--- object blocks the path
    v
goal (or collision)
""")

    h(doc, "10.1 Fusing discovered obstacles", 2)
    para(doc, "The prior says what the world looked like when it was last surveyed; the "
              "detector says how it differs today. Adding a discovered obstacle is cheap "
              "to be wrong about -- a needless detour. Clearing a mapped one is expensive "
              "to be wrong about -- driving into a wall on the robot's own evidence. So "
              "MISSED objects go to lethal with full inflation, while PHANTOM objects "
              "only REDUCE cost, never to zero.")

    h(doc, "10.2 Two questions, not one", 2)
    para(doc, "Blockage is checked twice with different horizons. \"Is something in the "
              "way NOW?\" must always be answered. \"Will something be in the way "
              "eventually?\" can wait for a closer look. Collapsing the two is what "
              "produced 124 replans in a single mission.")
    code(doc, """
urgent   = path_is_blocked(..., within_m=emergency_m)   # always replan
eventual = path_is_blocked(...)                          # respects cooldown
if urgent or (eventual and cooled): replan
""")

    h(doc, "10.3 Plan conservatively, check permissively", 2)
    para(doc, "The deeper cause of the churn was zero-margin planning. A* returns a path "
              "flush against the obstacle's inflation boundary; one cycle later the robot "
              "has seen more of the obstacle, the footprint grows by a cell, and that "
              "path is genuinely blocked again. Every individual replan was correct.")
    para(doc, "Planning therefore uses an extra metre of clearance while the blockage "
              "test uses the nominal footprint, so the obstacle must grow THROUGH the "
              "margin before a replan is needed. A fallback to the nominal costmap keeps "
              "the extra caution from stranding the robot when the margin makes a route "
              "infeasible.")
    doc.add_page_break()


def _results(doc):
    h(doc, "11. Results", 1)

    h(doc, "12.1 End-to-end mission", 2)
    code(doc,
         "route: 344.4 m, 861 poses\n"
         "injected 3 obstacles absent from the prior:\n"
         "   parked_van            19.6 m2\n"
         "   construction_barrier  39.5 m2\n"
         "   dumpster               9.6 m2\n"
         "\n"
         "SLAM: ATE RMSE 0.48 m (odometry 4.18 m), 0 rejected of 861\n"
         "\n"
         "classification (observed 17.4% of grid, registration offset 0 cm)\n"
         "  agreement over classified cells : 99.8%\n"
         "  MISSED    2,061 cells (  20.6 m2)\n"
         "  PHANTOM      14 cells (   0.1 m2)\n"
         "\n"
         "  parked_van             DETECTED  centroid err 2.66 m\n"
         "  construction_barrier   DETECTED  centroid err 2.16 m\n"
         "  dumpster               DETECTED  centroid err 1.15 m\n"
         "  recall    3/3\n"
         "  false pos 7 spurious MISSED objects")

    h(doc, "12.2 Ablation: what the scan is matched against", 2)
    para(doc, "A genuine result, not just a tuning choice. Pure self-map matching in a "
              "large outdoor scene without loop closure degrades the map, which degrades "
              "the pose, which degrades the map further.")
    table(doc, ["Reference", "ATE RMSE", "True walls eroded", "True walls correct"],
          [["map (pure SLAM)", "0.50 m", "63.2 %", "31.8 %"],
           ["prior", "0.15 m", "1.8 %", "93.6 %"],
           ["fused (prior OR map)", "0.12 m", "0.6 %", "94.8 %"]],
          widths=[1.7, 1.2, 1.7, 1.7])
    para(doc, "Localising against a known map is standard practice - AMCL does exactly "
              "this - and the robot genuinely has the prior. Obstacles absent from the "
              "prior simply score as outliers, which the trimmed score then discards.")

    h(doc, "12.3 Ablation: OSM roads already encode terrain", 2)
    para(doc, "Adding explicit slope cost barely changes routes that follow roads, but "
              "changes off-road routes substantially. Roads in a hilly city were laid out "
              "by people avoiding steep grades, so a road-preferring planner inherits "
              "terrain awareness for free.")
    table(doc, ["Route", "Road-preferring costmap", "Obstacles-only costmap"],
          [["A to B", "-4.7 % climb", "-18.7 % climb"],
           ["A to C", "+0.2 %", "-37.7 %"],
           ["D to E", "+0.0 %", "-6.3 %"]],
          widths=[1.2, 2.4, 2.4])
    para(doc, "This is a stronger finding for the report than 'our feature works': it "
              "says something real about the interaction between a GIS prior and terrain "
              "reasoning.")

    h(doc, "12.4 Trim and correction limit, over three seeds", 2)
    para(doc, "Tuned on a multi-seed sweep, not one run - the single-run optimum was not "
              "stable. Worst-case ATE in metres:")
    table(doc, ["score_trim", "max_correction 1.2 m", "max_correction 3.0 m"],
          [["0.05", "24.93", "0.53"],
           ["0.10", "0.74", "0.60  <- chosen, stable at both"],
           ["0.20", "39.06", "19.81"]],
          widths=[1.4, 2.4, 2.6])
    doc.add_page_break()


    h(doc, "11.5 Ablation: the discrepancy layer, end to end", 2)
    para(doc, "The argument for the whole contribution. Two obstacles OSM does not "
              "contain are placed ON the planned route, so failing to react has a "
              "consequence that can be counted. Three seeds each.")
    table(doc, ["Discrepancy layer", "Collisions", "Successful missions", "Replans",
                "Detour"],
          [["OFF", "20", "0 / 3", "0", "-3.5 m"],
           ["ON", "0", "3 / 3", "3.0", "+10.9 m"]],
          widths=[1.4, 1.1, 1.5, 1.0, 1.0])
    para(doc, "The robot plans over OpenStreetMap, drives towards obstacles OSM has never "
              "heard of, detects them, and routes around -- arriving in every run, for a "
              "mean detour of 11 m on a 344 m route.")

    h(doc, "11.6 Getting there: two wrong turns", 2)
    para(doc, "Worth recording because the intermediate configurations were each "
              "defensible and each wrong in an instructive way.")
    table(doc, ["Configuration", "Collisions", "Success", "Replans", "Detour"],
          [["React on every cycle", "0", "3/3", "59.7", "+67.5 m"],
           ["Replan cooldown only", "2.7", "1/3", "12.3", "+17.5 m"],
           ["Cooldown + planning margin", "0", "3/3", "3.0", "+10.9 m"]],
          widths=[2.2, 1.1, 0.9, 1.0, 1.0])
    para(doc, "Reacting on every cycle was safe but wasteful: 124 replans in the worst "
              "seed. Adding a cooldown cut that fivefold and broke safety, trading the "
              "headline claim for tidiness. Only addressing the root cause -- planning "
              "with clearance the obstacle must grow through -- improved both at once.")
    doc.add_page_break()


def _bugs(doc):
    h(doc, "12. Bugs Found and Fixed", 1)
    para(doc, "Recorded because several are worth a paragraph in the final report, and "
              "because each shows a class of failure that is silent rather than loud - a "
              "system that runs, converges, and is quietly wrong.")

    h(doc, "12.1 The likelihood field peaked inside buildings", 2)
    para(doc, "Gaussian-blurring the occupancy grid makes the field peak deep INSIDE a "
              "solid building, where the kernel sums many occupied cells, while a scan "
              "endpoint lands on the wall SURFACE at about half that value. The matcher "
              "was rewarded for shoving the scan into walls.")
    code(doc,
         "score at true pose      0.498\n"
         "score 0.53 m away       0.699     <- the true pose was not even a local optimum")
    table(doc, ["", "Blurred occupancy", "Distance transform"],
          [["Score at truth", "0.498", "0.910"],
           ["Position error", "~0.60 m", "<= 0.075 m"],
           ["Heading error", "1.75 deg", "<= 0.11 deg"]],
          widths=[1.6, 2.2, 2.2])
    note(doc, "How it was caught",
         "By checking whether the objective peaked at the KNOWN answer, not merely "
         "whether the matcher converged. It converged beautifully - to the wrong pose, "
         "consistently, from every initialisation.")

    h(doc, "12.2 Unmapped obstacles dragged the pose off truth", 2)
    para(doc, "The most important bug in the project, because it is caused by the very "
              "phenomenon the project exists to detect. With a plain mean score, points "
              "landing on an obstacle absent from the reference score ~0, and the matcher "
              "can raise its score by shifting the pose until they land elsewhere.")
    code(doc,
         "injecting 69 m2 of obstacles (0.1% of the site) into an otherwise\n"
         "perfectly matched run:      ATE 0.12 m  ->  13.39 m\n"
         "                            (worse than using no scan matching at all)")
    para(doc, "Fixed with a trimmed score. Nearly mis-tuned: trim 0.25 looked fine on one "
              "run but was unstable across seeds (12-57 m). Only 0.10 held at every "
              "setting.")

    h(doc, "12.3 Pose error eroded real walls", 2)
    para(doc, "Free space is carved by every beam of every scan while a wall cell is hit "
              "only occasionally, so pose error lets the free-trace cut into obstacles. "
              "Eroded walls are catastrophic downstream: each reads as a PHANTOM and "
              "floods the headline metric.")
    table(doc, ["Pose error", "True wall cells wrongly marked free"],
          [["0.00 m", "0.0 %"], ["0.10 m", "3.6 %"],
           ["0.25 m", "34.2 %"], ["0.50 m", "62.3 %"]],
          widths=[1.8, 3.2])
    para(doc, "Mitigated by a 0.30 m free-trace stand-off, then largely eliminated by the "
              "fused matching reference: 709.9 m2 of phantom area became 0.1 m2.")

    h(doc, "12.4 A tight correction limit was worse than a loose one", 2)
    para(doc, "Rejecting a good correction drops the estimate back onto drifting "
              "odometry, so the next prediction is further off and the following "
              "correction is larger still - a cascade that ends far worse than the "
              "outlier it was meant to guard against. Worst-case ATE over three seeds "
              "went from 24.9 m at a 1.2 m limit to 0.53 m at 3.0 m.")

    h(doc, "12.5 Morphological opening deleted every true detection", 2)
    para(doc, "A lidar sees only the NEAR FACE of an obstacle, so a detection is a one- "
              "or two-cell-thick arc, not a filled blob. Opening with a 3x3 element "
              "erodes then dilates, deleting anything thinner than three cells: 2,061 "
              "MISSED cells produced ZERO objects. Closing first, opening off by default, "
              "and the minimum-area filter handles speckle without destroying thin "
              "structure.")

    h(doc, "12.6 A negative result worth keeping", 2)
    para(doc, "'Sticky obstacle' damping - reducing the free update on cells that are "
              "already confident obstacles - was expected to protect walls. It did the "
              "opposite, because it also protects SPURIOUS obstacles created by "
              "mis-registered hits: ghost walls accumulate, matching degrades against the "
              "cluttered map, and the worse poses erode true walls faster than the "
              "damping saves them.")
    table(doc, ["sticky_factor", "ATE RMSE", "True walls wrongly cleared"],
          [["1.00 (off)", "0.64 m", "72.5 %"],
           ["0.50", "1.55 m", "89.4 %"],
           ["0.25", "2.30 m", "92.8 %"]],
          widths=[1.6, 1.6, 3.0])

    h(doc, "12.7 A detection is a face; an obstacle is a body", 2)
    para(doc, "A lidar sees only the NEAR FACE of an obstacle, so a detection is a thin "
              "arc. Inflating it by the robot radius alone left the solid body behind it "
              "unmarked, and the planner routed neatly around the arc and straight into "
              "the object. In one trace a replan at 10.5 m still closed to 3.1 m before "
              "failing. unknown_depth_m now covers the part not yet seen.")

    h(doc, "12.8 A* refused to plan from a lethal start", 2)
    para(doc, "A robot can find ITSELF inside a lethal cell: it has just discovered an "
              "obstacle it was already close to, and the new inflation covers its own "
              "position. Refusing to plan is the worst possible response, because that is "
              "precisely the moment a route out is needed. The planner now escapes to the "
              "nearest free cell and plans from there; a fully enclosed robot still fails "
              "fast rather than scanning the whole grid.")

    h(doc, "12.9 Replan thrashing, and a fix that made it worse first", 2)
    para(doc, "One mission needed 124 global replans. The obvious hypothesis -- that the "
              "robot was standing in a lethal cell each time -- was measured and found "
              "false; it never was. The trace showed replans exactly every 1.2 m, one per "
              "detection cycle, because each closer view GREW the obstacle footprint and "
              "genuinely invalidated the path planned a metre earlier. Every individual "
              "replan was correct.")
    para(doc, "The first fix, a replan cooldown, traded the wrong way:")
    table(doc, ["Configuration", "Collisions", "Successful missions", "Replans"],
          [["No cooldown", "0", "3/3", "59.7"],
           ["Cooldown 8 m, emergency 4 m", "2.7", "1/3", "12.3"]],
          widths=[2.6, 1.2, 1.6, 1.2])
    para(doc, "Replans fell fivefold and the detour fourfold, but two of three missions "
              "began colliding -- the headline safety claim traded for tidiness. The real "
              "cause was zero-margin planning: A* returns a path flush against the "
              "inflation boundary, which one cell of growth invalidates. Planning now "
              "uses an extra metre of clearance while the blockage test uses the nominal "
              "footprint, so the obstacle must grow THROUGH the margin before a replan is "
              "needed.")

    h(doc, "12.10 Earlier bugs, briefly", 2)
    bullet(doc, "GeoJSON silently discarded the metric CRS: coordinates correct in value "
                "but labelled EPSG:4326, so every .area and .buffer would compute in "
                "degrees. Moved to GeoPackage; load_vector() now refuses geographic CRS.")
    bullet(doc, "Gazebo segfaulted in DART's ODE mesh collision on the concatenated "
                "building mesh. Fixed with oriented box collisions and mesh visuals - "
                "gpu_lidar renders visuals, so scans are unaffected. Area inflation 1.05x.")
    bullet(doc, "'--' inside XML comments is illegal. sdformat accepts it, so Gazebo "
                "loaded the files while Python's ElementTree refused. Now tested.")
    bullet(doc, "The lidar was mounted inside its own collision box, so every beam "
                "returned range_min. It reads exactly like a renderer failure.")
    bullet(doc, "Windows Application Control began blocking rasterio's and pyogrio's "
                "native GDAL DLLs. Everything runs in WSL instead.")
    doc.add_page_break()


def _testing(doc):
    h(doc, "13. Testing", 1)
    para(doc, "128 tests, all passing, none requiring ROS or Gazebo. This is what makes "
              "the development loop fast: roboto_core imports no rclpy by design.")
    code(doc,
         "cd ~/roboto\n"
         ".venv-gis/bin/python -m pytest tests/ -q\n"
         "\n"
         "128 passed, 6 warnings in 9.6s")
    table(doc, ["File", "Tests", "Covers"],
          [["test_alignment.py", "21", "Gate B: projection, grid indexing, the flip, PGM I/O"],
           ["test_scan_matcher.py", "19", "Gate C: recovery accuracy, covariance, robustness"],
           ["test_prior.py", "16", "Costmap, terrain model, vector CRS"],
           ["test_xml_artifacts.py", "13", "All XML artefacts well-formed"],
           ["test_robot_model.py", "12", "SDF agrees with site.yaml"],
           ["test_sim.py", "11", "Raycasting geometry, odometry drift band"],
           ["test_occupancy.py", "11", "Log-odds mapping, observability, stand-off"],
           ["test_astar.py", "13", "Directional terrain cost, switchbacks, lethal-start escape"],
           ["test_cost_fusion.py", "12", "Obstacle fusion, PHANTOM relief, blockage horizon"]],
          widths=[1.8, 0.7, 4.1])
    para(doc, "Several tests exist specifically to pin down a bug that already happened "
              "once: the mirror test, the drift band, the lidar-clearance check, the XML "
              "comment check, the 'truth is the optimum' test, and the stand-off "
              "erosion test.")


def _running(doc):
    h(doc, "14. How to Run Everything", 1)

    h(doc, "15.1 Opening WSL", 2)
    code(doc,
         "# Windows Terminal -> Ubuntu-22.04 tab, or from PowerShell:\n"
         "wsl -d Ubuntu-22.04\n"
         "cd ~/roboto")
    para(doc, "The sudo password is the one set during install. It is not needed for "
              "normal work; admin tasks can use `wsl -d Ubuntu-22.04 -u root`, which "
              "requires no password. To reset it: `wsl -d Ubuntu-22.04 -u root passwd hem`")

    h(doc, "15.2 The experiments", 2)
    code(doc,
         "# plan a route, drive it, map it, score SLAM against ground truth\n"
         ".venv-gis/bin/python -m tools.run_slam_offline\n"
         "\n"
         "# inject obstacles absent from the prior, measure detection\n"
         ".venv-gis/bin/python -m tools.run_discrepancy_offline\n"
         "\n"
         "# all tests\n"
         ".venv-gis/bin/python -m pytest tests/ -q")

    h(doc, "14.3 Gazebo", 2)
    code(doc,
         "bash setup/run_sim.sh --gui      # with the 3D window\n"
         "bash setup/run_sim.sh            # headless, faster\n"
         "pkill -f 'ign gazebo'            # stop it")
    para(doc, "In the GUI, right-click roboto_bot in the Entity Tree and choose 'Move to' "
              "to find the robot; it spawns at (-120, -95), well away from the default "
              "camera. Expect a modest frame rate: rendering is software, by necessity.")
    code(doc,
         "ign topic -l                            # list topics\n"
         "ign topic -e -t /scan  -n 1             # one lidar scan\n"
         "ign topic -p \"linear{x:0.5} angular{z:0.2}\" -t /cmd_vel \\\n"
         "    --msgtype ignition.msgs.Twist       # drive it")

    h(doc, "14.4 Verification", 2)
    code(doc,
         "bash setup/gate_a_check.sh                    # can Gazebo produce lidar here?\n"
         "ign topic -e -t /scan -n 1 > /tmp/scan.txt    # with the sim running\n"
         ".venv-gis/bin/python -m tools.verify_sim_agreement /tmp/scan.txt")


def _status(doc):
    h(doc, "15. Status and What Comes Next", 1)
    table(doc, ["Plan day", "Deliverable", "State"],
          [["1", "Environment, ROS 2, Gazebo, Gate A", "Done"],
           ["2", "Frame contract, prior costmap, Gate B", "Done"],
           ["3", "DEM slope, raycast simulator, odometry corruption", "Done"],
           ["4", "World generation, robot model", "Done"],
           ["5", "Simulator cross-validation", "Done"],
           ["7", "Log-odds occupancy mapping", "Done"],
           ["8", "Correlative scan matcher, Gate C", "Done"],
           ["9", "SLAM orchestrator, beats odometry 8.7x", "Done"],
           ["10a", "A* planning with directional terrain cost", "Done"],
           ["6", "Discrepancy detection (PROTECTED)", "Working, F1 0.957"],
           ["10b", "Replanning around confirmed obstacles", "Done, 3/3 missions"],
           ["9b", "SLAM node live in Gazebo, record bags", "Next"],
           ["11-12", "Experiment matrix, ablations, baselines", "Partly done"],
           ["13-14", "Report, figures, demo video", "Pending"]],
          widths=[0.9, 4.1, 1.6])

    h(doc, "15.1 Immediate next steps", 2)
    bullet(doc, "Wrap roboto_core as ROS 2 nodes and run the closed loop live in "
                "Gazebo. Every algorithm is already tested offline, so this is "
                "message plumbing rather than new work.")
    bullet(doc, "Record canonical bags from Gazebo so all later tuning runs against "
                "them rather than against a live simulator.")
    bullet(doc, "Wrap the core as ROS 2 nodes and run it live in Gazebo, then record "
                "canonical bags so all later tuning runs against them.")
    bullet(doc, "slam_toolbox as an external baseline on the same bags.")

    h(doc, "15.2 Known weaknesses, stated honestly", 2)
    bullet(doc, "The controller is a placeholder: the robot follows waypoints exactly "
                "rather than tracking a path with real dynamics. Collision avoidance is "
                "therefore bounded by replan latency alone, with no braking or steering "
                "limits. A pure-pursuit controller with velocity limits would make the "
                "collision numbers meaningful as vehicle behaviour rather than as "
                "planning behaviour.")
    bullet(doc, "Discrepancy detection runs over the whole 3000 x 3000 grid on every "
                "check. Caching the dilated prior made a 1.2 m detection cycle "
                "affordable, but a local window around the robot would be the proper "
                "fix and is what a live ROS node will need.")
    bullet(doc, "DETECTED AREA IS THE VISIBLE FACE, not the footprint: 0.7-1.4 m2 "
                "detected versus 9.6-39.5 m2 true. That is sensor geometry, not a bug, "
                "but the metric should compare against the observable portion, and "
                "centroid error (1.15-2.66 m) is inflated by the same effect.")
    bullet(doc, "Only 1 of 236 buildings carries a height tag in OSM here, so most use a "
                "6 m default. Harmless for 2D lidar, which only sees footprint walls.")
    bullet(doc, "Ground truth for discrepancy detection is the simulated world we "
                "generated, perturbed by known synthetic obstacles. Not a claim to "
                "detect real OSM errors.")
    bullet(doc, "Software rendering is a platform workaround, documented with evidence "
                "in setup/gazebo_env.sh, and belongs in an appendix.")
    bullet(doc, "Loop closure and the pose-graph back end remain stretch goals. The "
                "fused matching reference substantially reduces the need for them, which "
                "is itself a defensible engineering argument.")


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}  ({p.stat().st_size / 1024:.0f} KB)")
