"""Every SDF/XML artefact must be well-formed XML.

Motivation: `--` is illegal inside an XML comment, and it is the natural
way to write a dash in prose. sdformat's parser tolerates it, so Gazebo
loads the file happily while Python's ElementTree refuses -- meaning the
breakage surfaces only in tooling and tests, long after the fact.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
XML_FILES = sorted(
    p for pat in ("*.sdf", "*.urdf", "*.config", "package.xml")
    for p in ROOT.rglob(pat)
    if ".git" not in p.parts and "site.sdf" not in p.name
)


@pytest.mark.parametrize("path", XML_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_parses_as_xml(path):
    ET.parse(path)


@pytest.mark.parametrize("path", XML_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_double_dash_in_comments(path):
    text = path.read_text(encoding="utf-8")
    bad = [c for c in re.findall(r"<!--(.*?)-->", text, re.S) if "--" in c]
    assert not bad, (
        f"'--' inside an XML comment is illegal and breaks ElementTree while "
        f"sdformat silently accepts it. Offending comment starts: "
        f"{bad[0].strip()[:60]!r}"
    )


def test_generated_world_is_well_formed():
    """The world SDF is generated, so check the emitter's output, not a file."""
    from tools.gis_pipeline.build_world import _model_sdf, _world_sdf
    from tools.gis_pipeline.common.crs import load_site

    site = load_site()
    # A couple of representative collision boxes: (cx, cy, h, w, l, yaw).
    boxes = [(-10.0, 5.0, 6.0, 8.0, 12.0, 0.35),
             (30.0, -20.0, 3.0, 5.0, 5.0, 0.0)]
    for name, xml in (("world", _world_sdf(site)),
                      ("model", _model_sdf("meshes/buildings.obj", boxes))):
        ET.fromstring(xml)
        bad = [c for c in re.findall(r"<!--(.*?)-->", xml, re.S) if "--" in c]
        assert not bad, f"generated {name} SDF has an illegal comment: {bad[0][:60]!r}"
