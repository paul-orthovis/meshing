import os
import sys

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.WebGl import x3dom_renderer

renderer = x3dom_renderer.X3DomRenderer()

for path in sys.argv[1:]:

    print(path)

    assert os.path.exists(path)  # ...since OCCT just segfaults

    step_reader = STEPControl_Reader()
    step_reader.ReadFile(path)
    step_reader.TransferRoot()
    shape = step_reader.Shape()
    renderer.DisplayShape(shape)

renderer.render()

