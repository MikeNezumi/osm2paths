"""
This is the main script, the application.

Inputs: GLOBALS (set below the docstring)
Output: Window animation + console metrics (potentially)
"""
WINDOW_RESOLUTION = (1920, 1020)  # 1020 + win head = 1080

import pyglet
import subprocess
import json
from pyglet.gl import *
from scripts.read_json import get_dict
from scripts.render_json import graph_lines, rails_lines
from scripts.write_json import replicate_json, insert_rails

def generate_rails(osm_path, half_gauge = 2, max_radius = 4.5, drive_right = True, display = True):
    """
    Integrates all components of the generator, from .osm file to .JSON with rails,
    optionally displays OpenGL lines of these rails in a system window
    Uses: Vendors/OsmToRoadGraph/run.convert_osm_to_roadgraph()
          read_json.get_dict()
          write_json.insert_rails()
          render_json.rails_lines()

    Input: osm_path (str), half_gauge (float), max_radius, drive_left (bool), display (bool)
    Output: all good -> 0
    """
    subprocess.run(["python", "vendors/OsmToRoadGraph/run.py", "-f", osm_path,  "-n", "c"])
    roads_JSON = get_dict(osm_path[:-3] + "pycgr")
    with open(osm_path[:-3] + "json", "w", encoding="utf-8") as json_file:
        json.dump(roads_JSON, json_file)
        print("Wrote graph in " + osm_path[:-3] + "json")

    insert_rails(osm_path[:-3] + "json", half_gauge, False, max_radius)
    print("\nInserted tracks into " + osm_path)

    # display window
    if display:
        win = pyglet.window.Window(*WINDOW_RESOLUTION, caption = "Generated JSON")
        @win.event
        def on_draw():
            roads = rails_lines("data/map.json", (1, 1, 1), padding = (15, 15), multiplier=1.5)
            for road in roads:
                road.draw(GL_LINES)

        pyglet.app.run()

    return 0
