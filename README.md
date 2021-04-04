osm2paths generates car tracks of any small OSM road network, intended to be used in simple traffic simulations.
It converts an exported OSM file (eg. map.osm) to a JSON dictionary defining the road network (crossroads and connections),
and car tracks (directed traces on commonplace two-way streets).

The tracks' graph is perfecly smooth - there are no sharp turns and any turn is defined as an infinitely zoomable quadratic
Bézier curve - it generates 'vector' paths.

=========== USAGE =============

1, place an .osm file inside data folder
2, call generate_rails() to create a .json files, arguments:

osm_path (string, required): path to source OSM file (eg. "data/map.osm")
half_gauge (float, default = 2): distance between road's central line and tracks in meters
min_radius (float, default = 4.5): minimum radius of osculating circle of curves in meters
drive_right (bool, default = True): regional road specification - True for Germany, False for the UK
display (bool, default = True): if set to True, displays generated traces in system window

CODE EXAMPLE:

generate_roads("data/kjoto.osm", 1.5, drive_right=False)
