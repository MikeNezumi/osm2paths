import json
from math import sqrt
if "." in __name__:  # file is called by a script outside of Scripts
    from .read_json import get_dict, order_neighbours, evaluate_offsets, smoothen_rails, add_crossings
    from .abstract_geometry import distance, intersection_point, bezier_intersection, beziers_crossing, offset_polyline, offset_bezier, orient_line
else:
    from read_json import get_dict, order_neighbours, evaluate_offsets, smoothen_rails, add_crossings
    from abstract_geometry import distance, intersection_point, bezier_intersection, beziers_crossing, offset_polyline, offset_bezier, orient_line


def replicate_json(file_path):
    """
    Takes a one-line road JSON nightmare and creates a readible .txt copy

    Input: absoute or relative path to .json file, eg. "lib/myData.json" (str)
    Output: example.txt file with proper newlines and indents (.txt)
    """
    # load data:
    path_bits = file_path.split("/")
    filename = path_bits.pop()
    with open(file_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    json_file.close()

    assert len(json_data) > 0  # loading successfull

    json_data = str(json_data)  # jsou_data is now string
    json_data = json_data.replace("'", "\"")  # correct quotes

    json_data = json_data.replace("""{"Vertices": {""", """{\n\t"Vertices": {\n\t\t""")  # before & after Vertices
    json_data = json_data.replace("""]}, """, """]}, \n\t\t""")  # before each vertice dict
    json_data = json_data.replace("""]}}, """, """]}\n\t},\n """)  # end Vertices

    json_data = json_data.replace(""""Edges": [""", """\n\t"Edges": [\n\t\t""")  # before & after Edges
    json_data = json_data.replace("""}, {"Vertices":""", """}, \n\t\t{"Vertices":""")  # inside Edges
    json_data = json_data.replace(""", "Shape": [""", """, \n\t\t\t"Shape": [""")  # before Shapes
    json_data = json_data.replace(""", {"Offse""", """, \n\t\t\t\t{"Offse""")  # bezier offsets
    json_data = json_data.replace("""]]}]]""", """]]}\n\t\t\t]]""")  # end last-num Offsets
    json_data = json_data.replace(""", False]}]]""", """, False]}\n\t\t\t]]""")  # end last-False Offsets
    json_data = json_data.replace("""]]},""", """]]\n\t\t},""")  # end Shapes
    json_data = json_data.replace("""}], """, """}\n\t], \n\n\t""")  # end Edges with Rails
    json_data = json_data.replace("""}] """, """}\n\t] \n""")  # end Edges without Rails


    if "Rails" in json_data:  # extra indent for everything in rails:
        rails_index = json_data.index("Rails")
        json_graph = json_data[:rails_index]
        json_rails = json_data[rails_index:]
        json_data = json_data.replace(""""Rails": {""", """\n\n\t"Rails": {""")  # before & after Vertices
        json_rails = json_rails.replace("""\n\t""", """\n\t\t""")  # add \t everywhere in Rails
        json_rails = json_rails.replace("""]}]}}""", """]}\n\t\t]\n\t}\n}""")  # last part of file
        json_rails = json_rails.replace("""}]}}""", """}\n\t\t]\n\t}\n}""")  # last last part of file (last rail edge has Rails)
        json_data = json_graph + json_rails

    base = file_path[:-5]  # 5 for ".json"
    with open(base + ".txt", "w", encoding="utf-8") as json_file:
        json_file.write(json_data)
    print("Made a legible .txt copy of " + file_path)

def zoom_graph(data_source, coefficient, copy = False):
    """
    Iterates through a JSON dictionary (directly as python dict or from file),
    multiplies all coordinates by the same coefficient and returns back.
    NOTE: SIDE EFFECT - mutates given data
    Uses: .replicate_json()

    Input: data_source (dict/str), coefficient (float), copy (bool)
    Output multiplied dict OR multiplied JSON into file (respecting data_source)
    """
    file = False
    if isinstance(data_source, str):
        file = True
        try:
            with open(data_source, "r", encoding="utf-8") as json_file:
                json_data = json.load(json_file)
                file = True
        except FileNotFoundError:
            print("File doesn't exist. ~ write_json.zoom_coordinates()")
            return False
    else:
        json_data = data_source

    assert "Vertices" in json_data.keys(), "JSON dict should include Vertices. ~ write_json.zoom_coordinates()"
    for vertice in json_data["Vertices"].values():
        # handled as lists, to rewrite JSON data:
        if isinstance(vertice["Coordinates"][0], list):  # complex vertice
            vertice["Coordinates"][0][0] *= coefficient
            vertice["Coordinates"][0][1] *= coefficient
            vertice["Coordinates"][1][0] *= coefficient
            vertice["Coordinates"][0][1] *= coefficient
        else:
            vertice["Coordinates"][0] *= coefficient
            vertice["Coordinates"][1] *= coefficient


    if "Edges" in json_data.keys():
        for edge in json_data["Edges"]:
            if "Shape" in edge:
                for section in edge["Shape"]:
                    if isinstance(section[0], list):  # bezier in shape
                        for xy in section:
                            if isinstance(xy, dict):
                                if xy["Offsets"][0]:
                                    xy["Offsets"][0][0] *= coefficient  # redundan now, useful for anticipated Offset rails
                                    xy["Offsets"][0][1] *= coefficient
                                if xy["Offsets"][1]:
                                    xy["Offsets"][1][0] *= coefficient
                                    xy["Offsets"][1][1] *= coefficient
                            else:
                                xy[0] *= coefficient
                                xy[1] *= coefficient
                    else:
                        section[0] *= coefficient
                        section[1] *= coefficient

    if file:
        with open(data_source, "w", encoding="utf-8") as json_file:
             json.dump(json_data, json_file)

        print("Extended coordinates in", data_source, "by", coefficient)
        if copy:
            replicate_json(data_source)

    return json_data

#replicate_json("../Data/map.json")

def insert_vertices(file_path, dict):
    """
    Converts a python "Vertices" dictionary into a json file

    Input: path to a new .json file OR a properly formatted "Vertices" python dict
    Output: (written in JSON file)
    """
    try:
        with open(file_path, "w", encoding="utf-8") as json_file:
             json.dump(dict, json_file)
             print("Added Vertices to " + file_path + "\n")
    except Exception as e:
        print(e)
        return False

def insert_edges(file_path, copy = False):
    """
    Primitive json "Edges" generator - defines a list of edges by their border nodes,
    computes "Cost" ONLY as distance - omits, curves, speed limit, traffic...
    Also, generated "Edges" don't include street names (hence 'Primitive' generator)
    Includes: label_value()
    Uses: .replicate_json()

    Input: json "Vertices" object  # describes the shape of network, OPTIONAL: copy (bool)
    Output: modifies json file - adds "Edges" array  # "Cost" is just computed length of edge
    """
    # label_value():
    # returns the int. value in established graph vertice naming system:
    # A = 0, G = 6, AD = 29 ect.
    #
    #   Input: label (str)
    #   Output: value of label (int)
    def label_value(label):
        value = 0  # Default: "A"
        label = label[::-1]
        for i in range(len(label)):
            assert 65 <= ord(label[i]) <= 90, "Graph label includes forbidden chars (only capitalised eng. alphabed allowed). ~ write_json.insert_edges()[label_value()]"
            value += (ord(label[i]) - 65) * (26 ** i)
        return value

    try:
        json_data = get_dict(file_path)
        edges = []
        pairs = []
        unordered_vertices = json_data["Vertices"]

        # sort vertice keys alphabetically to allow lazy loading. Continuous labels assumed, therefore using InsertSort
        sorted_keys = list(unordered_vertices.keys())
        for shuffled_i in range(1, len(sorted_keys)):  # iterating through elements (L-R)
            key = sorted_keys[shuffled_i]
            key_value = label_value(key)
            compared_i = shuffled_i - 1
             # move all elements of higher label value than key right by 1
            while compared_i >= 0 and key_value < label_value(sorted_keys[compared_i]):
                sorted_keys[compared_i+1] = sorted_keys[compared_i]
                compared_i -= 1

            sorted_keys[compared_i+1] = key  # insert key to its newly allocated spot

        # reorganize vertices according to sorted keys:
        vertices = {key : unordered_vertices[key] for key in sorted_keys}
        json_data["Vertices"] = vertices

        # get all edges as node pairs:
        for key in sorted_keys:
            neighbours = vertices[key]["Neighbours"]
            for node in neighbours:
                if label_value(node) > label_value(key):  # keys alph. sorted - lower value edges already established
                    pairs.append([key, node])

        # establish costs by measuring distances
        for pair in pairs:
            start_xy = vertices[pair[0]]["Coordinates"]
            target_xy = vertices[pair[1]]["Coordinates"]
            x_diff = abs(start_xy[0] - target_xy[0])
            y_diff = abs(start_xy[1] - target_xy[1])
            distance = sqrt(x_diff**2 + y_diff**2)  # Pythagorian theorem

            edge = {"Vertices": pair, "Cost": round(distance)}
            edges.append(edge)

        json_data["Edges"] = edges
        with open(file_path, "w", encoding="utf-8") as json_file:
             json.dump(json_data, json_file)

        if copy:  # make a legible .txt copy of auto-generated, 1-line json
            replicate_json(file_path)
            print("Added Edges to " + file_path + ", made a legible .txt copy")
        else:
            print("Added Edges to " + file_path)
        return edges

    except Exception as e:
        print(e)
        return False

# TODO: account for multiple parallel lines in 1 direction, if necessary - after SOČ
# perhaps TEMP - wholly skipped subbezier (in overcrossing) is now being simply deleted
def insert_rails(file_path, half_gauge, drive_right = True, mr = 12.5, copy = True):  # TODO - print 'progress messages' for done steps
    """
    Deduces all rails' vertices from graph data and adds them to the json blueprint
    half_gauge represents distance between center of the road and of the vehicle,
    drive_right characterizes whether cars in network drive right (eg. Germany) or left (eg. UK)
    mr stands for 'minimal radius' - minimal turning radius of the network in meters
    (...default - 12.5 meters, German car standard)

    Function's present inner structure is:
        (0 - define nested strip())
        1 - validate input JSON dict (must contain valid Vertices and Edges)
        2 - iterate through all intersections (graph vertices)
            * load and order all its neighbours (establish edges)
            * double each edge into 'intersection_rails' to find new rails' points
            * find 'intersection_rails' crossings - Rails: Vertices xys
        3 - add found Rails' Vertices into emerging pyJSON dict
        4 - deduce rails (edges) from vertices, add into pyJSON dict
        5 - add Shapes of complex rails (Rails' Edges) derived from complex graph edges
            * divide Shape by type into polyline and bezier graph segments
            * double shape segments one by one, then glue them back together in correct driving order
            * Chop off multi-segment Shapes' inner overcrossings and insert Shapes to rails_dict
        6 - Fix Shapes' incorrect, ignored 1st points and overflowing end bits
            * Correct Shape's first, incorrect, ignored point
            * Chop off Shapes' overflowing last points
        7 - Recompute beziers to fit inside Offsets
        8 - Smoothen out all corners and insert crossings
        9 - Write finished "Rails" pyJSON into the source .json file
            * (OPTIONAL) Create legible .txt copy
    Includes: strip()
    Uses: abstract_geometry.offset_polyline(), .offset_bezier(), .intersection_point(), .bezier_crossing(), .orient_line(), .bezier_intersection()
          read_json.get_dict(), .order_neighbours(), .smoothen_rails()
          .replicate_json

    Input: json file (path str), half_gauge (int) IN METERS, drive_right (bool), step (float), mr (False or float - meters) copy (bool)
    Output: modifies json file - adds "Rails" dictionary
    """

    """ ~~~~~~~~~~ (0 - define nested strip()) ~~~~~~~~~~ """
    # strip():
    #   Strips vertice indexes from label ("ACE41" -> "ACE")
    #
    # Input: label (str)
    # Output: stripped label (str)
    def strip(label):
        label = list(label)
        ciphers = set([str(num) for num in range(10)])
        for char in label[::-1]:
            if char in ciphers:
                label.pop()
        return "".join(label)

    """ ~~~~~~~~~~ 1 - validate input JSON dict (must contain valid Vertices and Edges) ~~~~~~~~~~ """
    try:
        json_data = get_dict(file_path)
        vertices = json_data["Vertices"]
        edges = json_data["Edges"]
        assert vertices and edges, "JSON invalid (\"Vertices\" or \"Edges\" could not be loaded). ~ write_json.insert_rails()"
    except KeyError:
        print("JSON invalid (\"Vertices\" or \"Edges\" could not be loaded). ~ write_json.insert_rails()")
    assert 1 < half_gauge < 10, "given road width (half_gauge) is out of this world! ~ write_json.insert_rails()"

    """ ~~~~~~~~~~ 2 - iterate through all intersections (graph vertices) ~~~~~~~~~~ """
    # 2.1 # finding rail points of all intersections:
    rails_dict = {"Vertices":{}, "Edges": []}  # will be written inside JSON
    for vertice_key in vertices:
        vertice_points = []  # cross points of intersection' rails
        vertice = vertices[vertice_key]
         # edges stemming from vertice must be in counter-clockwise order...
        intersection_rails = []  # list of [xy1, xy2] pairs (1 pair for each rail)

        # 2.2 # doubling edges into rails, ALWAYS: 1, right-of-edge 2, left-of-edge:

        # orders neighbour vertices counter-clockwise (clockwise == False)
        neighbours = order_neighbours(vertice_key, vertices, False)
        for neighbour in neighbours:  # go through all neighbours
            try:
                doubled_edge = next(edge
                    for edge in edges
                        if edge["Vertices"] == [vertice_key, neighbour]
                        or edge["Vertices"] == [neighbour, vertice_key]
                )
            except StopIteration:
                print("Could not find edge for", [vertice_key, neighbour], "- meaning:")
                print("Invalid entry data (Vertices' Neighbours don't cross-match). ~ write_json.insert_rails()")
                return False

            xy1 = vertice["Coordinates"]
            neighbour_index = 1 if doubled_edge["Vertices"].index(neighbour) == 1 else -2  # second or second last (is edge oriented vertice -> neighbour?)

            if "Shape" not in doubled_edge.keys():  # simple line
                xy2 = vertices[neighbour]["Coordinates"]
                intersection_rails.append(offset_polyline([xy1, xy2], half_gauge, True))  # First append right rail
                intersection_rails.append(offset_polyline([xy1, xy2], half_gauge, False)) # Then append left rail
            elif not isinstance(doubled_edge["Shape"][1-abs(neighbour_index)][0], list):  # 1st (or -1st for inverted) segment is a line (Ie. is not Bezier)
                xy2 = doubled_edge["Shape"][neighbour_index]  # 2nd polyline control point
                intersection_rails.append(offset_polyline([xy1, xy2], half_gauge, True))  # First append right rail
                intersection_rails.append(offset_polyline([xy1, xy2], half_gauge, False)) # Then append left rail
            else:  # edge is bezier, append doubled control points as a rail
                points = doubled_edge["Shape"][1-abs(neighbour_index)][::3-abs(neighbour_index)*2]  # sliced either ::1 (no change) or ::-1 (reversed)
                assert len(points) == 3, "Only quadratic (3-control-points) beziers are allowed. ~ write_json.insert_rails()"
                # only append first offset subbeziers:
                if mr:
                    iterations = int((distance(*points[:2]) + distance(*points[1:])) // (mr * 3))  # adjust subdivision to the length of bezier's control polygon
                else:
                    iterations = 2

                iterations = 2 if iterations > 2 else iterations
                intersection_rails.append(offset_bezier(points, half_gauge, True, split_iterations=iterations))  # First append right rail
                intersection_rails.append(offset_bezier(points, half_gauge, False, split_iterations=iterations)) # Then append left rail

        # shuffle first rail to the end to change order to L, R-L, R-L, (...) , R-L, R
        first_vertice = intersection_rails.pop(0)
        intersection_rails.append(first_vertice)

        # 2.3 # find 'intersection_rails' crossings - Rails: Vertices xys

        # ...first found intersection point (or intersection line) is always between: 1st vertice's left rail and 2nd vertice's right rail
        for i in range(len(intersection_rails) // 2):
            if len(neighbours) == 1:  # this vertice is a dead-end, no intersection point
                # find out which end of edge is further from graph:
                for rail in intersection_rails:
                    distances = []
                    ref_point = vertices[neighbours[0]]["Coordinates"]
                    if isinstance(rail[0][0], list):  # dead-end is bezier
                        end_points = [rail[0][0], rail[0][-1]]
                    else:  # dead-end is line (edge itself or 1st polyline section)
                        end_points = rail

                    for end_point in end_points:
                        distances.append(distance(end_point, ref_point))

                    dead_point = rail[0] if distances[0] > distances[1] else rail[1]
                    vertice_points.append(dead_point)

            if isinstance(intersection_rails[2*i][0][0], list) or isinstance(intersection_rails[2*i+1][0][0], list):  # at least one bezier in currently computed rail pair
                if isinstance(intersection_rails[2*i][0][0], list):
                    vertice_point = beziers_crossing(intersection_rails[2*i], intersection_rails[2*i+1])[0]
                else:
                    vertice_point = beziers_crossing(intersection_rails[2*i+1], intersection_rails[2*i])[0]
            else:
                vertice_dict = intersection_point(intersection_rails[2*i], intersection_rails[2*i+1], cut_lines=True)
                if vertice_dict:
                    intersection_rails[2*i] = vertice_dict["line1"]
                    intersection_rails[2*i+1] = vertice_dict["line2"]
                    vertice_point = vertice_dict["point"]
                else:
                    vertice_point = False

            if vertice_point != False:  # point found
                vertice_points.append(vertice_point)

            else:  # lines don't cross
                vertice_line = []
                for rail in [intersection_rails[2*i], intersection_rails[2*i+1]]:
                    if isinstance(rail[0][0], list):  # line is bezier
                        rail = orient_line(vertice["Coordinates"], [rail[0][0], rail[0][-1]])  # transforms bezier into properly oriented line
                        vertice_line.append(rail[0])  # beginning of properly ordered line abstracted from bezier
                    else:
                        rail = orient_line(vertice["Coordinates"], rail)  # Order lines' points by proximity to vertice point:
                        vertice_line.append(rail[0])  # beginning of properly ordered line
                    # Insert beginnings of a rail - [[xy], [xy]] - Vertice Coordinates are a line!
                vertice_line = vertice_line[::(drive_right*2)-1]  # vertice line counter-clockwise, flip it if drive_right == False (0)
                vertice_points.append(vertice_line)

        if len(neighbours) == 1:  # parallel lines - skip crossing process in step 3 (right below)
            for i in range(2):
                if isinstance(vertice_points[i][0], list):  # vertice_point is bezier
                    rails_dict["Vertices"][vertice_key + str(i+1)] = {"Coordinates" : vertice_points[i][0]}
                else:  # vertice_point is part of line
                    rails_dict["Vertices"][vertice_key + str(i+1)] = {"Coordinates" : vertice_points[i]}

                # adhere to "NeighbourVertices" naming convention:
                # [first's left rail, counter-clockwise second's right rail] (from present vertice's perspective)
                if i == 0:
                    rails_dict["Vertices"][vertice_key + str(i+1)]["NeighbourVertices"] = [neighbours[0], " "]
                else:
                    rails_dict["Vertices"][vertice_key + str(i+1)]["NeighbourVertices"] = [" ", neighbours[0]]
            continue

        """ ~~~~~~~~~~ 3 - write Rails' Vertices into emerging pyJSON dict ~~~~~~~~~~ """
        # Write JSON vertices:
        for i in range(len(vertice_points)):
            rails_dict["Vertices"][vertice_key + str(i+1)] = {"Coordinates" : vertice_points[i]}
            rails_dict["Vertices"][vertice_key + str(i+1)]["NeighbourVertices"] = [neighbours[i]]  # making use of the prior counter-clockwise ordering
            try:
                rails_dict["Vertices"][vertice_key + str(i+1)]["NeighbourVertices"].append(neighbours[i+1])  # Neighbours: [left rail's, right rails's]
            except IndexError:  # last intersection - 2nd neighbour is right rail of first neighbour's edge
                rails_dict["Vertices"][vertice_key + str(i+1)]["NeighbourVertices"].append(neighbours[0])

    """ ~~~~~~~~~~ 4 - deduce rails (edges) from vertices, add into pyJSON dict ~~~~~~~~~~ """
    # deduce rails from vertices, thanks to naming convention:
    for key in rails_dict["Vertices"].keys():  # add Neighbours list to vertices
        rails_dict["Vertices"][key]["Neighbours"] = []

    for vertice_label, vertice_data in rails_dict["Vertices"].items():
        neighbours = []
        label = strip(vertice_label)
        # inserting Neighbours in vertices:
        searched_neighbours = vertice_data["NeighbourVertices"]
        for neighbour_label, neighbour_data in rails_dict["Vertices"].items():
            if neighbour_label == vertice_label:
                continue
            # insert "Rails": "Edges"
            if strip(neighbour_label) == searched_neighbours[0] and neighbour_data["NeighbourVertices"][1] == label:
                rails_dict["Vertices"][vertice_label]["Neighbours"].insert(0, neighbour_label)
                if drive_right == False:  # rail Edges format: [start, end]
                    rails_dict["Edges"].append({"Vertices": [vertice_label, neighbour_label]})
            elif strip(neighbour_label) == searched_neighbours[1] and neighbour_data["NeighbourVertices"][0] == label:
                rails_dict["Vertices"][vertice_label]["Neighbours"].append(neighbour_label)
                if drive_right:
                    rails_dict["Edges"].append({"Vertices": [vertice_label, neighbour_label]})

    """ ~~~~~~~~~~ 5 - add Shapes of complex rails (Rails' Edges) derived from complex graph edges ~~~~~~~~~~ """
    # modify the shapes of those Rails edges based on complex graph edges:
    # note for direction - first load and compute, eventually reverse order only at the end
    complex_edges = {}  # format: set(vertice1, vertice2) : [shape]
    for edge in json_data["Edges"]:  # find complex edges
        if "Shape" in edge.keys():
            complex_edges[tuple(edge["Vertices"])] = edge["Shape"]

    # complex rails' last part wasn't chopped off in step 2, it can only be done ex-post
    unchopped_shapes = {}  # dict, fotmat: {rail_index : complex rail, ...}
    # insert "Shape" into Rails' Edges
    for rail_index in range(len(rails_dict["Edges"])):  # iterate through all rails
        label1 = rails_dict["Edges"][rail_index]["Vertices"][0]
        label2 = rails_dict["Edges"][rail_index]["Vertices"][1]
        original_labels = [strip(label1), strip(label2)]
        if tuple(original_labels) in complex_edges.keys() or tuple(original_labels[::-1]) in complex_edges.keys():  # rail should have complex Shape
            original_shape = complex_edges[tuple(original_labels)] if tuple(original_labels) in complex_edges.keys() else complex_edges[tuple(original_labels[::-1])]

            # 5.1 # divide doubled Shape into individual polyline and bezier sublines:
            shape_segments = []  # Bezier distinguished by virtue of being a nested list, as customary
            segment_index = 0
            polyline_started = False  # algorithm assumes we begin on bezier
            for shape_index in range(len(original_shape)):  # going through the entire "Shape"
                if isinstance(original_shape[shape_index][0], list):  # bezier encountered
                    if polyline_started:
                        segment_index += 1  # move from polyline to this new segment
                    polyline_started = False
                    shape_segments.append([original_shape[shape_index]]) # bezier distinguished by being nested list, as is conventional
                    segment_index += 1  # move to next segment
                else:  # polyline point encountered
                    if polyline_started == False:
                        shape_segments.append([])  # new segment buffer
                        polyline_started = True
                    shape_segments[segment_index].append(original_shape[shape_index])

            # 5.2 # double shape segments one by one, then glue them back together in correct driving order (into doubled_segments):
            doubled_segments = []
            for segment in shape_segments:
                if isinstance(segment[0][0], list):  # Bezier curve, add shifted control points
                    if tuple(original_labels) in complex_edges.keys():  # control points are in the right direction
                        if mr:
                            iterations = int((distance(*segment[0][:2]) + distance(*segment[0][1:])) // (mr * 3))  # adjust subdivision to the length of bezier's control polygon
                        else:
                            iterations = 2

                        iterations = 2 if iterations > 2 else iterations
                        subbeziers = offset_bezier(segment[0], half_gauge, drive_right, split_iterations=iterations)
                        for subbez in subbeziers:
                            doubled_segments.append([subbez])  # drive_right True -> we want right rail
                    elif tuple(original_labels[::-1]) in complex_edges.keys():  # control points are in opposite direction - reverse control points, append to start
                        if mr:
                            iterations = int((distance(*segment[0][:2]) + distance(*segment[0][1:])) // (mr * 3))  # adjust subdivision to the length of bezier's control polygon
                        else:
                            iterations = 2

                        iterations = 2 if iterations > 2 else iterations
                        subbeziers = offset_bezier(segment[0][::-1], half_gauge, drive_right, split_iterations=iterations)
                        wrapped_subbeziers = []
                        for subbez in subbeziers:
                            wrapped_subbeziers.append([subbez])
                        doubled_segments = wrapped_subbeziers + doubled_segments

                else:  # polyline, add shifted points
                    if tuple(original_labels) in complex_edges.keys():  # polyline is in right direction
                        doubled_segments.append(offset_polyline(segment, half_gauge, drive_right))  # drive_right True -> we want right rail
                    elif tuple(original_labels[::-1]) in complex_edges.keys():  # polyline is in opposite direction - reverse rail points, append to start
                        doubled_segments = [offset_polyline(segment, half_gauge, 1-drive_right)[::-1]] + doubled_segments  # append to front

            # 5.4 # Chop off multi-segment Shapes' inner overcrossings and insert Shapes to rails_dict:
            if len(doubled_segments) == 1:  # just 1 segment
                rails_dict["Edges"][rail_index]["Shape"] = doubled_segments[0]
            else:
                # solve inner crossings, only then insert:
                doubled_index = 0
                while doubled_index < len(doubled_segments) - 1:  # list may dynamically expand, this prevents overflow
                    segment = doubled_segments[doubled_index]
                    next_segment = doubled_segments[doubled_index+1]
                    if isinstance(segment[0][0], list) and isinstance(next_segment[0][0], list): # segments: bezier ; bezier
                        chop_point = bezier_intersection(segment, next_segment)
                        last_dict = int(isinstance(segment[0][-1], dict))  # segment ends on dict -> 1 / doesn't -> 0
                        if chop_point:  # interection exists
                            if chop_point != segment[0][-1]: # interection exists, it's not just a touch
                                if isinstance(segment[0][-1], dict):
                                    segment[0][-1]["Offsets"][1] = chop_point
                                else:
                                    segment[0].append({"Offsets":[False, chop_point]})

                                if chop_point != next_segment[0][-1]:  # next bezier needs to be chopped, it's not just a touch
                                    next_segment[0].append({"Offsets":[chop_point, False]})
                            doubled_segments[doubled_index] = segment
                            doubled_segments[doubled_index+1] = next_segment
                        elif segment[0][-1-last_dict] != next_segment[0][0]:  # beziers don't touch, insert connecting line (complex_vert alternative)
                            doubled_segments.insert(doubled_index + 1, [segment[0][-1-last_dict], next_segment[0][0]])  # insert new "poly"line between beziers
                            doubled_index += 1  # move on to next segment

                    elif isinstance(segment[0][0], list): # segments: bezier ; line
                        offsets_dict = segment[0].pop() if isinstance(segment[0][-1], dict) else False
                        chop_point = bezier_intersection(segment, [next_segment[0], next_segment[1]])
                        if offsets_dict:  # offsets were lost in bezier_intersection()
                            segment[0].append(offsets_dict)
                        if chop_point:
                            if isinstance(segment[0][-1], dict):
                                segment[0][-1]["Offsets"][1] = chop_point
                            else:
                                segment[0].append({"Offsets":[False, chop_point]})
                            next_segment[0] = chop_point
                        else:
                            last_dict = int(isinstance(segment[0][-1], dict))  # segment ends on dict -> 1 / doesn't -> 0
                            next_segment = [segment[0][-1-last_dict]] + next_segment  # insert point at the beginning of next polyline
                        doubled_segments[doubled_index] = segment
                        doubled_segments[doubled_index+1] = next_segment

                    elif isinstance(next_segment[0][0], list): # segments: line ; bezier
                        chop_point = bezier_intersection(next_segment, [segment[-1], segment[-2]])
                        if chop_point:
                            segment[-1] = chop_point
                            next_segment[0].append({"Offsets":[chop_point, False]})
                        else:
                            segment.append(next_segment[0][0])  # append bezier's point to this polyline
                        doubled_segments[doubled_index] = segment
                        doubled_segments[doubled_index+1] = next_segment

                    else:  # segments: line ; line
                        chop_point = intersection_point([segment[-1], segment[-2]], [next_segment[0], next_segment[1]])
                        if chop_point:
                            segment[-1] = chop_point
                            next_segment[0] = chop_point
                        else:
                            segment.append(next_segment[0])  # append point to this polyline
                        doubled_segments[doubled_index] = segment
                        doubled_segments[doubled_index+1] = next_segment
                    doubled_index += 1

                rails_dict["Edges"][rail_index]["Shape"] = []
                for doubled_segment in doubled_segments:  # finally insert Shape to rails_dict
                    if isinstance(doubled_segment[0][0], list):  # appending bezier
                        rails_dict["Edges"][rail_index]["Shape"].append(doubled_segment[0])
                    else:  # appending polyline
                        rails_dict["Edges"][rail_index]["Shape"] += doubled_segment

            unchopped_shapes[rail_index] = rails_dict["Edges"][rail_index]  # add for multi-edge corrections in step 6

    """ ~~~~~~~~~~ 6 - Fix Shapes' incorrect, ignored 1st points and overflowing end bits ~~~~~~~~~~ """
    # chop off last part of complex edges, that ignore intersection point (bit of line or bit of curve):
    for rail_index, unchopped_rail in unchopped_shapes.items():

        # 6.1 # Correct Shape's first, incorrect, ignored point
        vert_label = unchopped_rail["Vertices"][0]  # current rail's start vertice label
        # load standardized (vertice) xy
        if isinstance(rails_dict["Vertices"][vert_label]["Coordinates"][0], list): # Complex vertice [[xy], [xy]]
            ignored_start = rails_dict["Vertices"][vert_label]["Coordinates"][1]
        else:  # Simple point
            ignored_start = rails_dict["Vertices"][vert_label]["Coordinates"]

        if isinstance(rails_dict["Edges"][rail_index]["Shape"][0][0], list):  # correcting 1st control point of bezier
            suspect_subbeziers = []  # ignored_start may cut some of those
            subbez_index = 0
            while isinstance(rails_dict["Edges"][rail_index]["Shape"][subbez_index][0], list):  # scans through subbeziers, will run at least once
                # create a protected copy to prevent bezier_crossing from overriding Offsets:
                if not isinstance(rails_dict["Edges"][rail_index]["Shape"][subbez_index][-1], dict):
                    subbez_copy = [xy.copy() for xy in rails_dict["Edges"][rail_index]["Shape"][subbez_index]]
                else:  # omit offset from creating subbez_copy
                    subbez_copy = [xy.copy() for xy in rails_dict["Edges"][rail_index]["Shape"][subbez_index][:-1]]

                suspect_subbeziers.append(subbez_copy)
                subbez_index += 1
                # suspecing last subbez, while condition would overflow
                if len(rails_dict["Edges"][rail_index]["Shape"]) == subbez_index:
                    break

            (crossed_start, crossed_index) = beziers_crossing(suspect_subbeziers, ignored_start)
            if crossed_start:
                # chop off omitted beginning:
                rails_dict["Edges"][rail_index]["Shape"] = rails_dict["Edges"][rail_index]["Shape"][crossed_index:]  # cut off entirely omitted subbeziers, perhaps TEMP?
                offsets = rails_dict["Edges"][rail_index]["Shape"][0][-1]  # protect offsets from bezier_intersection()
                offsets_defined = isinstance(rails_dict["Edges"][rail_index]["Shape"][0][-1], dict)  # Were offsets already inserted?
                if offsets_defined:
                    rails_dict["Edges"][rail_index]["Shape"][0][-1]["Offsets"][0] = ignored_start
                else:
                    rails_dict["Edges"][rail_index]["Shape"][0].append({"Offsets":[ignored_start, False]})
            else:  # ignored point is not part of bezier
                if isinstance(rails_dict["Vertices"][vert_label]["Coordinates"][0], list):  # Coordinates already a complex vertice [[xy], [xy]]
                    rails_dict["Vertices"][vert_label]["Coordinates"][1] = rails_dict["Edges"][rail_index]["Shape"][0][0]
                else:
                    rails_dict["Vertices"][vert_label]["Coordinates"] = [rails_dict["Vertices"][vert_label]["Coordinates"], rails_dict["Edges"][rail_index]["Shape"][0][0]]  # Make coordinates a complex vertice
        else:  # correcting 1st point of polyline
            rails_dict["Edges"][rail_index]["Shape"][0] = ignored_start  # rewrite 1st polyline point

        # 6.2 # Chop off Shapes' overflowing last point
        end_vertice = next(vert  # finding the vertice at the end of doubled complex edge
            for vert in rails_dict["Vertices"]
                if vert == unchopped_rail["Vertices"][1]
        )
        sibling_vertice = next(vert  # finding sibling graph vertice (needed to fing its crossed copy)
            for vert in rails_dict["Vertices"][end_vertice]["NeighbourVertices"]
                if vert != strip(vert_label)  # target's NeighbourVertices are *this one* (vert_label) and *the other one*. We want the other one
        )
        if sibling_vertice != " ":  # ignore dead-end vertices, no cutting required there...
            try:
                ignored_label = next(rail["Vertices"][0]  # finding crossed intersection
                    for rail in rails_dict["Edges"]
                        if rail["Vertices"][0] == end_vertice
                        and sibling_vertice in rail["Vertices"][1]
                )
            except StopIteration:
                print("JSON doesn't have properly linked Vertices' Neighbours ~ write_json.insert_rails()")
                return False

            if isinstance(rails_dict["Vertices"][ignored_label]["Coordinates"][0], list):  # is intersection a complex vertice?
                ignored_end = rails_dict["Vertices"][ignored_label]["Coordinates"][0]
            else:
                ignored_end = rails_dict["Vertices"][ignored_label]["Coordinates"]

            if type(rails_dict["Edges"][rail_index]["Shape"][-1][-1]) in [list, dict]:  # unchopped_rail ends on a bezier
                last_dict = isinstance(rails_dict["Edges"][rail_index]["Shape"][-1][-1], dict)  # unchopped_rail ends on a bezier with defined Offsets
                if last_dict:
                    offsets = rails_dict["Edges"][rail_index]["Shape"][-1].pop(-1)["Offsets"]

                if ignored_end == rails_dict["Edges"][rail_index]["Shape"][-1][-1]:  # does part of bezier need to be chopped off at all?
                    continue

                else:  # check for beziers' crossing:
                    end_subbeziers = []  # ignored_start may cut some of those, in end-start order! (respective to vertice)
                    negative_index = -1
                    while isinstance(rails_dict["Edges"][rail_index]["Shape"][negative_index][0], list):  # iterate through ending subbeziers
                        # create a protected copy to prevent bezier_crossing from overriding Offsets:
                        if not isinstance(rails_dict["Edges"][rail_index]["Shape"][negative_index][-1], dict):
                            subbez_copy = [xy.copy() for xy in rails_dict["Edges"][rail_index]["Shape"][negative_index]]
                        else:  # omit offset from creating subbez_copy
                            subbez_copy = [xy.copy() for xy in rails_dict["Edges"][rail_index]["Shape"][negative_index][:-1]]
                        end_subbeziers.append(subbez_copy)
                        negative_index -= 1
                        # suspecing first subbez, while condition would underflow
                        if negative_index == -len(rails_dict["Edges"][rail_index]["Shape"]) - 1:
                            break

                    (crossed_end, crossed_index) = beziers_crossing(end_subbeziers, ignored_end)
                    if crossed_end:
                        # chop off omitted end subbeziers:
                        if crossed_index != 0:  # [:-0] slice would delete whole Shape
                            rails_dict["Edges"][rail_index]["Shape"] = rails_dict["Edges"][rail_index]["Shape"][:-crossed_index]
                        if last_dict and crossed_index == 0:
                            rails_dict["Edges"][rail_index]["Shape"][-1].append({"Offsets":[offsets[0], ignored_end]})
                        else:
                            rails_dict["Edges"][rail_index]["Shape"][-1].append({"Offsets":[False, ignored_end]})

                    elif isinstance(rails_dict["Vertices"][ignored_label]["Coordinates"][0], list):  # no intersection, modify complex vertice
                        rails_dict["Vertices"][ignored_label]["Coordinates"][0] = rails_dict["Edges"][rail_index]["Shape"][-1][-1]
                    else:   # no intersection, complexify vertice
                        rails_dict["Vertices"][ignored_label]["Coordinates"] = [rails_dict["Edges"][rail_index]["Shape"][-1][-1], rails_dict["Vertices"][ignored_label]["Coordinates"]]
            else:  # unchopped_rail ends on a polyline:
                rails_dict["Edges"][rail_index]["Shape"][-1] = ignored_end

    """ ~~~~~~~~~~  7 - Recompute beziers to fit inside Offsets  ~~~~~~~~~~ """
    rails_dict["Edges"] = evaluate_offsets(rails_dict["Edges"])

    """ ~~~~~~~~~~  8 - Smoothen out all corners and insert crossings ~~~~~~~~~~ """
    rails_dict = smoothen_rails(rails_dict, mr)
    json_data["Rails"] = rails_dict
    json_data["Vertices"] = vertices
    json_data["Edges"] = edges
    json_data = add_crossings(json_data, mr, drive_right)

    """ ~~~~~~~~~~  9 - Insert finished "Rails" pyJSON into the source .json file ~~~~~~~~~~ """
    with open(file_path, "w", encoding="utf-8") as json_file:
         json.dump(json_data, json_file)

    print("Added Rails to " + file_path)
    # 9.1 #: OPTIONAL 9 - create legible .txt copy)
    if copy:
        replicate_json(file_path)


# MAKE A DESTROYABLE test_graph.json TO RUN THIS TEST:
#edges = insert_edges("../Data/BezError.json", True)
#print(edges)
#zoom_graph("../Data/sm_rails.json", 1.5, True)
#insert_rails("../Data/test.json", 4, False)
#replicate_json("../Data/sm_rails.json")
