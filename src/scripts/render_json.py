import pyglet
if "." in __name__:  # filed is called by a script outside of Scripts
    from .read_json import get_dict
    from .write_json import insert_edges, insert_rails
    from .abstract_geometry import approx_bezier
else:
    from read_json import get_dict
    from write_json import insert_edges, insert_rails
    from abstract_geometry import approx_bezier

def multiplied_coordinates(json_dict, window_res, padding, multiplier = False):  # TODO: subtract unused x and y pixels
    """
    'miltiplier' is a float coefficient used to zoom/unzoom displayed vertices, so
    that they fit the display. The final px length of a vertice is computed as its
    length in meters (from JSON) times this multiplier. Function also returns
    processed simple graph dictionary
    Uses: read_json.get_dict()

    Input: pythonised json graph data - "Vertices" and "Edges"(dict),
           window resolution (int,int),
           padding (int,int)
    OPTIONAL: multiplier (float OR False)
    Output: multiplier (float), reduced vertice dict: {"A" : [3, 6], "B" : [10, 5], ...}
    """
    vertices = json_dict["Vertices"]  # if (lowest-left point is [200, 60], set it to [0, 0], subtract 200 from every x and 60 from all y)
    edges = json_dict["Edges"]
    # Get measurements needed for full-network viewport (how much do we un-zoom?)
    max_x, max_y = 0, 0
    min_x, min_y = 0, 0
    coordinates = {}  # reduced vertice dict: {"A" : [3, 6], "B" : [10, 5], ...}
    for label, point in vertices.items():
        coordinates[label] = point["Coordinates"]
        if not isinstance(point["Coordinates"][0], list):  # 1 simple xy list
            x, y = point["Coordinates"]
            max_x = x if max_x < x else max_x
            max_y = y if max_y < y else max_y
            min_x = x if min_x > x else min_x
            min_y = y if min_y > y else min_y
        elif not isinstance(point["Coordinates"][0][0], list):  # line vertice
            xy1, xy2 = point["Coordinates"]
            for [x, y] in [xy1, xy2]:
                max_x = x if max_x < x else max_x
                max_y = y if max_y < y else max_y
                min_x = x if min_x > x else min_x
                min_y = y if min_y > y else min_y
        else:  # curve vertice
            [xy1, xy2, xy3] = point["Coordinates"][0]
            for [x, y] in [xy1, xy2, xy3]:
                max_x = x if max_x < x else max_x
                max_y = y if max_y < y else max_y
                min_x = x if min_x > x else min_x
                min_y = y if min_y > y else min_y

    # Now, acount for non-intersection parts of the edge (they might be further out)
    # Yes, Beziér control points are checked as regular ones. I'm not happy either
    complex_edges = [r for r in edges if "Shape" in r.keys()]
    for edge in complex_edges:
        for shape in edge["Shape"]:
            if isinstance(shape[0], list):  # bezier
                if isinstance(shape[-1], dict):
                    shape = shape[:-1]  # Chop off offset dict before interpolating
                actual_points = approx_bezier(shape, [], step=0.125)  # bezier reduced to 8 points
                for xy in actual_points:
                    max_x = xy[0] if max_x < xy[0] else max_x
                    max_y = xy[1] if max_y < xy[1] else max_y
                    min_x = x if min_x > x else min_x
                    min_y = y if min_y > y else min_y
            else:
                max_x = shape[0] if max_x < shape[0] else max_x
                max_y = shape[1] if max_y < shape[1] else max_y
                min_x = x if min_x > x else min_x
                min_y = y if min_y > y else min_y
    x_ratio = (window_res[0] - (padding[0]*2)) / max_x
    y_ratio = (window_res[1] - (padding[1]*2) - 80) / max_y  # subtract 80px for window head
    if not multiplier:
        multiplier = x_ratio if x_ratio < y_ratio else y_ratio
    return multiplier, coordinates  # float, dict

def graph_lines(data_source, float_rgb, window_res=(1920, 1080), padding=(150, 150), multiplier=False, step=0.005, drive_right = True, half_gauge = 2.1):
    """
    Converts a JSON road network graph into a set of pyglet vertices.
    Uses: read_json.get_dict()
          approx_bezier()  # note: both before and after zooming

    Input: json file path (str) OR ready JSON dictionary (dict),
           rgb tri-tuplet
    OPTIONAL: window resolution (int,int),
              padding (int,int),
              multiplier (float OR False)
              bezier step (float)
              drive right (bool)
    Output: array of pyglet indexed vertex lists
    """
    # set values to adjust viewport (fit all the edges into the window)
    if isinstance(data_source, str):
        json_dict = get_dict(data_source)
    else:
        json_dict = data_source
    vertices = json_dict["Vertices"]
    if "Edges" not in json_dict:  # Deduce edges, if necessary
        insert_edges(data_source)
        print("...insertion commanded by render_json.graph_lines().")
        json_dict = get_dict(data_source)
    edges = json_dict["Edges"]
    (r,g,b) = float_rgb   # rgb for edge color in floats

    # bulid indexed vertice list:
    multiplier, coordinates = multiplied_coordinates(json_dict, window_res, padding, multiplier)
    # ...coordinates format: {label: vertice xy OR [xy, xy] OR [xy, xy, xy]}
    graph_vertices = []
    registred_CPs = []  # list of smoothener CPs to efficiently avoid doubling smootheners
    for edge in edges:
        (vert_a, vert_b) = edge["Vertices"]
        coordinate_pair = [coordinates[vert_a], coordinates[vert_b]]
        border_xys = []  # UN-multiplier-ED border lines
        for i in [0, 1]:
            if not isinstance(coordinate_pair[i][0], list):  # vertice is 1 simple xy
                border_xys.append(coordinate_pair[i])
            elif not isinstance(coordinate_pair[i][0][0], list):  # vertice is a line
                border_xys.append(coordinate_pair[i][1-i])
                coordinate_pair[i] = [xy.copy() for xy in coordinate_pair[i]]  # protect coordinates from mutating
                (vert_x1, vert_y1) = coordinate_pair[i][0]
                (vert_x2, vert_y2) = coordinate_pair[i][1]
                vert_x1 = float(vert_x1 * multiplier + padding[0])  # resolution adjustments
                vert_x2 = float(vert_x2 * multiplier + padding[0])
                vert_y1 = float(vert_y1 * multiplier + padding[1])
                vert_y2 = float(vert_y2 * multiplier + padding[1])
                vertex = pyglet.graphics.vertex_list_indexed(2, [0,1],
                    ("v3f", [vert_x1,vert_y1,0, vert_x2,vert_y2,0]),
                    ("c3f", [r,g,b, r,g,b])
                )
                if vertex not in graph_vertices:
                    graph_vertices.append(vertex)  # appending the vertice itself as a line
            else:  # vertice is a curve
                border_xys.append(coordinate_pair[i][0][i-1])
                coordinate_pair[i] = [[xy.copy() for xy in coordinate_pair[i][0]]]  # protect coordinates from mutating
                CPs = coordinate_pair[i][0]
                #  zoom bezier smoothener's CPs:
                for j in range(3):
                    x, y = CPs[j]
                    x = float(x * multiplier + padding[0])
                    y = float(y * multiplier + padding[0])
                    CPs[j] = [x, y]
                smoothener_xys = approx_bezier(CPs, [], step * 5)
                if CPs not in registred_CPs and CPs[::-1] not in registred_CPs:  # avoid adding duplite bezier lines
                    registred_CPs.append(CPs)
                    for j in range(len(smoothener_xys)-1):
                        line = smoothener_xys[j:j+2]
                        vertex = pyglet.graphics.vertex_list_indexed(2, [0,1],
                            ("v3f", [line[0][0],line[0][1],0, line[1][0],line[1][1],0]),
                            ("c3f", [r,g,b, r,g,b])
                        )
                        if vertex not in graph_vertices:
                            graph_vertices.append(vertex)  # appending the vertice itself as a line

        [[x1, y1], [x2, y2]] = border_xys
        x1 = float(x1 * multiplier + padding[0])  # resolution adjustments
        x2 = float(x2 * multiplier + padding[0])
        y1 = float(y1 * multiplier + padding[1])
        y2 = float(y2 * multiplier + padding[1])

        # convert graph edges to a list of pyglet vertexes
        if "Shape" in edge.keys():  # multiple corners and Beziér curves - crossing-curves omitted!
            if "From" in edge.keys():  # displaying crossing smoothener
                [x1, y1] = edge["Shape"][0][0] if isinstance(edge["Shape"][0][0], list) else edge["Shape"][0]
                x1 = float(x1 * multiplier + padding[0])
                y1 = float(y1 * multiplier + padding[1]) 

            previous_corner = (x1, y1)
            for unscaled_corner in edge["Shape"]:
                if isinstance(unscaled_corner[0], list):  # Bezier curve
                    if isinstance(unscaled_corner[-1], dict):  # curve has defined Offsets dict
                        offsets = unscaled_corner.pop()["Offsets"]
                        points = approx_bezier(unscaled_corner, [], step, offsets)
                    elif "From" in edge.keys():
                        points = approx_bezier(unscaled_corner, [], step * 5)
                    else:
                        points = approx_bezier(unscaled_corner, [], step)

                    x1 =  float(points[0][0] * multiplier + padding[0])
                    y1 =  float(points[0][1] * multiplier + padding[0])
                    first_corner = (x1, y1)
                    for point in points[1:]:
                        x1, y1 = first_corner
                        x2 = float(point[0] * multiplier + padding[0])
                        y2 = float(point[1] * multiplier + padding[1])
                        first_corner = (x2, y2)
                        vertex = pyglet.graphics.vertex_list_indexed(2, [0,1],
                            ("v3f", [x1,y1,0, x2,y2,0]),
                            ("c3f", [r,g,b, r,g,b])
                        )
                        graph_vertices.append(vertex)
                    previous_corner = (x2, y2)  # rewrite old (x1, y1), that was just the first control point
                elif edge["Shape"].index(unscaled_corner) != 0:  # skip 1st iteration
                    x1, y1 = previous_corner
                    x2 = float(unscaled_corner[0] * multiplier + padding[0])
                    y2 = float(unscaled_corner[1] * multiplier + padding[1])
                    previous_corner = (x2, y2)
                    vertex = pyglet.graphics.vertex_list_indexed(2, [0,1],
                        ("v3f", [x1,y1,0, x2,y2,0]),
                        ("c3f", [r,g,b, r,g,b])
                    )
                    graph_vertices.append(vertex)

        else:  # direct A->B line
            vertex = pyglet.graphics.vertex_list_indexed(2, [0,1],
                ("v3f", [x1,y1,0, x2,y2,0]),
                ("c3f", [r,g,b, r,g,b])
            )
            graph_vertices.append(vertex)

    return graph_vertices

# SAFE TESTS:
#print(graph_vertices("../Data/capital.json", [100.0,100.0,100.0]))
#points = approx_bezier([[270,130],[310,130],[310,100]], 0.1)
#print(points)

def rails_lines(data_source, float_rgb, resolution=(1920, 1080), padding=(150, 150), multiplier=False, half_gauge=2.1, step=0.005, drive_right=True, copy=True):
    """
    Converts a JSON road network graph (including "Vertices", "Edges", and "Rails")
    into a set of pyglet vertices. Deduces parts of JSON (using write_json's functions),
    if necessary. Since 'Rails' obey conventional structure, just a wrapper around
    render_graph.graph_lines()
    Uses: read_json.get_dict()
          write_json.insert_edges(), write_json.insert_rails()

    Input: json file path (str) OR ready JSON dictionary (dict),
           rgb tri-tuplet
    OPTIONAL: window resolution (int,int),
              padding (int,int),
              half_gauge (float),
              drive_right (bool)
              multiplier (float OR False)
    Output: array of pyglet indexed vertex lists, possibly rewritten JSON
    """
    # input deduction and testing
    assert len(float_rgb) == 3 and all(isinstance(x, (int, float)) for x in float_rgb), "rgb invalid ~ render_json.rails_lines()"
    if isinstance(data_source, str):
        json_dict = get_dict(data_source)
        if "Edges" not in json_dict:  # Deduce edges, if necessary
            insert_edges(data_source)
            print("...insertion commanded by render_json.graph_lines().")
            json_dict = get_dict(data_source)
        if "Rails" not in json_dict:
            insert_rails(data_source, half_gauge, drive_right, copy)
            print("...insertion commanded by render_json.rails_lines().")
            json_dict = get_dict(data_source)
    else:
        json_dict = data_source
    assert isinstance(json_dict, dict), "data corrupt (failed loading json_dict) ~ render_json.rails_lines()"

    rails = json_dict["Rails"]
    return graph_lines(rails, float_rgb, window_res=resolution, padding=padding, drive_right=drive_right, step=step, multiplier=multiplier, half_gauge=half_gauge)
