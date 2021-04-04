import json
import utm
import pyglet
from math import pi
if "." in __name__:  # filed is called by a script outside of Scripts
    from .abstract_geometry import distance, get_k, narrow_bezier, smoothen_lines, smoothen_combined, smoothen_beziers, includes_point, \
    extended_intersection, check_curvature, biaxial_connector, connect_line, connect_bezier, extend_abiscissa, bezier_intersection, angle
else:
    from abstract_geometry import distance, get_k, narrow_bezier, smoothen_lines, smoothen_combined, smoothen_beziers, includes_point, \
    extended_intersection, check_curvature, biaxial_connector, connect_line, connect_bezier, extend_abiscissa, bezier_intersection, angle

def get_dict(file_path, half_gauge = 1.5, max_curvature = 12.5, objects = None):  # TODO - check that SOC doc mentions it's a .pycgr parser
                                          # TODO - implement automatic curve-registering
                                          # TODO - should gap_tolerance really equal 1.5 * max_curvature here? (it's 3 * mr for smootheners)
                                          # TODO - implement registering isolated abiscissa (after SOČ)
    """
    Loads all data of a .pycgr or JSON file, or only its specified TOP LEVEL object/s.
    If given .pycgr file, parses its data into .JSON dict. The nested function compute_label()
    converts integer ID to a standardized capital label, eg. EFA.
    Network settings - half_gauge and max_curvature - are necessary for Shapes simplification
    Returns corresponding pythonic arrays and dictionaries
    Includes: compute_label(), .turns_clockwise()

    Input: a .json file path, all OR 'objects' selector, half_gauge (float), max_curvatre (float), objects (string(s))
    Output: standard pythonic Roads dictionary
    """
    def compute_label(integer):  # 0 -> "A"; 2 -> "C"; 29 -> "BC"
        highest_power = 1
        label = ""
        while True:
            if integer >= 26 ** highest_power:
                highest_power += 1
                continue

            highest_power -= 1
            break

        for i in range(highest_power + 1):
            letter_num = integer // (26 ** (highest_power - i))
            integer -= (26 ** (highest_power - i)) * letter_num
            label += chr(ord("A") + letter_num)

        return label

    def turns_clockwise(A, B, C):  # determines whether given polyline turns right, left or neither
        A, B, C = A.copy(), B.copy(), C.copy()
        last_k, this_k = get_k(A, B), get_k(B, C)
        alpha = angle(A, B, C)
        if this_k == last_k:
            return None
        elif last_k == "Vertical":  # sharp angle with verticala
            return (this_k < 0) == (alpha < pi / 2)
        elif this_k == "Vertical":
            return (last_k > 0) == (alpha < pi / 2)
        elif last_k > 0:
            return (-1/last_k <= this_k <= last_k) == (alpha > pi / 2)
        elif last_k == 0:
            return this_k < 0 == (alpha > pi / 2)
        else:
            return (last_k <= this_k<= -1/last_k) == (alpha < pi / 2)


    if file_path[-5:] == ".json":
        with open(file_path, "r", encoding="utf-8") as json_file:
            if objects == None:
                json_data = json.load(json_file)
            else:
                json_data = []
                raw_data = json.load(json_file)
                for object_name in objects:
                    json_data.append(raw_data[object_name])

            json_file.close()
            return json_data

    elif file_path[-6:] == ".pycgr":
        """ ================================  HERE STARTS .pycgr PARSER  ============================= """
        with open(file_path, "r", encoding="utf-8") as pycgr_file:
            data = pycgr_file.readlines()[7:]  # skip first 7 lines (version and comments)
            point_count = int(data[0])
            connect_count = int(data[1])
            points = []  # format: {ID : [x, y], ...}
            connects = []  # format: (ID1, ID2)
            max_id = 0
            for i in range(2, point_count + connect_count):
                line = data[i].split(" ")
                if i < (point_count + 2):
                    y, x = utm.from_latlon(float(line[2]), float(line[1]))[:2]
                    points.append((int(line[0]), [x, y]))
                else:
                    connects.append([int(line[0]), int(line[1])])

        # recalibrate points' coordinates to relative distances from most east and most south point
        smallest_x = points[0][1][0]
        smallest_y = points[0][1][1]

        for p in points:
            smallest_x = p[1][0] if p[1][0] < smallest_x else smallest_x
            smallest_y = p[1][1] if p[1][1] < smallest_y else smallest_y

        for p in points:  # recompute coordinates with 20 meters recalibration padding
            p[1][0] -= smallest_x - 20
            p[1][1] -= smallest_y - 20

        # find natural vertices (look for multiple occurances):
        points = {k:v for k, v in points}
        vertices = {}
        for id in range(point_count):
            as_start, as_end = 0, 0  # counts id's occurances
            for (s, e) in connects:
                if s == id:
                    as_start += 1
                elif e == id:
                    as_end += 1
            if as_start + as_end > 2:
                vertices[id] = {"Coordinates": points[id], "Neighbours": []}

        # build edges from natural vertices:
        edges = []
        for i in range(len(vertices)):
            id, xy = list(vertices.items())[i]
            vert_xy = xy["Coordinates"]
            prev_points = []
            next_points = []
            for c in connects:
                if id == c[1]:
                    prev_point = next(p for p in points.items() if p[0] == c[0])
                    prev_points.append(prev_point)
                elif id == c[0]:
                    next_point = next(p for p in points.items() if p[0] == c[1])
                    next_points.append(next_point)

            neighbours = prev_points + next_points
            for point in neighbours:  # list of points just proceeding or just following vertice
                # is the iterated point just preceeding (0) or just following (1) vertice?
                p_i = neighbours.index(point) >= len(prev_points)
                edge = {"Vertices":[id]}
                chain = [(id, vert_xy)]

                # if vert-vert direct line, register it and continue
                if point[0] != id \
                and ([id, point[0]] in connects or [point[0], id] in connects) \
                and point[0] in vertices.keys():
                    if {"Vertices": [id, point[0]]} not in edges and {"Vertices": [point[0], id]} not in edges:
                        if point[0] not in vertices[id]["Neighbours"]:
                            vertices[id]["Neighbours"].append(point[0])

                        if id not in vertices[point[0]]["Neighbours"]:
                            vertices[point[0]]["Neighbours"].append(id)

                        edge["Vertices"].append(point[0])
                        edges.append(edge)
                    continue

                while True:
                    if chain[-1] != point:  # avoid duplicates
                        chain.append(point)

                    xy = point[1]
                    next_id = next((c[p_i] for c in connects if c[1-p_i] == point[0]), False)
                    if not next_id:  # bridge ends on unregistered (or dead-end) vert
                        dead_end = next(p[0] for p in points.items() if p[1] == xy)
                        edge["Vertices"].append(dead_end)
                        if dead_end not in vertices.keys():
                            vertices[dead_end] = {"Coordinates": xy, "Neighbours": [id]}
                        elif id not in vertices[dead_end]["Neighbours"]:
                            vertices[dead_end]["Neighbours"].append(id)

                        if dead_end not in vertices[id]["Neighbours"]:
                            vertices[id]["Neighbours"].append(dead_end)

                        if len(chain) > 2:
                            edge["Shape"] = [c[1] for c in chain]

                        is_duplicate = next(
                            (e for e in edges if edge["Vertices"] in [e["Vertices"], e["Vertices"][::-1]]),
                            False
                        )
                        if not is_duplicate:
                            edges.append(edge)

                        break

                    point = (next_id, points[next_id])
                    if point[0] in vertices.keys():  # another vertice reached
                        if chain[-1] != point:  # avoid duplicates
                            chain.append(point)

                        edge = {"Vertices":[id, point[0]], "Shape": [c[1] for c in chain]}
                        is_duplicate = next(
                            (e for e in edges if edge["Vertices"] in [e["Vertices"], e["Vertices"][::-1]]),
                            False
                        )
                        if not is_duplicate:
                            edges.append(edge)

                        # register new Neighbour:
                        if id not in vertices[next_id]["Neighbours"]:
                            vertices[next_id]["Neighbours"].append(id)

                        if next_id not in vertices[id]["Neighbours"]:
                            vertices[id]["Neighbours"].append(next_id)

                        break

        # insert wrongly omitted direct edges:
        for label, vertice in vertices.items():
            for neighbour in vertice["Neighbours"]:
                is_included = next((e for e in edges if e["Vertices"] in [[label, neighbour], [neighbour, label]]),
                    False
                )
                if not is_included:
                    edges.append({"Vertices": [neighbour, label]})


        # remove point duplicates from Shapes:
        for edge in edges:
            if "Shape" in edge.keys():
                i = 1
                while i < len(edge["Shape"]):
                    if edge["Shape"][i-1] == edge["Shape"][i]:
                        edge["Shape"].pop(i-1)

                    i += 1

        # Replace numeric IDs with standardized capital labels:
        IDs = []
        max_id = max(list(vertices.keys()))
        for i in range(max_id + 1):
            if i in vertices.keys():
                IDs.append(i)

        # Create a dict of upper-case labels...
        labels = {}
        for i in range(len(IDs)):
            labels[IDs[i]] = compute_label(i)

        # letter-fy vertice labels (including 'Neighbours' labels):
        vertices = [[label, v] for label, v in sorted(vertices.items())]
        for v_pair in vertices:
            v_pair[0] = labels[v_pair[0]]
            if "Neighbours" in v_pair[1]:  # transcript Neighbours list and remove duplicates within it
                v_pair[1]["Neighbours"] = [labels[n] for n in list(dict.fromkeys(v_pair[1]["Neighbours"]))]

        vertices = {label : v for [label, v] in vertices}

        """ ================================  SIMPLIFYING AND CURVING EDGES  ============================= """
        # letter-fy edges and simplify their Shapes:
        gap_tolerance = 1.5 * max_curvature
        total_shapepoints = 0 # TEMP
        clustered_shapepoints = 0 # TEMP
        for edge in edges:
            edge["Vertices"] = [labels[v] for v in edge["Vertices"]]
            if "Shape" in edge.keys():
                shape = edge["Shape"]

                # amass together groups of near points:
                clustered_shape = {tuple(p):False for p in shape}  # format: {(xy) : bool}, bool indicates whether point part of cluster
                total_shapepoints += len(shape)
                for i in range(1, len(shape)):
                    if distance(*shape[i-1:i+1]) <= gap_tolerance:
                        clustered_shape[tuple(shape[i-1])] = True
                        clustered_shape[tuple(shape[i])] = True

                # abolish clusters of 3 and more, where border points aren't wider than gap tolerance apart:
                i = 0
                cluster_start = False
                while i < len(shape):
                    if clustered_shape[tuple(shape[i])] and not cluster_start:
                        cluster_start = shape[i]
                        i += 1  # omit just following point

                    if cluster_start and (not clustered_shape[tuple(shape[i])] or i == len(shape) - 1):  # cluster of 3 and more ended
                        if distance(cluster_start, shape[i-1]) <= gap_tolerance:
                            for _ in range(shape.index(cluster_start) + 1, i-1):
                                shape.pop(shape.index(cluster_start) + 1)
                                i -= 1

                        cluster_start = False
                    i += 1

                # abolish clusters of 2 close midpoints (supplement them with center of their abiscissa):
                i = 3
                while i < len(shape):
                    foursome = shape[i-3:i+1]
                    if [clustered_shape[tuple(p)] for p in foursome] == [False, True, True, False]:
                        pair = foursome[1:3]
                        center = [
                            (pair[0][0] + pair[1][0]) / 2,
                            (pair[0][1] + pair[1][1]) / 2
                        ]
                        clustered_shape[tuple(center)] = False  # create a record for supplemented point
                        shape = shape[:i-2] + shape[i:]
                        shape.insert(i-2, center)
                    i += 1

                # before turning clusters into curves, filter out Shapes completely reduced to direct line:
                if len(shape) <= 2:
                    edge.pop("Shape")
                    continue

                # to distinguish one cluster from two directly joint clusters, insert midpoints where appropriate:
                i = 3
                while i < len(shape):
                    pair = [shape[i-2], shape[i-1]]
                    if clustered_shape[tuple(shape[i-3])] and clustered_shape[tuple(shape[i-2])] \
                    and clustered_shape[tuple(shape[i-1])] and clustered_shape[tuple(shape[i])] \
                    and distance(*pair) > (gap_tolerance * 2):
                        center = [
                            (pair[0][0] + pair[1][0]) / 2,
                            (pair[0][1] + pair[1][1]) / 2
                        ]
                        clustered_shape[tuple(center)] = False
                        shape.insert(i-1, center)
                        i += 1

                    i += 1

                # turn groups of near points into curves - in clustered_shape, change the status of True to Left/Right:
                i = 0 if clustered_shape[tuple(shape[2])] else 2
                back_gauge = 0 if clustered_shape[tuple(shape[-3])] else 2
                curve_points = []
                turning_clockwise = None  # stagnant k -> None; rising k -> False; falling k -> True
                angle_sum = 0 # A valid bezier may not have angle sum higher than pi radians
                while i < len(shape) - back_gauge:  # i and gauge ensure skipping isolated tiny border polylines
                    if not curve_points and not clustered_shape[tuple(shape[i])]: # non-cluster point and no curve to end
                        i += 1
                        continue

                    elif not clustered_shape[tuple(shape[i])]:  # curve ended on previous point
                        if len(curve_points) > 2:
                            p0, p2 = curve_points[0], curve_points[-1]
                            p1 = extended_intersection([curve_points[:2], curve_points[-2:]])
                            if p0 in [shape[0], shape[i-len(curve_points)-1][-1]]:  # no front point repetition necessary (first point of Shape, or preceeded by a touching curve)
                                shape = shape[:i-len(curve_points)] + [[p0, p1, p2]] + shape[i-1:]
                                i -= len(curve_points) - 2
                            else:
                                shape = shape[:i-len(curve_points)+1] + [[p0, p1, p2]] + shape[i-1:]
                                i -= len(curve_points) - 3

                        angle_sum = 0
                        curve_points = []

                    elif i == len(shape) - back_gauge - 1:  # Shape ends with a curve point
                        p0, p2 = curve_points[0], shape[i]
                        p1 = extended_intersection([curve_points[:2], [curve_points[-1], shape[i]]])
                        shape = shape[:i-len(curve_points)] + [[p0, p1, p2]]
                        break

                    elif clustered_shape[tuple(shape[i])] and len(curve_points) == 2:  # third cluster point encountered, establish turning
                        curve_points.append(shape[i])
                        angle_sum += angle(*curve_points)
                        turning_clockwise = turns_clockwise(*curve_points)

                    elif clustered_shape[tuple(shape[i])] and len(curve_points) > 2 \
                    and (turns_clockwise(*curve_points[-2:], shape[i]) not in [turning_clockwise, None] \
                    or angle_sum + angle(*curve_points[-2:], shape[i]) >= pi / 2):  # over-third curve point encountered, but changed turning direction or overflown angle
                        p0, p2 = curve_points[0], curve_points[-1]
                        p1 = extended_intersection([curve_points[:2], curve_points[-2:]])
                        if p0 in [shape[0], shape[i-len(curve_points)-1][-1]]:  # no front point repetition necessary (first point of Shape, or preceeded by a touching curve)
                            shape = shape[:i-len(curve_points)] + [[p0, p1, p2]] + shape[i-1:]
                            i -= len(curve_points) - 2
                        else:
                            shape = shape[:i-len(curve_points)+1] + [[p0, p1, p2]] + shape[i-1:]
                            i -= len(curve_points) - 3
                        curve_points = [curve_points[-1], shape[i]]
                        angle_sum = 0

                    elif clustered_shape[tuple(shape[i])] and (len(curve_points) < 2 or (len(curve_points) > 2 \
                    and turns_clockwise(*curve_points[-2:], shape[i]) in [turning_clockwise, None])):  # first, second or turning-conforming, angle-ok cluster point encountered
                        curve_points.append(shape[i])
                        if len(curve_points) > 2:
                            angle_sum += angle(*curve_points[-3:])

                    i += 1

                # abolish non-defining midpoints (inner points laying almost precisely on abiscissas):
                i = 2
                while i < len(shape):  # scan through remaining point triplets:
                    triplet = shape[i-2 : i+1]
                    if isinstance(triplet[0][0], list) or isinstance(triplet[1][0], list) or isinstance(triplet[2][0], list):
                        i += 1
                        continue

                    if includes_point([triplet[0], triplet[2]], triplet[1], half_gauge):
                        shape.pop(i-1)
                        i = 2
                        continue
                    i += 1

                # second direct line filtering out (don't filter out any curves!)
                if len(shape) == 2 and not isinstance(shape[1][0], list) and not isinstance(shape[0][0], list):
                    edge.pop("Shape")
                    continue
                else:
                    edge["Shape"] = shape

        if not objects:
            return {"Vertices": vertices, "Edges": edges}
        elif objects == "Vertices":
            return {"Vertices" : vertices}
        else:
            return {"Edges": edges}
    else:
        print("Path must point to a valid .json or .pycgr file. ~ read_json.get_dict")
        return False

#d = get_dict("../Data/map.pycgr")
#print(d)

def sort_points(labels, az_order = True):
    """
    Sorts a list of letter-indexed (or letter-number-mixed) points, eg. "Neighbours"
    Uses bubble sort, since list is already supposed to be sorted almost perfectly
    NOTE: a-z order changes the order of num postfixes, too (C12, C8, B5, B3, ...)

    Input: array of strings, az_order (bool)
    Output: (sorted) array of strings
    """
    assert any(labels.count(label) > 1 for label in labels) == False, "Duplicate entries! ~ read_json.sort_points()"
    # letters are sorted as numbers, alphabet is a 26-system
    # (A = 0, D = 3, Z = 25, BC = 28 ...)
    ciphers = set([str(num) for num in range(10)])  # chars of numbers 0 - 10
    labels_dict = {}  # label : computed weight

    for label in labels:
        key = ""
        postfix = []
        weight = 0  # "AACD2" will be converted into 1 134 002
        label = list(label)
        reversed_label = label[::-1]
        for glyph in reversed_label:  # strip number postix from label
            if glyph in ciphers:
                weight += int(glyph) * (10 ** reversed_label.index(glyph))
                postfix.append(str(label.pop()))

        for i in range(len(label)):
            letter_value = (ord(label[-(i+1)]) - 65)  # A = 0, Z = 25, starting from back
            letter_order = 26 ** i  # 26-base system
            weight += letter_value * letter_order * 1000  # BD87 = 14 087
        labels_dict[weight] = key.join(label + postfix[::-1])  # unique names -> unique weights

    sorted_keys = sorted(labels_dict, reverse = not az_order)
    sorted_labels = {key : labels_dict[key] for key in sorted_keys}
    return list(sorted_labels.values())

def order_neighbours(intersection_label, vertices, clockwise = True):
    """
    Checks a list of vertices neighbours to an intersection vertice to graph's Edges,
    puts them by their coordinates in a clockwise/counter-clockwise order
    Uses: abstract_geometry.get_k()
    Includes: clockwise_half()

    Input: label of the intersection (str), pythonised Vertices (dict), clockwise (bool)
    Output: sorted array of strings
    """
    assert intersection_label in vertices.keys(), "Vertice not in given vertices ~ read_json.order_neighbours()"
    # clockwise_half():
    #   Orders tuples of points' coordinates clockwise
    #   around control_xy(clockwise => descending slope). VERTICAL POINTS FORBIDDEN!
    #
    #   Input: control_xy, points (list of xy lists), clockwise (bool)
    #   Output:  (list of xy lists)
    def clockwise_half(control_xy, points):
        # InsertSort neighbours by foci radia k, descending
        k_dict = {tuple(p) : get_k(control_xy, p) for p in points}  # format: {xy: k}, created for efficiency in InsertSort
        for point_index in range(1, len(points)):
            point = points[point_index]
            scan_index = point_index - 1
            # move all points with smaller k right by 1
            while scan_index >= 0 and k_dict[tuple(point)] > k_dict[tuple(points[scan_index])]:
                points[scan_index+1] = points[scan_index]
                scan_index -= 1
            points[scan_index+1] = point

        return points

    # order_neighbours begins:
    left_points = []  # points more on the left than intersection_label
    right_points = []
    left_vertical = False
    right_vertical = False
    # isolate only neighbour vertices, format: {(x, y) : label}
    neighbours = {tuple(v["Coordinates"]) : label for label, v in vertices.items() if intersection_label in v["Neighbours"]}
    if len(neighbours) <= 2:
        return list(neighbours.values())  # 1 or 2 neighbours, no clockwise/counter-clockwise distinction, return right away

    xy = vertices[intersection_label]["Coordinates"]
    # generate dict of neighbours' xy (omit ordering vertice)}:
    points = [list(key) for key in neighbours.keys() if key != xy]
    # divide points to left and right of ordering vertice:
    for point in points:
        if point[0] == xy[0]:
            if point[1] > xy[1]:  # point directly above is the most clockwise point of left half
                left_vertical = point
            else:
                right_vertical = point  # point directly below is the most clockwise point of right half
            continue

        elif point[0] > xy[0]:
            right_points.append(point)
        else:
            left_points.append(point)

    # get halves' points (with added vertica points, if these exist)
    left_keys = clockwise_half(xy, left_points) + [left_vertical] if left_vertical else clockwise_half(xy, left_points)
    right_keys = clockwise_half(xy, right_points) + [right_vertical] if right_vertical else clockwise_half(xy, right_points)
    keys = left_keys + right_keys
    ordered_neighbours = [neighbours[tuple(key)] for key in keys]
    return ordered_neighbours if clockwise else ordered_neighbours[::-1]

def evaluate_offsets(edges_list):
    """
    Takes edges of rails graph with "Offsets" on Beziers, turns offset points into new
    control points, finds new top control point (using curve tangents)
    Uses: abstract_geometry.narrow_bezier()

    Input: rails graph with "Offsets" (list of pythonic dicts)
    Output:  "Offsets"-free rails graph (pythonic dict)
    """
    for edge in edges_list:
        if "Shape" in edge.keys():
            for i in range(len(edge["Shape"])):
                if isinstance(edge["Shape"][i][-1], dict):
                    control_points = edge["Shape"][i][:-1]
                    offsets = edge["Shape"][i][-1]["Offsets"]
                    edge["Shape"][i] = narrow_bezier(control_points, offsets)

    return edges_list

def smoothen_rails(rails_dict, mr, step=0.01):
    """
    Takes a roughly doubled smooth graph graph produced by insert_rails, modifies all
    corners into tiny bezier curves of given maximum curvature (mr - minimal radius
    of curve's inner osculating circle).
    Very small shape sections - length of 3 * mr - are skip-smoothened as gaps.
    Function assumes, that two gaps never follow successivelly (mid-shape nor shape/complex vertice)
    A complex vertice of any connector length is also considered a gap.
    Uses: abstract_geometry.distance(), smoothen_lines(), smoothen_combined(), smoothen_beziers()

    Input: a rough rails graph (pythonic dict), mr (float)
    Output: smoothened vehicle graph (pythonic dict)
    """
    # 1 #, smoothen edge shape's inner corners:
    gap_tolerance = 1.5 * mr
    for edge_i in range(len(rails_dict["Edges"])):
        if "Shape" in rails_dict["Edges"][edge_i].keys():
            # divide line into continuous polyline / curve segements:
            segments = []  # structure: ("lines", [points]) OR ("curves", [curves])
            shape = []
            now_adding = False
            for element in rails_dict["Edges"][edge_i]["Shape"]:
                if not now_adding and isinstance(element[0], list):  # first curve
                    now_adding = ("curves", [element])
                    continue
                elif not now_adding:  # first curve
                    now_adding = ("lines", [element])
                    continue

                if isinstance(element[0], list) and now_adding[0] == "curves":  # curve while adding curves
                    now_adding[1].append(element)
                elif isinstance(element[0], list):  # curve while adding lines
                    segments.append(now_adding)
                    now_adding = ("curves", [element])
                elif now_adding[0] == "curves":  # line while adding curves
                    segments.append(now_adding)
                    now_adding = ("lines", [element])
                else:  # line while adding lines
                    now_adding[1].append(element)

            segments.append(now_adding)

            # filter away mid-shape 1-tiny-line segments:
            i = 1
            while i < len(segments):
                if segments[i][0] == "lines" \
                and len(segments[i][1]) == 2 \
                and distance(*segments[i][1][:2]) < gap_tolerance:
                    segments.pop(i)
                    if segments[i-1][0] == segments[i][0]:  # line was between curves
                        segments[i] = list(segments[i])
                        segments[i][1] = segments[i-1][1] + segments[i][1]
                        segments[i] = tuple(segments[i])
                        segments.pop(i-1)
                else:
                    i += 1

            # simplyfy lines with tiny-line borders:
            for segment in segments:
                if segment[0] == "lines":  # smoothening mid-polyline
                    # reshape border lines shorter than gap_tolerance:
                    if distance(*segment[1][-2:]) <= gap_tolerance:
                        segment[1].pop(-2)

                    if distance(*segment[1][:2]) <= gap_tolerance:
                        segment[1].pop(1)

                    assert len(segment[1]) > 1, "Segment mustn't consist just of 2 sub-gap lines. ~ read_json.smoothen_rails()"

            # smoothen in and between segments; keep in mind:
            # - points and CPs are always added to shape as line1 or curve1 (inside switch)
            # - line1 is 1st in segment, or prev_element (prev_element is always smoothened)
            # - combined smoothenings are always done 'forward', Ie. with tne NEXT segment
            # - combined smoothenings add 1st line/curve and smoothener to shape, and alter
            #   the 2nd line/curve in segments list - (1st of next segment) - for later
            now = segments[0][0]
            if isinstance(segments[0][1][0][0], list):  # shape beginning with bezier
                prev_element = segments[0][1][0]
            else:  # shape beginning with poly(line)
                prev_element = segments[0][1][:2]

            for segment in segments:
                if segment[0] == "lines":  # smoothening mid-polyline
                    now = "lines"
                    i = 2
                    prev_element = segment[1][:2]
                    while i < len(segment[1]):
                        line1 = segment[1][i-2:i] if i == 2 else prev_element  # smooth continuity, avoid 're-edging'
                        line2 = segment[1][i-1:i+1]
                        # account for mid-polyline lines shorter than gap_tolerance:
                        if distance(*line2) <= gap_tolerance:
                            line2 = segment[1][i:i+2]
                            i += 1

                        smooth = smoothen_lines(line1, line2, mr)
                        assert smooth, "Error smoothening lines (is network mr-ok?). ~ read_json.smoothen_rails()"
                        shape += smooth["line1"]
                        if smooth["smoothener"]:
                            shape += smooth["smoothener"]
                        prev_element = smooth["line2"]
                        i += 1

                else:  # smoothening mid-polycurve
                    now = "curves"
                    prev_element = segment[1][0]
                    for i in range(1, len(segment[1])):
                        curve1 = segment[1][i-1] if i == 1 else prev_element  # smooth continuity, avoid 're-edging'
                        curve2 = segment[1][i]
                        smooth = smoothen_beziers([curve1], [curve2], step, mr)
                        if not smooth:
                            print(curve1, "\n", curve2)
                        assert smooth, "Error smoothening beziers (is network mr-ok?). ~ read_json.smoothen_rails()"
                        shape += smooth["curve1"]
                        if smooth["smoothener"]:
                            shape += smooth["smoothener"]
                        prev_element = smooth["curve2"][0]

                # smoothen with next segment (meaning also: alter its beginning), if there is one:
                seg_i = segments.index(segment)
                if seg_i != len(segments) - 1:
                    if len(prev_element) == 2:  # line / curve smoothening
                        next_element = segments[seg_i + 1][1][0]
                        smooth = smoothen_combined(prev_element, [next_element], step, mr)
                        assert smooth,  "Error smoothening line/bezier (is network mr-ok?). ~ read_json.smoothen_rails()"
                        shape += smooth["line1"]
                        if smooth["smoothener"]:
                            shape += smooth["smoothener"]
                            segments[seg_i+1][1][0] = smooth["curve2"][0]
                    else:  # curve / line smoothening
                        next_element = segments[seg_i + 1][1][:2]
                        smooth = smoothen_combined(next_element[::-1], [prev_element[::-1]], step, mr)
                        assert smooth,  "Error smoothening bezier/line (is network mr-ok?). ~ read_json.smoothen_rails()"
                        shape += [smooth["curve2"][0][::-1]]
                        if smooth["smoothener"]:
                            shape += [smooth["smoothener"][0][::-1]]
                        segments[seg_i+1][1][0] = smooth["line1"][1]

                else:   # sew already smoothened last element of shape at the end
                    shape += prev_element if len(prev_element) == 2 else [prev_element]

                rails_dict["Edges"][edge_i]["Shape"] = shape

    # 2 #, smoothen vertices:
    for vert_label, vertice in rails_dict["Vertices"].items():
        if " " in vertice["NeighbourVertices"]:  # dead-end vertice, no corner to smoothen
            continue

        # Try to find all edge pairs on Vertices:
        try:
            prior_edge = next(edge for edge in rails_dict["Edges"] if edge["Vertices"][1] == vert_label)
            next_edge = next(edge for edge in rails_dict["Edges"] if edge["Vertices"][0] == vert_label)
            edge_pair = [prior_edge, next_edge]
            edge_indexes = [rails_dict["Edges"].index(prior_edge), rails_dict["Edges"].index(next_edge)]

            # gather border segments:
            elements = []
            for i in range(2):
                if "Shape" not in edge_pair[i].keys():  # edge is a direct line
                    line = []
                    coordinates = [rails_dict["Vertices"][edge_pair[i]["Vertices"][i]]["Coordinates"], vertice["Coordinates"]]
                    for v in coordinates[::1-2*i]:  # reverse if i == 1
                        v_i = coordinates[::1-2*i].index(v)
                        if not isinstance(v[0], list):  # simple point
                            line.append(v)
                        elif not isinstance(v[0][0], list):  # line vertice
                            line.append(v[v_i-1])
                        else:  # curve vertice
                            line.append(v[0][v_i-1])

                    elements.append(line)
                elif not isinstance(edge_pair[i]["Shape"][i-1][0], list):  # edge is a shape bordering on line
                    if i == 0:
                        elements.append(edge_pair[i]["Shape"][-2:])
                    else:
                        elements.append(edge_pair[i]["Shape"][:2])
                else:  # edge is a shape bordering on curve
                    elements.append(edge_pair[i]["Shape"][i-1])

        except:
            raise Exception("Edges are not properly linked through Neighbours ~ read_json.smoothen_rails()")

        # modify vertice and, if need, Shapes:
        if len(elements[0]) == len(elements[1]) == 2:  # line / line
            smooth = smoothen_lines(*elements, mr)
        elif len(elements[0]) < len(elements[1]):  # line / bezier
            smooth = smoothen_combined(elements[0], [elements[1]], step, mr)
        elif len(elements[0]) > len(elements[1]):  # bezier / line
            inverted_smooth = smoothen_combined(elements[1][::-1], [elements[0][::-1]], step, mr)

            assert inverted_smooth, "Curve/line vertice could not be smoothened (is network mr-okay?). ~ read_json.smoothen_rails()"
            if not inverted_smooth["smoothener"]:
                reinverted_smoothener = False
            elif not isinstance(inverted_smooth["smoothener"][0][0], list):
                reinverted_smoothener = inverted_smooth["smoothener"][::-1]
            else:
                reinverted_smoothener = [inverted_smooth["smoothener"][0][::-1]]

            smooth = {
                "curve1": [inverted_smooth["curve2"][0][::-1]],
                "smoothener": reinverted_smoothener,
                "line2": inverted_smooth["line1"][::-1]
            }
        else:  # bezier/bezier
            smooth = smoothen_beziers([elements[0]], [elements[1]], step, mr)

        assert smooth, "Error smoothening vertice (is network mr-okay?) ~ read_json.smoothen_rails()"
        if smooth["smoothener"]:
            rails_dict["Vertices"][vert_label]["Coordinates"] = smooth["smoothener"]
            for i in [0, 1]:
                if "Shape" in edge_pair[i].keys():
                    if isinstance(edge_pair[i]["Shape"][i-1][0], list):  # shape borders on bezier
                        rails_dict["Edges"][edge_indexes[i]]["Shape"][i-1] = smooth["curve" + str(i+1)][0]
                    else:
                        rails_dict["Edges"][edge_indexes[i]]["Shape"][i-1] = smooth["smoothener"][0][-i]
    return rails_dict

def add_crossings(full_dict, mr, drive_right, step = 0.005):  # TODO - fix issues with drive_right
    """
    Takes smoothened rails_dict, inserts inner crossing traces through its crossroads,
    made of either single or double quadratic beziers of given minimal osculating radius (mr).
    Intersection lines are added as new edges with Shapes.
    Uses: abstract_geometry.smoothen_lines(), .smoothen_combined(), .smoothen_beziers(), .biaxial_connector(), .check_curvature(),
          abstarct_geometry.extend_abiscissa(), .includes_point(), .extended_intersecton(), .bezier_intersection()
          .order_neighbours()
    Includes: find_element()

    Input: a smoothened rails graph (pythonic dict), mr (float), drive_right (bool)
    Output: vehicle graph with crossings edges (pythonic dict)
    """
    # find_element():
    #   Given a rail vertice, finds its preceeding/succeeding element (curve or line).
    #   The rail may be origin or target, as determined by last bool parameter. Returns
    #   xys of line or curve
    #
    #   Input: rail ((label:dict) tuple), full_dict (dict), origin (bool)
    #   Output:  element ([xy, xy] or [[xy, xy, xy]])
    def find_element(rail_vert, full_dict, origin):
        origin = int(origin)  # False = 0; True = 1
        element_edge = next(e for e in full_dict["Rails"]["Edges"] if e["Vertices"][origin] == rail_vert[0])
        element = []
        if "Shape" in element_edge.keys():
            if isinstance(element_edge["Shape"][-origin][0], list):
                return element_edge["Shape"][-origin]  # border curve
            else:
                return element_edge["Shape"][-2:] if origin else element_edge["Shape"][:2]  # border line

        rail_xys = rail_vert[1]["Coordinates"]
        other_xys = full_dict["Rails"]["Vertices"][element_edge["Vertices"][origin-1]]["Coordinates"]
        xys = [rail_xys, other_xys][::1-2*origin]
        for i in [0, 1]:
            if not isinstance(xys[i][0], list):
                element.append(xys[i])  # simple point constituting vertice
            elif not isinstance(xys[i][0][0], list):
                element.append(xys[i][i-1])  # line constituting vertice
            else:
                element.append(xys[i][0][i-1])  # curve constituting vertice

        return element

    # add_crossings() STARTS HERE # Tterate through graph's intersections one by one:
    for graph_label, graph_vert in full_dict["Vertices"].items():
        if len(graph_vert["Neighbours"]) < 3:  # dead-end or dead-arm vertice, no inner crossings
            continue

        # Order neighbours for connecting Rails vertices; drive_right -> counter-clockwise. not drive_right -> clockwise
        ordered_neighbours = order_neighbours(graph_label, full_dict["Vertices"], not drive_right)
        for graph_origin in ordered_neighbours:
            # create crossings from origin to targets; shift order_neighbours, so they start with graph_origin
            graph_targets = ordered_neighbours[ordered_neighbours.index(graph_origin):] + ordered_neighbours[:ordered_neighbours.index(graph_origin)]
            origin_railvert = next(vert for vert in full_dict["Rails"]["Vertices"].items() if vert[1]["NeighbourVertices"] == graph_targets[:2][::-1])  # rail vertice
            origin_edge = next(rail for rail in full_dict["Rails"]["Edges"] if rail["Vertices"][1] == origin_railvert[0])

            # establish origin element and preceeding_railvert:
            origin_element = find_element(origin_railvert, full_dict, True)  # line / curve before origin smoothener
            preceeding_railvert = next(r for r in full_dict["Rails"]["Vertices"].items() if r[1]["Neighbours"][0] == origin_railvert[0])

            # establish target elements and following railvert:
            graph_targets = graph_targets[1:]  # omit itself (illegal U-turn) and nearest (already connected by regular smoothener) from targeting
            target_railverts = [
                next(vert for vert in full_dict["Rails"]["Vertices"].items() if vert[1]["NeighbourVertices"] == graph_targets[i:i+2][::-1])
                for i in range(len(graph_targets)-1)  # target vertice progresses from 1st NeighbourVertice to 2nd - another from 2nd to 3rd etc.
            ]
            # create all crossings stemming from original element
            for target_railvert in target_railverts:
                target_element = find_element(target_railvert, full_dict, False)
                following_railvert = next(r for r in full_dict["Rails"]["Vertices"].items() if r[1]["Neighbours"][-1] == target_railvert[0])

                origin_line = origin_element[-2:]  # if bezier, take its control line
                target_line = target_element[:2]
                e1, e2 = origin_line[1], target_line[0]
                V0 = extended_intersection([origin_line, target_line])

                if (includes_point([origin_line[0], target_line[1]], e1)  \
                and includes_point([origin_line[0], target_line[1]], e2)) \
                or includes_point([e1, e2], V0):  # CROSSING IS A DIRECT LINE
                    crossing = [e1, e2]
                elif not V0 or not check_curvature(mr, e1, V0, e2):  # Try biaxial connector, cutting elements with conventional smootheners - AFTER EXTENDING TO INTERSECTION (too long gap is not disqualifying here)...
                    smooth = biaxial_connector(origin_line, target_line, mr)                                                               # - In that case, ensure SMOOTHENER EXTENDS BACK - this might make crossing [curve, line]
                    if smooth:  # CROSSING IS A BIAXIAL CONNECTOR
                        crossing = [smooth["smoothener1"], smooth["smoothener2"]]
                    else:  # Try smoothening: (ensure gab is bridged by employing extenders):
                        if len(origin_element) == len(target_element) == 2:  # line / line
                            smooth = smoothen_lines([origin_element[0], V0], [V0, target_element[1]], mr)
                        elif len(origin_element) < len(target_element):  # line / curve
                            # extend line (to ensure gap is bridged):
                            origin_line = extend_abiscissa(origin_element, 100)
                            smooth = smoothen_combined(origin_line, [target_element], step, mr)
                        elif len(origin_element) > len(target_element):  # curve / line
                            # extend line (to ensure gap is bridged):
                            target_line = extend_abiscissa(target_element, 100)
                            smooth = smoothen_combined(target_line[::-1], [origin_element[::-1]], step, mr)
                            assert smooth, "Error crossing with cutting combined smoothener (is network mr-ok?). ~ read_json.add_crossings()"
                            if smooth["smoothener"] and isinstance(smooth["smoothener"][0][0], list):  # re-reverse bezier smoothener
                                smooth["smoothener"] = [smooth["smoothener"][0][::-1]]
                            elif smooth["smoothener"]:  # re-reverse line smoothener
                                smooth["smoothener"] = smooth["smoothener"][::-1]
                            # (... no need re-reversing False smoothener already excluded by 'CROSSING A DIRECT LINE' clause)
                        else:  # curve / curve
                            smooth = smoothen_beziers([origin_element], [target_element], step, mr)

                        # ensure smoothener reaches e1 and e2 (= ensure gap was bridged):
                        assert smooth, "Error crossing with any cutting smoothener (is network mr-ok?). ~ read_json.add_crossings()"
                        assert smooth["smoothener"], "A valid False-smoothener would've been already excluded. ~ read_json.add_crossings()"
                        smoothener = smooth["smoothener"]
                        s_borders = smoothener if len(smoothener) == 2 else [smoothener[0][0], smoothener[0][-1]]
                        back_extenders = []
                        elements = [origin_element, target_element]
                        for i in [0, 1]:
                            if len(elements[i]) == 3:  # checking if curve bridged
                                if bezier_intersection([elements[i]], s_borders[i]):
                                    back_extenders.append(False)
                                else:
                                    back_extenders.append([elements[i][i-1], s_borders[i]][::1-2*i])
                            else:  # checking if line bridged
                                if includes_point(elements[i], s_borders[i], 0.01):  # ensure total smootheness - higher precision
                                    back_extenders.append(False)
                                else:
                                    back_extenders.append([elements[i][i-1], s_borders[i]][::1-2*i])

                        for ext in back_extenders:
                            if ext and len(smoothener) == 2:  # simply prolong smoothener line
                                smoothener[i] = ext
                            elif ext and back_extenders.index(ext) == 0:  # bridge with line before curve
                                smoothener = ext + smoothener
                            elif ext:  # bridge with line after curve
                                smoothener += ext

                        crossing = smoothener
                else:  # V0 found eazily as crossing of elements
                    crossing = [[e1, V0, e2]]  # CROSSING IS A SIMPLE BEZIER

                edge = {
                    "Vertices": [origin_railvert[0], target_railvert[0]],
                    "From": preceeding_railvert[0], "To": following_railvert[0],
                    "Shape": crossing
                }
                full_dict["Rails"]["Edges"].append(edge)

    return full_dict
