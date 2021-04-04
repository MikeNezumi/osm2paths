from math import sqrt, sin, cos, asin, acos, pi
from pythematics import polynomials

def distance(A, B):
    """
    Takes coordinates of two points, returns their distance

    Input: A (xy list), B (xy list)
    Output: distance (float)
    """
    A = A.copy()
    B = B.copy()
    return sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)  # Thanks, Pythagoras!

def angle(A, V, B):
    """
    Takes in three points - A for arm, V for tip, B for 2nd arm
    returns smaller angle an V (in radians)
    Uses: .distance()

    Input: A (xy list), V (xy list), B (xy list)
    Output: alpha (float)
    """
    A, V, B = A.copy(), V.copy(), B.copy()
    A_vector = [A[0] - V[0], A[1] - V[1]]
    B_vector = [B[0] - V[0], B[1] - V[1]]
    dot_product = A_vector[0] * B_vector[0] + A_vector[1] * B_vector[1]
    a = distance(A, V)
    b = distance(B, V)
    acos_arg = dot_product / (a * b)
    acos_arg = -1 if acos_arg < -1 else acos_arg  # fix underflowting errors
    acos_arg = 1 if acos_arg > 1 else acos_arg  # fix overfloating errors
    return acos(acos_arg)  # in radians

def orient_line(vertice_xy, line):
    """
    Checks/corrects order of xy lists, so that the line is directed from vertice
    (the xy that is closer to point is put first, the further one is put second)
    Uses: .distance()

    Input: vertice coordinates (xy list), line (list of xy lists)
    Output: oriented line (list of lists)
    """
    xy1, xy2 = line[0], line[1]
    distance_1 = distance(vertice_xy, xy1)
    distance_2 = distance(vertice_xy, xy2)
    if distance_1 < distance_2:
        return line
    else:
        return line[::-1]

def get_k(xy1, xy2, perpendicular = False):
    """
    Establishes steepness of line, or its perpendicular - as if line were a Linear
    function graph, and we returned the k value from y = kx. Vertical line -> "Vertical"

    Input: 2 points (xy, xy), perpendicular (bool)
    Output: k of line OR k of perpendicular line (float OR str: "Vertical")
    """
    if xy1[0] > xy2[0]:  # left-right line convention
        xy1, xy2 = xy2.copy(), xy1.copy()
    else:
        xy1, xy2 = xy1.copy(), xy2.copy()

    if xy1[0] == xy2[0]:
        return 0 if perpendicular else "Vertical"
    k = (xy2[1] - xy1[1]) / (xy2[0] - xy1[0])
    if k == 0:
        return "Vertical" if perpendicular else k
    return (-1) / k if perpendicular else k

def extend_abiscissa(line, x_prolongment):
    """
    Prolongs given abiscissa on both ends by given x_prolongmnent (line's border
    points will shift, so that their x-shift equals x_prolongment (if not vertical),
    and their y_shift equals k * x_prolongment)
    Uses: .get_k()

    Input: line ([xy, xy]), x_prolongment (int/float)
    Output: extended_line ([xy, xy])
    """
    line = [xy.copy() for xy in line]  # keep function pure
    line_k = get_k(*line)
    if line_k == "Vertical" and line[0][1] > line[1][1]:  # up-down line
        return [
            [line[0][0], line[0][1] + x_prolongment],
            line[1]
        ]
    elif line_k == "Vertical" and line[1][1] > line[0][1]:  # down-up line
        return [
            [line[0][0], gap[0][1] - x_prolongment],
            line[1]
        ]
    elif line[0][0] < line[1][0]:  # left-right line
        return [
            [line[0][0] - x_prolongment, line[0][1] - x_prolongment * line_k],
            line[1]
        ]
    else:  # right-left line
        return [
            [line[0][0] + x_prolongment, line[0][1] + x_prolongment * line_k],
            line[1]
        ]

def includes_point(line, point, precision = 0.2):  # default - 0.2 meters precision
    """
    Takes a line, checks if point is on that line or not, with given float precision
    (precision determines the width of a 'checking rectangle' - rectangle, whose cental axis is the given line.
    Function then returns True if point lies in this rectangle, or in a precision distance from line's border points
    Uses: .distance(), .get_k()

    Input: line ([xy list, xy list])
    Output: includes or not (bool)
    """
    assert isinstance(line[0], list) and isinstance(line[1], list), "First argument must define a line: [[xy], [xy]] ~ abstract_geometry.includes_point()"
    line = line[::-1] if line[0][0] > line[1][0] else line  # conventional point order: -> left-right
    if (distance(line[0], point) <= precision) \
    or (distance(line[1], point) <= precision):
        return True

    k = get_k(*line)
    if k == "Vertical":  # vertical line
        line = line[::-1] if line[0][1] > line[1][1] else line  # vertical convention: down-up
        horizontal_deviance = abs(point[0] - line[0][0])
        return (horizontal_deviance <= precision) and (line[0][1] <= point[1] <= line[1][1])
    elif k == 0:  # horizontal line
        vertical_deviance = abs(point[1] - line[0][1])
        return (vertical_deviance <= precision) and (line[0][0] <= point[0] <= line[1][0])

    else:  # general line, vars nomenclature taken from notebook TODO
        v0 = point[1] - line[0][1]  # negative if point below line[0]
        H = [line[0][0] + v0 / k, line[0][1] + v0]  # horizontal projection of point on abscissa (or its prolonged line)
        if H[0] == point[0]:  # point projected onto itself (within precision)
            return (line[0][1] <= point[1] <= line[1][1]) or (line[1][1] <= point[1] <= line[0][1])  # True - point on abiscissa, False - point on prolonged line

        a = (point[0] - H[0]) / (1 + k ** 2) # a > 0: point is right of line (a < 0: point left of line)
        K = [H[0] + a, H[1] + a * k]  # perpendicular projection of point on abscissa/prolonged line (foci radius' slope is -1/k)
        # yield True if:
        # point deviance from line is in precision
        # and projected point is on abscissa,
        return (distance(K, point) <= precision) and (line[0][0] <= K[0] <= line[1][0])

def intersection_point(line1, line2, cut_lines = False):
    """
    Takes two lines, chops off their part!, returns EITHER their intersection point
    OR None in case they have no intersection point

    Input: line1 (list of xy-lists), line2 (list of xy-lists), cut lines (bool)
    Output: EITHER dict with point and both lines || FALSE in no-intersection case
            OR intersection xy (list) || False in no-intersection case
    """
    line1 = [xy.copy() for xy in line1]  # protect lines from altering state
    line2 = [xy.copy() for xy in line2]
    # Inital math data mining
    for i in range(2):
        line = [line1, line2][i]
        if line[0][0] > line[1][0]:  # ensure xys are left-to-right
            line[0], line[1] = line[1], line[0]  # swap, if needed
        # to avoid ambiguity, all vertical lines are going up [lower, higher]:
        elif line[0][0] == line[1][0] and line[0][1] > line[1][1]:
            line[0], line[1] = line[1], line[0]  # swap to get proper order

    variables = {
        "line1" : line1,
        "line2" : line2,
        "k1" : 0,
        "k2" : 0
    }

    intersect_check = [False, False]  # two-phase check if they intersect at all
    for i in range(2):
        line = [line1, line2][i]
        x_shift = line[1][0] - line[0][0]
        y_shift = line[1][1] - line[0][1]

        if x_shift == 0:  # vertical line (formula below would divide by 0)
            other_line = [line1, line2][1-i]  # index is 0 for i == 1, and 1 for i == 0
            other_x_shift = other_line[1][0] - other_line[0][0]
            other_y_shift = other_line[1][1] - other_line[0][1]
            if other_x_shift == 0:  # both perfectly vertical, none or infinite intersection points
                return False
            k_other = other_y_shift / other_x_shift
            # eazy check if lines intersect at all:
            if (other_line[0][0] > line[0][0] or other_line[1][0] < line[0][0]) \
            or (not line[0][1] < other_line[0][1] + k_other * (line[0][0] - other_line[0][0]) < line[1][1]):  # both points on 1 side OR other_line not raising/declining fast enough
                return False

            if other_line[0][0] == line[0][0]:
                crossing_y = other_line[0][1]  # lines 'touch' - form a down pointing arrow
            else:
                crossing_y = other_line[0][1] + (line[0][0] - other_line[0][0]) * k_other  # non-vertical line's y value on vertical line's x
            if (line[0][1] <= crossing_y <= line[1][1]) or (line[1][1] <= crossing_y <= line[0][1]):  # line is crossed by verticala
                if cut_lines:
                    return {"point": [line[0][0], crossing_y], "line1": line1, "line2": line2}
                else:
                    return [line[0][0], crossing_y]
            else:
                return False

        k = y_shift / x_shift  # establish steepness (k as in: f(x) = kx + q)
        variables["k" + str(i+1)] = k

        # check whether lines intersect at all (vertical lines already returned :)
        # where does line cross y-axis?
        q_abs = line[0][1] - k * line[0][0]  # q_abs = y0 - k * x0 (q-absoulute)
        other_line = [line1, line2][1-i]  # index is 0 for i == 1, and 1 for i == 0
        other_x_shift = other_line[1][0] - other_line[0][0]
        if other_x_shift == 0: # next line is vertical, address in next iteration
            continue
        over = [False, False]  # are points over line? (Crossing = [True, False] or [False, True])
        for j in range(2):
            other_x, other_y = other_line[j]
            line_y = k * other_x + q_abs
            if line_y == other_y:  # lines touch (parallel edge case covered elsewhere)
                intersect_check[i] = True
                break
            over[j] = other_y > line_y
        if over[0] is not over[1]:  # One point above, one below, line crosses
            intersect_check[i] = True
    if not (intersect_check[0] and intersect_check[1]):  # lines do not intersect
        return False

    k1, k2 = variables["k1"], variables["k2"]
    if k1 == k2:  # lines run parallel, no (single) intersection
        return False
    del variables  # obsolete
    if line1[0][0] > line2[0][0]:  # ensure line1 starts further left
        line1, line2 = line2, line1
        k1, k2 = k2, k1
    # chop off the beginning of x-axis where line1 is alone (shift line1's starting point):
    start_x = line2[0][0]
    line1_y_shift = (start_x - line1[0][0]) * k1
    line1[0] = [start_x, (line1_y_shift + line1[0][1])]  # line1's beginning chopped off
    # q1 = 0, calculate q2 (may be negative)
    q2 = line2[0][1] - line1[0][1]
    rel_x = q2 / (k1 - k2)  # length between start_x and intersection point
    common_x = start_x + rel_x
    common_y = line1[0][1] + rel_x * k1
    if cut_lines:
        return {"point": [common_x, common_y], "line1" : line1, "line2": line2}
    else:
        return [common_x, common_y]

def extended_intersection(variables):
    """
    Takes 1 list of two lines [[xy, xy], [xy, xy]] OR prepared variables dict in format:
    {"l_copy1": [[13,50], [39,42]], "x_shift1": 26, "y_shift1": -8, "k1": -0.3076923076923077
    "l_copy2": [[40,10], [-2,60]], "x_shift2": -42, "y_shift2": 50, "k2": -1.1904761904761905},
    extends them, until an intersection point is reached, returns intersection point or False
    Uses: .intersection_point()

    Input: variables (dict or list of list of xy lists)
    Output: intersection point (xy_list) OR False (bool)
    """
    if not isinstance(variables, dict):
        line1 = [xy.copy() for xy in variables[0]]
        line2 = [xy.copy() for xy in variables[1]]
        lines = [line1, line2]
        A = line1[0]
        B = line2[1]
        variables = {}
        for i in range(2):
            variables["l_copy" + str(i+1)] = lines[i].copy()
            variables["x_shift" + str(i+1)] = lines[i][1][0] - lines[i][0][0]
            variables["y_shift" + str(i+1)] = lines[i][1][1] - lines[i][0][1]
            variables["k" + str(i+1)] = get_k(*lines[i])

    # keep function pure (protect lists):
    variables["l_copy1"] = [xy.copy() for xy in variables["l_copy1"]]
    variables["l_copy2"] = [xy.copy() for xy in variables["l_copy2"]]

    # conventional line order: left-right
    if variables["l_copy1"][0][0] > variables["l_copy1"][1][0]:
         variables["l_copy1"] = variables["l_copy1"][::-1]
         variables["x_shift1"] = -variables["x_shift1"]

    if variables["l_copy2"][0][0] > variables["l_copy2"][1][0]:
         variables["l_copy2"] = variables["l_copy2"][::-1]
         variables["x_shift2"] = -variables["x_shift2"]

    # keep extending both lines in both direction by 1 km untill they cross
    for _ in range(1000):  # Try this thousand times, then break out
        if intersection_point(variables["l_copy1"].copy(), variables["l_copy2"].copy()) != False:  # Intersection not found yet, keep extending
            return intersection_point(variables["l_copy1"], variables["l_copy2"])

        if variables["x_shift1"] == 0:  # for l_copy1, extend only y coordinate
            # line1 (skip x1, x2. No k1, cuz line is vertical)
            variables["l_copy1"][0][1] -= 1000 if variables["y_shift1"] > 0 else -1000  # line1: y1
            variables["l_copy1"][1][1] += 1000 if variables["y_shift1"] > 0 else -1000  # line1: y2
            # line2:
            variables["l_copy2"][0][0] -= 1000 if variables["x_shift2"] > 0 else -1000  # line2: x1
            variables["l_copy2"][1][0] += 1000 if variables["x_shift2"] > 0 else -1000  # line2: x2
            variables["l_copy2"][0][1] -= 1000 * variables["k2"]  # line2: y1
            variables["l_copy2"][1][1] += 1000 * variables["k2"]  # line2: y2
            # (...multiplying by k accounts for up-down conditions)
        elif variables["x_shift2"] == 0:   # for l_copy2, extend only y coordinate
            # line1:
            variables["l_copy1"][0][0] -= 1000 if variables["x_shift1"] > 0 else -1000  # line1: x1
            variables["l_copy1"][1][0] += 1000 if variables["x_shift1"] > 0 else -1000  # line1: x2
            variables["l_copy1"][0][1] -= 1000 * variables["k1"]  # line1: y1
            variables["l_copy1"][1][1] += 1000 * variables["k1"]  # line1: y2
            # line2 (skip x1, x2. No k2, cuz line is vertical)
            variables["l_copy2"][0][1] -= 1000 if variables["y_shift2"] > 0 else -1000  # line2: y1
            variables["l_copy2"][1][1] += 1000 if variables["y_shift2"] > 0 else -1000  # line2: y2
        else:
            for i in range(2):
                variables["l_copy" + str(i+1)][0][0] -= 1000 if variables["x_shift" + str(i+1)] > 0 else -1000  # line1: x1
                variables["l_copy" + str(i+1)][1][0] += 1000 if variables["x_shift" + str(i+1)] > 0 else -1000  # line1: x2
                # k valid for left-to-right lines...
                variables["l_copy" + str(i+1)][0][1] -= 1000 * variables["k" + str(i+1)]
                variables["l_copy" + str(i+1)][1][1] += 1000 * variables["k" + str(i+1)]   # TODO: simplify extending with .extend_abiscissa()

def approx_bezier(control_points, points, step = 0.005, offsets = [False, False]):
    """
    Evaluates many points of a bezier curve defined by input control_points - by
    convention, first and last control point are the curve's beginning and end.
    Using recursion, we reduce everything to linear interpolation
    Uses: ./approx_bezier(), ./includes_point()
    Includes: interpolate()
    Uses: .approx_bezier(), .includes_point()

    Input: control_points - absolute xy coordinates (list of lists)
           (points) - used for recursion, must be set to [] to start
           offsets - curve beginning and end chop-off points ([[xy],[xy]] list)
    OPTIONAL: step - smothness of interpolation (float <0;1>)

    Output: array of points on bezier curve
    """
    def interpolate(xy1, xy2, t):  # In: border XY's, step Out: 1 interpolated point
        x = xy1[0] * (1-t) + xy2[0] * t
        y =  xy1[1] * (1-t) + xy2[1] * t
        if t > 1:  # Step overfloat fix
            return xy2
        return [x, y]

    if points == []:  # calculate first points
        prev_point = control_points[0]
        for control_point in control_points[1:]:
            t = 0
            xy_set = []
            while t <= 1:  # Linear interpolation (gaining one set of points)
                xy = interpolate(prev_point, control_point, t)
                xy_set.append(xy)
                t += step
            points.append(xy_set)
            prev_point = control_point
        if len(points) == 1:  # EVERY TIME 2 CONTROL POINTS ARE GIVEN
            return points[0]
        return approx_bezier(control_points, points, step, offsets)

    new_points = []
    prev_set = points[0]
    for current_set in points[1:]:  # equivalent to the method above
        xy_set = []
        for xy_index in range(len(points[0])):
            xy = interpolate(prev_set[xy_index], current_set[xy_index], step*xy_index)
            xy_set.append(xy)
        new_points.append(xy_set)
        prev_set = current_set

    if len(new_points) == 1:   # reduction complete, slice points by Offsets and return. RECURSION ENDED
        new_points[0].append(control_points[-1])  # add unused last point
        if offsets[0] == False and offsets[1] == False:  # no offsets, return points as-are
            return new_points[0]

        bounds = [0, len(new_points[0])]
        for i in range(len(new_points[0]) - 2):
            xy1, xy2 = new_points[0][i], new_points[0][i+1]
            for offset in offsets:
                if not offset:
                    continue
                # Check if point on line (or, technically, at least in its tolerance bubble)
                if includes_point([xy1, xy2], offset):
                    new_points[0][i + offsets.index(offset)] = offset  # substitute point
                    bounds[offsets.index(offset)] = i + offsets.index(offset) # we want xy2 for end bound

        return new_points[0][bounds[0]:bounds[1]+1]

    return approx_bezier(control_points, new_points, step, offsets)

def narrow_bezier(control_points, offsets, step = 0.005):
    """
    Takes in bezier curve (its control points), delimited by offsets.
    Returns its new narrow (section-curve between offsets)
    Uses: .approx_bezier, .extended_intersection

    Input: control_points (xy, xy, xy), offsets (xy OR False, xy OR False), step (float)
    Output: control_points (xy, xy, xy)
    """
    control_points = [xy.copy() for xy in control_points]
    # exclude out-of-curve Offset points
    for i in [0, 1]:
        if not offsets[i]:
            continue
        if not bezier_intersection([control_points], offsets[i], step):
            offsets[i] = False

    if offsets == False or (offsets[0] == offsets[1] == False):  # empty offsets, no narrowing
        return control_points

    points = approx_bezier(control_points, [], step, offsets)
    # remove all identical points at the end:
    last_point = points[-1]
    points_count = len(points)
    i = 2
    while i < points_count:
        if points[-i] == last_point:
            points.pop(-i)
        else:
            break

    lines = [points[:2], points[-2:]]
    p1 = extended_intersection(lines)  # not very computation-efficient, TODO: change to linear scan?
    if not p1:
        return control_points

    p0 = offsets[0] if offsets[0] else control_points[0]
    p2 = offsets[1] if offsets[1] else control_points[2]
    return [p0, p1, p2]

def bezier_intersection(bezier_points, checked_points, step=0.005):  # As it stands, horribly inefficient
    """
    Returns intersection point of a bezier curve (defined by its control points and step),
    and another object - a point[xy], a line [[x,y], [x,y]], or another bezier
    [[[x,y], [x,y], [x,y]]]. In case of multiple intersections, only the first on
    bezier (in direction of 1st arg) is returned.
    Uses: .distance(), .approx_bezier(), .intersection_point(), .includes_point()

    Input: bezier_points (list-list of xy lists),
           checked_points (point: [x, y] OR line: [xy, xy] OR bezier: [[xy lists]])
           step (float)
    Output: intersection_point (xy list) OR False
    """
    # protect inputs - keep function pure:
    if isinstance(bezier_points[0][-1], dict):
        offsets = bezier_points[0][-1]["Offsets"]
        bezier_points = [[xy.copy() for xy in bezier_points[0][:-1]]]
    else:
        offsets = [False, False]
        bezier_points = [[xy.copy() for xy in bezier_points[0]]]

    # initial math mining:
    point = False
    bezier_points = approx_bezier(*bezier_points, [], step, offsets)
    jump = int(len(bezier_points) // (1 / (step * 10))) # eg. step=0.005: 20 + 1 checked squares per curve (1 for last len % 20 points)
    # tile's height is length of 1st subline * jump - 1st checked tile will be double the necessary size (sublines vary in length, we need the leeway)
    half_tile = distance(*bezier_points[:2]) * jump # Thanks, Pythagoras
    try:
        checking_bezier = type(checked_points[0][0][0])  # is second arg point/line or bezier?
    except TypeError:
        checking_bezier = False
        for i in range(len(bezier_points)-1):
            xy1, xy2 = bezier_points[i], bezier_points[i+1]
            if xy1[0] > xy2[0]:  # order coordinates left - right
                xy1, xy2 = xy2, xy1

            if isinstance(checked_points[0], list):  # checked_points is a line
                point = intersection_point([xy1, xy2], checked_points)
                if point:  # any xy list is not False
                    return point
            else:  # checked_points is 1 point
                # is checked point in line (or, precisely, in its parallelogram)?
                if includes_point([xy1, xy2], checked_points):  # default precision
                    return checked_points

    if checking_bezier:  # Notebook question TODO: given 2 curves, how to find intersection efficiently?
        alt_points = approx_bezier(*checked_points, [], step) # second bezier curve xys
        near_misses = []  # list of (bez_index, alt_index) tuples
        i = 0
        while True:
            if i > len(bezier_points) - 1:  # list would be overflown on next iteration
                break
            for [alt_x, alt_y] in alt_points:  # 'scan' alt bezier's to see whether any points appear in checked tile
                if bezier_points[i][0] - half_tile < alt_x < bezier_points[i][0] + half_tile \
                and bezier_points[i][1] - half_tile < alt_y < bezier_points[i][1] + half_tile:  # is alt point in tile?
                    near_misses.append((i, alt_points.index([alt_x, alt_y])))
                    break
            i += jump
        # check last part of bezier
        if bezier_points[-1][0] - half_tile < alt_x < bezier_points[-1][0] + half_tile \
        and bezier_points[-1][1] - half_tile < alt_y < bezier_points[-1][1] + half_tile:  # is alt point in tile?
            near_misses.append((len(bezier_points) - 1, (alt_points.index([alt_x, alt_y]))))
        for (index, alt_index) in near_misses:
            for i in range(-jump*10, jump*10):  # smaller jump would've been faster, but would break on uneasonably narrow angles!
                if (0 <= index+i <= len(bezier_points)-2) == False:  # check for index overflow, len-2 because we always need line, see 2 lines below
                    continue
                bez_line = [bezier_points[index+i], bezier_points[index+i+1]]
                for j in range(-jump*3, jump*3):
                    if (0 <= index+j <= len(alt_points)-2) == False:  # check for index overflow, len-2 because we always need line, see 2 lines below
                        continue
                    alt_line = [alt_points[alt_index+j], alt_points[alt_index+j+1]]
                    point = intersection_point(bez_line, alt_line)
                    if point:
                        return point

    return point  # if point = False, intersection non-existent

def beziers_crossing(rail1, rail2):
    """
    Takes two assumingly crossing rails (step 2 of write_json.insert_rails()), if they
    really do intersect, returns intersecting point and index(es) of intersecting subbezier(s)
    OR False, if no intersection is found.
    OFFSETS ARE FORBIDDEN, as they may get overwritten by bezier_intersection()
    Uses: .bezier_intersection()

    Input: rail1 - beziers, rail2 - beziers / polyline / point
    Output: tuple: (
                vertice_point (xy list) OR False,
                crossed_index (int) 0 IF False
            )
    """
    if not isinstance(rail2[0], list):  # second rail is point
        for subbez in rail1:
            crossing_point = bezier_intersection([subbez], rail2)
            if crossing_point:
                return (crossing_point, rail1.index(subbez))
    elif not isinstance(rail2[0][0], list):  # second rail is line
        for subbez in rail1:
            crossing_point = bezier_intersection([subbez], rail2)
            if crossing_point:
                return (crossing_point, rail1.index(subbez))
    else:  # second rail is bezier
        for rail2_subbez in rail2:
            for subbez in rail1:
                crossing_point = bezier_intersection([subbez], rail2_subbez)
                if crossing_point:
                    return (crossing_point, rail1.index(subbez))

    return (False, 0)  # no crossing found

def shifted_lines(coordinates, half_gauge, right):
    """
    shifted_lines():
    Takes a series of points and shifts them into right or left lines
    Uses: distance()

    Input: coordinates - absolute xy coordinates (list of lists)
           half_gauge - space between old and new polyline (float)
           right - determines whether to return right or left line copy (bool)

    Output: bare, sometimes non-crossing lines (list of xy - xy list lists)
    """
    points = coordinates.copy()
    assert int(right) in [0, 1], "Input 'right' as True or False (1 or 0)! ~ abstract_geometry.shifted_lines()"
    lines = []
    for i in range(len(points) - 1):
        # simple line:
        xy1 = points[i]
        xy2 = points[i + 1]
        segment_length = distance(xy1, xy2)
        k = half_gauge / segment_length
        x_diff = abs(xy2[0] - xy1[0])
        y_diff = abs(xy2[1] - xy1[1])
        x_shift = k * y_diff
        y_shift = k * x_diff

        if xy1[1] > xy2[1]:  # right rail going under x-axis => decrease x
            x_shift *= -1
        if xy1[0] < xy2[0]:  # right rail going right of y-axis => decrease y
            y_shift *= -1

        if right:  # we want right rail
            right_line = [   # shifts are calibrated for right rail
                [xy1[0] + x_shift, xy1[1] + y_shift],
                [xy2[0] + x_shift, xy2[1] + y_shift]
            ]
            lines.append(right_line)
        else:  # we want left rail
            left_line = [  # left rail, so flip shifts
                [xy1[0] - x_shift, xy1[1] - y_shift],
                [xy2[0] - x_shift, xy2[1] - y_shift]
            ]
            lines.append(left_line)
    return lines

def offset_polyline(coordinates, half_gauge, right):
    """
    Doubles a complex graph edge into its right or left rail, respectively.
    NOTE: function strictly doesn't deal with Bezier curves (offset_bezier()
    handles that).
    Uses: .intersection_point(), .shifted_lines()

    Input: coordinates - absolute xy coordinates (list of lists)
           half_gauge - space between old and new polyline (float)
           right - determines whether to return right or left line copy (bool)

    Output: EITHER right polyline's xys (list of lists) OR left polyline's xys (list of lists)
    """
    points = coordinates.copy()
    assert int(right) in [0, 1], "Input 'right' as True or False (1 or 0)! ~ abstract_geometry.offset_polyline()"
    lines = shifted_lines(coordinates, half_gauge, right)
    points = [lines[0][0]]
    for i in range(len(lines) - 1):
        new_point = intersection_point(lines[i].copy(), lines[i+1].copy())
        if new_point == False:  # rails don't cross, append short line as in rail vertices
            points.append(lines[i][1])
            points.append(lines[i+1][0])
        else:
            points.append(new_point)
    points.append(lines[-1][1])  # append end point
    return points

def triangle_tangent(triangle):
    """
    Takes section of curve defined by 3 points, finds tangent to the second point
    (This tangent is defined by its 2 points - triangle[1] and triangle[2]')

    Input: triangle [[xy], [xy], [xy]]
    Output: tangent [[xy], [xy]]
    """
    assert len(triangle) == 3, "Input must be 3 points (finding tangent impossible). ~ abstract_geometry.offset_bezier[triangle_tangent()]"
    triangle = [xy.copy() for xy in triangle]  # make a copy of input list to protect it (prevent state change)
    (ac_line, cb_line) = (triangle[:2], triangle[1:])
    ac_x_shift = ac_line[1][0] - ac_line[0][0]
    ac_y_shift = ac_line[1][1] - ac_line[0][1]
    tangent = [
        triangle[1],
        [triangle[2][0] + ac_x_shift, triangle[2][1] + ac_y_shift]
    ]
    return tangent

def offset_bezier(coordinates, half_gauge, right, step = 0.005, split_iterations = 2):
    """
    Takes list of bezier curve control points, returns right or left control_points.
    Implements: http://microbians.com/math/Gabriel_Suchowolski_Quadratic_bezier_offsetting_with_selective_subdivision.pdf
    Uses: .get_k(), .includes_point(), .intersection_point(), .extended_intersection(),
          .approx_bezier(), .triangle_tangent(), .distance()
    Includes: offset_subbeziers()[is_left(), near_vertical()]
    Yes, there are double nested functions. It's because these are only valid within the context of offset_bezier()!

    Input: Coordinates - curve control points' xy coordinates (list of lists)
           half_gauge - space between old and new polyline (float)
           left - determines whether to return right or left line copy (bool)
           step - determines smoothness of bezier curve
           split_iterations - sets number of times the curve will get split
    Output: New real points (list of xy-list)
    """
    # offset_subbeziers():
    #   Takes points of a bezier curve (doesn't modify any) and list of control points
    #   finds lines that include these points, and are perpendicular to curve's tangents
    #   Shifts bezier points by half_gauge on these lines by half_gauge, left or right
    #   Includes: get_k(), is_left(), near_vertical()
    #   Uses: .includes_point()
    #
    #   Input: bezier_points of whole quadratic bezier (list of xy lists)
    #          subbeziers - control point lists of all subcurves (list of list of xy lists),
    #          right (bool)
    #          half_gauge (float)
    #   Output: offset_points (list of xy lists)
    def offset_subbeziers(bezier_points, subbeziers, right, half_gauge):

        """ Input: 2 lines' coefficients(float/"Vertical", float/"Vertical") NOTE: k != q """
        """ Output: whether or not q is left of k (bool) """
        def is_left(k, q):
            assert k != q, "k may not equal q - lines would by one ~ abstract_geometry.offset_bezier()[offset_subbeziers()[is_left()]]"
            if k == "Vertical":
                return q < 0  # see Notebook TODO (find sources from board in Android gallery)
            elif k == 0:
                return q > 0;

            if q == "Vertical":
                return k > 0

            if k < 0:
                return (k < q < -(1/k))
            elif k > 0:
                return (not k > q > -(1/k))  # True if q line is right of k line


        """ Input: Trapezoid of 2 neighbour triangles - 4 points list; checked control point ([xy, xy, xy, xy], xy) """
        """ Output: IF point between tangents: (closer base xy, vertical_k) tuple ELSE: False """
        def near_vertical(trapezoid, control_point):
            trapezoid = [xy.copy() for xy in trapezoid]  # 2nd point is A, 3rd point is B
            tangent_A = triangle_tangent(trapezoid[:3])
            tangent_B = triangle_tangent(trapezoid[1:])
            k_A = get_k(*tangent_A, True)
            k_B = get_k(*tangent_B, True)
            q_A = get_k(trapezoid[1], control_point.copy())
            q_B = get_k(trapezoid[2], control_point.copy())

            # If control point lays exactly on vertical, return its base right away:
            if k_A == q_A:
                return (trapezoid[1], k_A)
            elif k_B == q_B:
                return (trapezoid[2], k_B)

            if is_left(k_A, q_A) == is_left(k_B, q_B):  # both left or both right
                return False

            A_distance = distance(control_point, trapezoid[1])
            B_distance = distance(control_point, trapezoid[2])
            if A_distance >= B_distance:
                return (trapezoid[1], k_A)
            else:
                return (trapezoid[2], k_B)


        # avoid modyfying inputs (offset_subbeziers() finally starts here):
        bezier_points = [xy.copy() for xy in bezier_points]
        control_points = []
        for bezier in subbeziers:
            control_points += [bezier[0].copy(), bezier[1].copy()]
        control_points.append(subbeziers[-1][-1])

        k_list = [] # sorted list of verticals' k values.
        # Note: vertical tangent's k (infinity) is represented by the string "Vertical"
        control_index = 0  # index of the control point whose tangent base we're looking for
        closest_distance = False  # used to localise on-curve points
        for point in bezier_points:  # TEMP IF OPTIMIZING, LOOK HERE

            # A, control point is a point of curve:
            if point in [control_points[0], control_points[-1]]:  # first and last point
                if control_points.index(point) == len(control_points) - 1:
                    point_2 = bezier_points[bezier_points.index(point)-1]  # 2nd last bezier point
                else:
                    point_2 = bezier_points[bezier_points.index(point)+1]  # 2nd bezier point

                k_list.append(get_k(point, point_2, True))
                control_index += 1
                continue

            next_point = bezier_points[bezier_points.index(point)+1]  # used for B, and C,last point already excluded
            if next_point in [bezier_points[-1], bezier_points[-2]]:  # skip 2nd and 3rd last point (can't make triangle for tangent)
                continue

            # B, control point is above curve
            elif control_index % 2 == 1:  # scan only when looking for an elevated point
                trapezoid = [  # 2 neighbour triangles - we'll check for point between their tangents
                    bezier_points[bezier_points.index(point)-1].copy(),
                    point.copy(),
                    next_point.copy(),
                    bezier_points[bezier_points.index(point)+2].copy()
                ]
                base = near_vertical(trapezoid, control_points[control_index])  # format ([x, y], vertical's k) - k may be float or "Vertical"
                if base:
                    k_list.append(base[1])
                    control_index += 1
                continue

            # C, CP is point of curve (lays approx. between approximated points) => take that interpolated point which is closest to this CP:
            if not closest_distance:  # scanning for on-curve point beginns, establish closest_distance and continue scan
                closest_distance = distance(control_points[control_index], point)
                continue

            last_triangle = [
                bezier_points[bezier_points.index(point)-2].copy(),
                bezier_points[bezier_points.index(point)-1].copy(),
                point.copy()
            ]
            current_distance = distance(control_points[control_index], point)
            if current_distance < closest_distance:
                closest_distance = current_distance
            else:
                closest_distance = False  # reset closest distance for next-next point
                tangent = triangle_tangent(last_triangle)
                k_list.append(get_k(*tangent, True))
                control_index += 1
                continue

        # Shift control_points by ks (from k_list) to both left and right:
        shifted_points = []  # list of control point copy pairs, in tuples
        points = []  # only points offset to the correct side
        # Establish l vectors for picking the right points (see notebook TODO) - oriented in control_points' direction!
        l_vectors = [[control_points[0], control_points[1]]]  # 1st (shorter) vector
        for i in range(len(control_points) - 2):
            l_vectors.append([control_points[i], control_points[i+2]])
        l_vectors.append([control_points[-2], control_points[-1]])   # last (shorter) vector
        # double and separate control points:
        for i in range(len(control_points)):
            [x, y] = control_points[i]
            if k_list[i] == "Vertical":
                shifted_points.append(([x, y + half_gauge], [x, y - half_gauge]))
            else:
                x_shift = sqrt(half_gauge ** 2 / (k_list[i] ** 2 + 1))
                y_shift = x_shift * k_list[i]
                shifted_points.append(([x + x_shift, y + y_shift], [x - x_shift, y - y_shift]))

            # complex switch to determine which shift is left/right of curve. Layers of switch: k of l_vector; point xys; 'right' func. argument
            # 1st layer: k of l_vector
                # 2nd layer: l orientation
                    # 3rd layer: offset xys --- sublayer: 'right' arg
            if l_vectors[i][0][0] == l_vectors[i][1][0]:  # special case: perfectly vertical vector
                dim_i = 0  # use offset_point's x as a standard
                if l_vectors[i][0][1] < l_vectors[i][1][1]:  # vector going down-up
                    point = shifted_points[i][0] if ((shifted_points[i][0][dim_i] > shifted_points[i][1][dim_i]) == right) else shifted_points[i][1]  # shifted point right of curve if absolutely right
                else:  # vector going up-down
                    point = shifted_points[i][0] if ((shifted_points[i][0][dim_i] < shifted_points[i][1][dim_i]) == right) else shifted_points[i][1]   # shifted point left of curve if absolutely right
            else:
                l_k = (l_vectors[i][1][1] - l_vectors[i][0][1]) / (l_vectors[i][1][0] - l_vectors[i][0][0])
                dim_i = int(-1 < l_k < 1)  # which dimention of offset points to use as a standard - 0: x, 1: y  ...see notebook
                if (l_vectors[i][0][0] > l_vectors[i][1][0]) == (l_k < 1):  # l_vector pointing left with k > 1, or right with k < 1
                    point = shifted_points[i][0] if ((shifted_points[i][0][dim_i] > shifted_points[i][1][dim_i]) == right) else shifted_points[i][1]  # shifted point right of curve if absolutely right
                else:
                    point = shifted_points[i][0] if ((shifted_points[i][0][dim_i] < shifted_points[i][1][dim_i]) == right) else shifted_points[i][1]  # shifted point right of curve if absolutely right
            points.append(point)

        # package shifted points back into subbeziers and return:
        assert len(points) - 1 == 2 ** (split_iterations + 1), "computing subbeziers went wrong (new control points count doesn't add up) ~ abstract_geometry.offset_bezier()[offset_subbeziers()]"
        subbeziers = []
        i = 0
        while i <= len(points) - 3:  # python do-while loop
            subbeziers.append(points[i : i+3])
            i += 2
        return subbeziers


    ################## set initial variables (offset_bezier() finally begins!): #####################
    control_points = [xy.copy() for xy in coordinates]  # protect input array
    # complexly structured array: beziers
    # stores arrays of multiple control point sets (one for each layer of pseudo-recursion), like a perfect binary tree:
    # (example where split_iterations = 3): beziers = [
    #   [[xy, xy, xy]],
    #   [[xy, xy, xy], [xy, xy, xy]],
    #   [[xy, xy, xy], [xy, xy, xy], [xy, xy, xy], [xy, xy, xy]],
    #   [[xy, xy, xy], [xy, xy, xy], [xy, xy, xy], [xy, xy, xy], [xy, xy, xy], [xy, xy, xy], [xy, xy, xy], [xy, xy, xy]]
    # ]
    # ...the function then takes the last list of new beziers and shifts them
    beziers = [[] for _ in range(split_iterations + 1)]  # prepare buffers - one for each layer
    beziers[0].append(control_points)  # iteration layer 0 - original points
    assert int(right) in [0, 1], "Input 'right' as True or False (1 or 0)! ~ abstract_geometry.offset_bezier()"

    # iterate through each layer of binary tree:
    for last_layer in range(split_iterations):  # pseudo-recursive loop for splitting
        for unextended_points in beziers[last_layer]:  # (running 'horizontally' through the binary tree)
            control_points = [xy.copy() for xy in unextended_points]

            # find closest_point (to Pc - 2nd control point):
            bezier_points = approx_bezier(control_points, [], step=step)
            closest_distance = distance(*control_points[:2])
            closest_point = control_points[0]
            for xy in bezier_points:
                CP_distance = distance(control_points[1], xy)
                if closest_distance >= CP_distance:
                    closest_distance = CP_distance
                    closest_point = xy
                else:  # efficiency - by the nature of quadratic beziers, closest point cannot be after distance begins to increase
                    break
            # defining proper tangent (height of triangle formed by 3 consecutive bezier points):
            closest_index = bezier_points.index(closest_point.copy())
            triangle = [bezier_points[closest_index-1], closest_point.copy(), bezier_points[closest_index+1]]
            tangent = triangle_tangent(triangle)

            # prepare for finding tangent-control line intersections:
            lines = [control_points[:2], tangent, control_points[1:]]
            lines = [xy.copy() for xy in lines]
            vars = []
            for line in lines:
                key_index = str(lines.index(line) % 2 + 1)  # 1-2-1; used to create 'zigzag' keys, eg. line1, line2, line1, which is useful for extended_intersection
                line = line[::-1] if line[0][1] > line[1][1] else line  # order vertical lines down to up (next line re-orders non-verticalls by x)
                line = line[::-1] if line[0][0] > line[1][0] else line  # order left to right
                x_shift = line[1][0] - line[0][0]
                y_shift = line[1][1] - line[0][1]
                if x_shift != 0:
                    k = y_shift / x_shift
                else:
                    k = False
                vars.append({
                    "l_copy" + key_index : [line[0].copy(), line[1].copy()],
                    "x_shift" + key_index : x_shift,
                    "y_shift" + key_index : y_shift,
                    "k" + key_index : k
                })
            vars[0].update(vars[1])
            vars[2].update(vars[1])
            Tc1 = extended_intersection(vars[0])  # intersection: control line 1 ; tangent
            vars[2]["l_copy1"][0] = unextended_points[1].copy()  # correct extended central control point
            Tc2 = extended_intersection(vars[2])  # intersection: control line 2 ; tangent
            new_beziers = [
                [unextended_points[0], Tc1, closest_point],
                [closest_point, Tc2, unextended_points[2]]
            ]
            beziers[last_layer+1] += new_beziers  # modyfying is safe, since current layer is always completed before moving on to it from last_layer

    # finally, offset subdivided beziers:
    bezier_points = approx_bezier(coordinates, [], step=step)
    return offset_subbeziers(bezier_points, beziers[-1], right, half_gauge)

def check_curvature(mr, p0, p1, p2):
    """
    Checks that a given bezier curve has minimal or greater osculation circle radius
    Uses: .distance()

    Input: mr (float), p0 (xy), p1 (xy), p2 (xy)
    Output: checked (bool)
    """
    p0, p1, p2 = p0.copy(), p1.copy(), p2.copy()  # protect inputs
    x_length = distance(p0, p1)
    y_length = distance(p1, p2)
    z_length = distance(p0, p2)
    s = (x_length + y_length + z_length) / 2
    area = sqrt(s * (s - x_length) * (s - y_length) * (s - z_length))  # heron's formula for area
    if x_length > z_length / 2 and y_length > z_length / 2:
        tz_length =  sqrt((x_length ** 2 * 2 + y_length ** 2 * 2 - z_length ** 2) / 4)  # triangle median, Apollion's formula
        osc_radius = area ** 2 / tz_length ** 3

    else:
        shorter = x_length if x_length < y_length else y_length
        osc_radius = shorter ** 3 / area

    return osc_radius >= mr

def deduce_p1s(mr, e0, e1, p2):
    """
    Takes incomplete bezier smoothener's control triangle, defined in these values:
        "mr": (float),  # minimrepresentsal radius
        "e0": (xy)  # second point of preceeding line or preceeding bezier tangent (next to e1)
        "e1": (xy),  # point, end of line that the smoothener stems from
        "p2": (xy)  # second point of z,

    deduces all quadratic bezier curves of exactly mr, returns a list of their
    second points (tops of control triangle - p1)
    Required: mr, e0, e1 (=p0), p2

    Other triangle's lines, points and scalars necessary for computation:
        "p1": (xy)  # THE GOAL - 2nd point of curve's control triangle
        "M1": (xy)  # middle point of z, projected onto x line from e1
        "M2": (xy)  # middle point of z, projected onto x line from CP
        "e": (xy, xy)  # original e0-e1 line
        "ex_shift":  # x trend of e (may be negative)
        "ey_shift":  # y trend of e (may be negative)
        "z": (xy, xy),  # base side (not used for interpolation)
        "y": (xy, xy),  # interpolated side, not connected to e1
        "x": (xy, xy),  # interpolated side connected to e1
        "tz_length": (float) # triangle median on side z
        "z_length": (float),
        "y_length": (float),
        "x_length": (float)
    Uses: .distance()

    Input: triangle_vars (dict), compute_minimal (bool)
    Output: False (IF none found; bool) OR p1s (list of xy lists)
    """
    e0, e1, p2 = e0.copy(), e1.copy(), p2.copy()  # protect inputs

    z_length = distance(e1, p2)
    alpha = pi - angle(e0, e1, p2)  # polygon's inner angle betwen z and x
    # 1st equation is valid for x < z_length / 2
    # 2nd equation is valid for x ∈ < z_length / 2; |e1 M2|>
    # (3rd equation is valid for x > |e1 M2|) ; - doesn't always happen, horizon2 can be infinity
    # establish x-horizons between which different curvature equations apply:
    horizon1 = z_length / 2
    if 2 * sin(alpha) > 1:
        horizon2 = False
        horizon3 = False
    else:
        horizon_gama1 = asin(2 * sin(alpha))  # from sine equation
        horizon_gama2 = pi - horizon_gama1
        horizon_beta1 = pi - (alpha + horizon_gama1)
        horizon_beta2 = pi - (alpha + horizon_gama2)
        horizon3 = sqrt(z_length ** 2 * (5/4 - cos(horizon_beta1)))  # from cosine equation, larger than horizon2
        horizon2 = sqrt(z_length ** 2 * (5/4 - cos(horizon_beta2)))  # smaller than horizon 3 (gama larger, beta smaller)

    # having horizons, find all xs between them:
    xs = []
    # 1st equation
    x1 = sqrt((mr * z_length * sin(alpha)) / 2)
    if x1 <= horizon1:
        xs.append(x1)

    # 2nd equation - prepare terms for substitution:
    try:
        a = 3 / 2
        b = (-2) * z_length * cos(alpha)
        c = (5 / 4) * (z_length ** 2)
        d_cubed = (((1 / 4) * z_length ** 2 * sin(alpha) ** 2) / mr) ** 2

        k_6 = a ** 3 / d_cubed
        k_5 = (3 * a ** 2 * b) / d_cubed
        k_4 = (3 * a * (a * c + b ** 2) - d_cubed) / d_cubed
        k_3 = (b * (6 * a * c - b ** 2)) / d_cubed
        k_2 = (3 * c * (a * c + b ** 2)) / d_cubed
        k_1 = (3 * b * c ** 2) / d_cubed
        k_0 = c ** 3 / d_cubed

        x = polynomials.x
        equation = k_6 * x**6 + k_5 * x**5  + k_4 * x**4  + k_3 * x**3  + k_2 * x**2  + k_1 * x**1  + k_0
        imaginary_xs = equation.roots(iterations=100)
        for complex_x in imaginary_xs:
            if not horizon2 and horizon1 <= complex_x.real:
                xs.append(complex_x.real)
            elif (horizon1 <= complex_x.real <= horizon2) or complex_x.real >= horizon3:
                xs.append(complex_x.real)
    except SyntaxError:
        print("abstract_geometry.deduce_p1s: ~ x polynomial failed, pressing on...")

    # possibly 3rd equation - prepare terms for substitution:
    if horizon2:
        try:
            proposed_xs = []

            k_6 = -4 / (mr ** 2 * z_length ** 2 * sin(alpha) ** 2)
            k_3 = (2 * cos(alpha)) / (mr * sin(alpha))
            k_0 = -(z_length ** 2)

            y = polynomials.x
            equation = k_6 * y**6 + k_3 * y**3 + y**2 + k_0
            imaginary_ys = equation.roots(iterations=100)
            for complex_y in imaginary_ys:
                if complex_y.real > 0:
                    maybe_x = (2 * complex_y.real ** 3) / (mr * z_length * sin(alpha))
                    if horizon2 <= maybe_x <= horizon3:
                        xs.append(maybe_x)
        except SyntaxError:
            print("abstract_geometry.deduce_p1s: ~ y polynomial failed, pressing on...")

    if not xs:
        return False

    # remove duplicates:
    i = 0
    while i < len(xs) - 1:
        if xs[i] == xs[i+1]:
            xs.remove(xs[i])
        i += 1

    # arrange xs in ascending order (select sort):
    for i in range(len(xs)):
        # find minimum element in remaining unsorted array:
        min_i = i
        for j in range(i+1, len(xs)):
            if xs[min_i] > xs[j]:
                min_i = j

        xs[i], xs[min_i] = xs[min_i], xs[i]  # swap minimum with first element

    # return a list of second points:
    p1s = []
    ex_shift = e1[0] - e0[0]
    ey_shift = e1[1] - e0[1]
    e = distance(e0, e1)
    for x in xs:
        coefficient = x / e
        p1s.append([
            e1[0] + ex_shift * coefficient,
            e1[1] + ey_shift * coefficient
        ])
    return p1s

def connect_line(point, line, mr):
    """
    Smoothly connects a fixed point to a minimally cut line (minimal cut necessary
    for given minimal radius). Input variables must follow this order:
    point, point-to-cut-of, faraway-point
    p0, p1, after-p2
    Uses: .distance(), .get_k(), .includes_point(), .angle()

    Input: point (xy list), line ([xy, xy]), mr (float)
    Output: {
        "smoothener": [xy, xy, xy] OR [xy, xy] OR False  (...anything needed to keep network smooth, except biaxial connectors)
        "line": [xy, xy]  (may or may not be cut)
    }  (points of dict ordered)
    """
    # protect inputs:
    point = point.copy()
    line = [xy.copy() for xy in line]

    line_length = distance(*line)
    if get_k(*line) == get_k(line[1], point):
        if includes_point(line, point, 0.1):
            smoothener = False
            line = [point, line[1]]
        else:
            smoothener = [point, line[0]]

        return {"smoothener": smoothener, "line": line}

    beta = angle(point, *line)
    y_length = distance(point, line[0])
    shortest_x = False

    # assume x < z/2:
    x = sqrt((mr * y_length * sin(beta)) / 2)
    z = x ** 2 + y_length ** 2 - 2 * x * y_length * cos(beta)
    if x <= z / 2:  # x inside assumed M' circle - return
        shortest_x = x

    # ...unsuccessful, assume x > z/2 and y > z/2:
    if not shortest_x:
        k_6 = 1 / 64
        k_5 = (3 / 32) * y_length * cos(beta)
        k_4 = (1 / 64) * y ** 2 * (12 * cos(beta) ** 2 - 4 * mr ** 2 * y_length ** 2 * sin(beta) ** 4 + 3)
        k_3 = (1 / 16) * y_length ** 3 * cos(beta) * (2 * cos(beta) ** 2+ 3)
        k_2 = (3 / 64) * y_length ** 3 * (4 * y_lengh * cos(beta) ** 2 + 1)
        k_1 = (3 / 32) * y_length ** 5 * cos(beta)
        k_0 = y_length ** 6 / 64

        x = polynomials.x
        equation = k_6 * x**6 + k_5 * x**5  + k_4 * x**4  + k_3 * x**3  + k_2 * x**2  + k_1 * x**1  + k_0
        imaginary_xs = equation.roots(iterations=100)

        for complex_x in imaginary_xs:
            z = sqrt(complex_x.real ** 2 + y_length ** 2 - 2 * complex_x.real * y_length * cos(beta))
            if (complex_x.real > z / 2) and (y_length > z / 2):
                shortext_x = complex_x.real if complex_x.real < shortest_x else shortest_x

    # ... also unsuccessful, assume y < z / 2
    if not shortest_x:
        x = (2 * y ** 2) / (mr * sin(beta))
        z = x ** 2 + y_length ** 2 - 2 * x * y_length * cos(beta)
        shortest_x = x if y_length < z / 2 else False

    # shift p2 on line, return gained link
    if 0 < shortest_x < line_length:
        coefficient = shortest_x / line_length
        x_shift = (line[1][0] - line[0][0]) * coefficient
        y_shift = (line[1][1] - line[0][1]) * coefficient
        p2 = [line[0][0] + x_shift, line[0][1] + y_shift]
        return {"smoothener": [point, line[0], p2], "line": [p2, line[1]]}

    return False

def connect_bezier(point, curve, step, mr, offsets = False):
    """
    Smoothly connects a point to a bezier curve (with separated offsets). All offsets
    are abolished in the process. Input variables must follow this order:
    point, [p0, p1, p2]
    Returns
    Uses: .check_curvature(), .triangle_tangent(), .extended_intersection(), .distance(), .get_k(), .narrow_bezier()

    Input: point (xy), curve ([xy, xy, xy]), step (float), mr (float), offsets ([xy/False, xy/False] or False)
    Output: {
        "smoothener": [xy, xy, xy] OR [xy, xy] OR False  (...anything needed to keep network smooth, except biaxial connectors)
        "curve": [xy, xy, xy]  (any original offsets already narrowed away)
    }  (points of dict ordered)
    """
    # protect inputs:
    point = point.copy()
    curve = [xy.copy() for xy in curve]
    if offsets:
        curve = narrow_bezier(curve, offsets)

    if point == curve[0]:
        return {"smoothener": False, "curve": curve}
    elif get_k(point, curve[0]) == get_k(*curve[:2]) \
    and distance(point, curve[0]) < distance(point, curve[1]):  #
        return {"smoothener": [point, curve[0]], "curve": curve}

    # try to construct connector from tangent and: pb_line (point-CP0) OR pb_alternative (point-skipped_tail_halfpoint)
    bez_points = approx_bezier(curve, [], step)
    pb_line = [point, curve[0]]
    for i in range(1, len(bez_points) - 2):
        triangle = bez_points[i-1 : i+2]
        tangent = triangle_tangent(triangle)
        p1 = extended_intersection([pb_line, tangent])
        if p1:  # trying pb_line---tangent:
            if check_curvature(mr, point, p1, bez_points[i]):
                smoothener = [point, p1, bez_points[i]]
                curve = narrow_bezier(curve, [bez_points[i], False])
                return {"smoothener": smoothener, "curve": curve}

    return False

def smoothen_lines(line1, line2, mr):
    """
    line/line smothener - takes xy of two properly directed lines
    (their points may have a gap from complex vertices or on intersections), and smallest
    allowed radius of smoothener's osculating circle in meters. Returns conventional smoothener
    dict (still directed) - {
        "line1":[xy, xy],
        "smoothener":   (...anything needed to keep network smooth, except biaxial connectors)
                   OR [[xy, xy, xy]]                (angle corner, most common)
                   OR [xy, xy]                      (1 line with a gap)
                   OR False                         (1 continuous line),
        "line2":[xy, xy]
    }
    Breaks down on impossible smoothener - adjust curvature and omit_connector to avoid that
    Implements: https://algorithmist.wordpress.com/2010/12/01/quad-bezier-curvature/
    Uses: .get_k(), .extended_intersection(), .angle(), .distance(), .includes_point()

    Input: line1 ([xy, xy]), line2 ([xy, xy]), mr (float)
    Output: directed modified lines with smoothener (*smoothener dict*) OR False
    """
    line1 = [xy.copy() for xy in line1]
    line2 = [xy.copy() for xy in line2]  # keep function pure - protect lists
    lines = [line1, line2]
    A = line1[0]
    B = line2[1]
    gap = line1[1] != line2[0]  # gap between lines - find connecting point
    if gap:
        V = extended_intersection(lines)
    else:
        V = line1[1]

    # Determine the nature of needed smoothener (False, line, Bezier of double_Bezier):
    if get_k(*line1) == get_k(*line2):
        if not gap or line1[1] == line2[0]:
            # 1: NONE (False) SMOOTHENER
            return {"line1": line1, "smoothener": False, "line2": line2}
        elif get_k(line1[0], line2[1]) == get_k(*line1):
            # 2: LINE SMOOTHENER:
            return {"line1": line1, "smoothener": [line1[1], line2[0]], "line2": line2}

    # 3: COMMON BEZIER SMOOTHENER - establish inner angle between crossing lines:
    alpha = angle(A, V, B)

    # find the base points of smoothener's control triangle, Ie. Bez p0 and p2:
    arm_length = (cos(alpha / 2) * mr) / (sin(alpha / 2) ** 2)
    points = []
    for i in range(2):  # find p0 and p1 on arms
        [V, R] = [[V, A], [V, B]][i]
        # evaluate shifts in I. quadrant (all values positive)
        line_length = distance(R, V)  # arm of angle === length of Bez control line
        x_shift = R[0] - V[0]
        y_shift = R[1] - V[1]
        arm_x = V[0] + x_shift * abs(arm_length / line_length)
        arm_y = V[1] + y_shift * abs(arm_length / line_length)
        points.append([arm_x, arm_y])

    [p0, p2] = points
    if not gap:
        if includes_point(line1, p0) and includes_point(line2, p2):
            return {"line1": [A, p0], "smoothener": [[p0, V, p2]], "line2": [p2, B]}
    else:
        if includes_point([A, V], p0) and includes_point([B, V], p2):
            return {"line1": [A, p0], "smoothener": [[p0, V, p2]], "line2": [p2, B]}

    return False  # corner is impossible to smoothen simply - try biaxial smoothener

def check_deviancy(line1, line2, allowed_deviancy):
    """
    Checks whether the angle between given two lines doesn't exceed allowed_deviancy,
    given as a float representing radians. Yields False if angle is too great.
    Uses: .get_k(), .angle()

    Input: line1 (xy, xy), line2 (xy, xy), allowed_deviancy (float)
    Output: allowed (bool)
    """
    k1, k2 = get_k(*line1), get_k(*line2)
    if k1 == k2:
        return True

    axis = [[0, 0], [1, 0]]
    a = [2, k1]
    b = [2, k2]
    angles = []
    for point in [a, b]:
        if point[1] == "Vertical":
            angles.append(pi)
        else:
            angles.append(angle(*axis, point))

    return abs(angles[0] - angles[1]) <= allowed_deviancy

def smoothen_combined(line, bezier, step, mr, allowed_deviancy = 0.05):
    """
    bezier/line smoothener - takes xy of properly directed bezier control points and line
    (their points may have a gap from complex vertices or on intersections), step for bezier
    interpolation, and smallest allowed radius of smoothener's osculating circle in meters.
    Smoothener is False or line when difference in tangent's angles stays within allowed_deviancy (in radians)
    Returns conventional smoothener dict: (...anything needed to keep network smooth, except biaxial connectors)
    dict (still directed) - {
        "line1":[xy, xy],
        "smoothener": [[xy, xy, xy]] OR [xy, xy] OR False
        "curve2":[[xy, xy, xy]]
        }
    Uses: .distance(), .get_k(), .extended_intersection(), .triangle_tangent(), .approx_bezier(), .angle(), .includes_point(), .narrow_bezier(), .check_deviancy()
    Implements: https://algorithmist.wordpress.com/2010/12/01/quad-bezier-curvature/

    Input:  line ([xy, xy]), bezier ([[xy, xy, xy]]), step (float), mr (float)
    Output: directed modified lines with smoothener (*smoothener dict*) OR False
    """
    # protect mutable inputs (keep function pure)
    line = [xy.copy() for xy in line]
    offsets = [False, False]
    bezier = [xy.copy() for xy in bezier[0]]
    if isinstance(bezier[-1], dict):
        offsets = bezier.pop()["Offsets"]
        bezier = narrow_bezier(bezier, offsets)

    bez_points = approx_bezier(bezier, [], step, offsets)
    line_k = get_k(*line)
    A = line[0]
    gap = bezier[-1] != line[0]

    # Determine the nature of needed smoothener (False, line, or Bezier):
    if (check_deviancy(line, bezier[:2], allowed_deviancy)):
        #  1st control abiscissa is an extention of 1st line (1 connect point, no more overlap)
        if  ((line[1][0] <= bezier[0][0] <= bezier[1][0]) or (line[1][0] >= bezier[0][0] >= bezier[1][0])) \
        and ((line[1][1] <= bezier[0][1] <= bezier[1][1]) or (line[1][1] >= bezier[0][1] >= bezier[1][1])):
            # 1: NONE (False) SMOOTHENER
            if not gap or line[1] == bez_points[0]:
                return {"line1": line, "smoothener": False, "curve2": [bezier]}
            # 2: LINE SMOOTHENER:
            elif get_k(line[0], bezier[1]) == line_k:
                return {"line1": line, "smoothener": [line[1], bez_points[0]], "curve2": [bezier]}

    # 3: COMMON BEZIER SMOOTHENER - establish inner angle between crossing lines:
    smoothener_found = False
    for i in range(1, len(bez_points) - 1):
        if i == 1:
            tangent = bez_points[:2]
        else:
            triangle = [bez_points[i-1], bez_points[i], bez_points[i+1]]
            tangent = triangle_tangent(triangle)

        V = extended_intersection([line, tangent])
        p2 = bez_points[i]
        if not V or V == p2:  # touching, unsmooth bezier
            continue  # TODO test this! - 'not V' added as protection against parallels

        # gather p0, V, p2] trinity and its angle
        arm_length = distance(V, p2)
        line_length = distance(V, A)
        coefficient = abs(arm_length / line_length)
        x_shift = V[0] - A[0]
        y_shift = V[1] - A[1]
        p0 = [V[0] - x_shift * coefficient, V[1] - y_shift * coefficient]
        alpha = angle(p0, V, p2)

        # length of arm on minimal curvature, specific to each angle:
        arm_minimal = (cos(alpha / 2) * mr) / (sin(alpha / 2) ** 2)
        if arm_minimal <= arm_length:
            smoothener_found = True
            break

    if smoothener_found and includes_point([line[0], V], p0):
        offsets[0] = p2
        bezier = narrow_bezier(bezier, offsets)
        return {"line1":[line[0], p0], "smoothener": [[p0, V, p2]], "curve2": [bezier]}

    return False  # corner is impossible to smoothen (try double-bezier smoothener?)

def smoothen_beziers(bezier1, bezier2, step, mr, allowed_deviancy = 0.05):
    """
    bezier/bezier smoothener - takes xy of properly directed bezier control points of 2 lines
    (their points may have a gap from complex vertices or on intersections), step for bezier
    interpolation, and smallest allowed radius of smoothener's osculating circle in meters.
    Smoothener is False or line when difference in tangent's angles stays within allowed_deviancy (in radians).
    Returns conventional smoothener dict: (...anything needed to keep network smooth, except biaxial connectors)
    dict (still directed) - {
        "curve1": [[xy, xy, xy]],
        "smoothener": [[xy, xy, xy]] OR [xy, xy] OR False
        "curve2": [[xy, xy, xy]]
        }
    Uses: .distance(), .get_k(), .extended_intersection(), .triangle_tangent(), .approx_bezier(), .narrow_bezier(), .check_deviancy()
    Implements: https://algorithmist.wordpress.com/2010/12/01/quad-bezier-curvature/

    Input:  bezier ([[xy, xy, xy]]), bezier ([[xy, xy, xy]]), step (float), mr (float)
    Output: directed modified lines with smoothener (*smoothener dict*) OR False
    """
    offsets1 = [False, False]
    offsets2 = [False, False]
    # protect input lists (keep function pure)
    bezier1 = [xy.copy() for xy in bezier1[0]]
    bezier2 = [xy.copy() for xy in bezier2[0]]
    if isinstance(bezier1[-1], dict):
        offsets1 = bezier1.pop()["Offsets"]

    if isinstance(bezier2[-1], dict):
        offsets2 = bezier2.pop()["Offsets"]

    bez1_points = approx_bezier(bezier1, [], step, offsets1)
    bez2_points = approx_bezier(bezier2, [], step, offsets2)
    gap = bez1_points[-1] == bez2_points[0]
    border_line1 = bezier1[1:] if not offsets1[1] else bez1_points[-2:]
    border_line2 = bezier2[:2] if not offsets2[0] else bez2_points[:2]

    if check_deviancy(border_line1, border_line2, allowed_deviancy):
        if offsets1[0] or offsets1[1]:
            bezier1 = narrow_bezier(bezier1, offsets1)
        if offsets2[0] or offsets2[1]:
            bezier2 = narrow_bezier(bezier2, offsets2)
        # 1: NONE (False) SMOOTHENER
        if not gap or border_line1[1] == border_line2[0]:
            return {"curve1":[bezier1], "smoothener": False, "curve2": [bezier2]}
        # 2: LINE SMOOTHENER:
        else:
            return {"curve1":[bezier1], "smoothener": [border_line1[1], border_line2[0]], "curve2": [bezier2]}

    # 3: LOOK FOR REGULAR BEZIER SMOOTHENER:
    smoothener_found = False
    for i in range(1, len(bez1_points) - 1):  # i can iterate through for both beziers, cuz step is the same
        if i == 1:
            tangent1 = bez1_points[-2:]
            tangent2 = bez2_points[:2]
        else:
            triangle1 = [bez1_points[-(i + 1)], bez1_points[-i], bez1_points[-(i - 1)]]
            triangle2 = [bez2_points[i-1], bez2_points[i], bez2_points[i+1]]
            tangent1 = triangle_tangent(triangle1)
            tangent2 = triangle_tangent(triangle2)

        # establish control triangle, point M and its area
        p0 = bez1_points[-(i + 1)]
        V = extended_intersection([tangent1, tangent2])   # TODO what when yielded False?
        p2 = bez2_points[i]
        p0p2 = distance(p0, p2)
        p0V = distance(p0, V)
        p2V = distance(p2, V)
        M = [
            p0[0] + (p2[0] - p0[0]) / 2,  # x right between p0 and p2
            p0[1] + (p2[1] - p0[1]) / 2
        ]
        MV = distance(M, V)
        p = (p0p2 + p0V + p2V) / 2  # Heron's formula - step 1: obtain p
        area = sqrt(p * (p - p0p2) * (p - p0V) * (p - p2V))  # Heron's formula - step 2: calculate area
        # calculate suggested
        if MV >= p0V or MV >= p2V:  # V is inside circles
            max_arm = p0V if p0V < p2V else p2V
            suggested_radius = (max_arm ** 3) / area
        else:
            suggested_radius = (area ** 2) / (MV ** 3)

        pre1 = bez1_points[-(i + 2)]  # bezier1 point preceding p0
        fol2 = bez2_points[i+1]  # bezier2 point following p2
        # conditions: suggested radius long enough
        #           + V in correct direction with respect to 1st bezier
        #           + V in correct direction with respect to 2nd bezier
        if (suggested_radius >= mr) \
        and ((pre1[0] <= p0[0] <= V[0]) or (pre1[0] >= p0[0] >= V[0])) \
        and ((pre1[1] <= p0[1] <= V[1]) or (pre1[1] >= p0[1] >= V[1])) \
        and ((fol2[0] <= p2[0] <= V[0]) or (fol2[0] >= p2[0] >= V[0])) \
        and ((fol2[1] <= p2[1] <= V[1]) or (fol2[1] >= p2[1] >= V[1])):  # BROKEN # I left 'ere
            smoothener_found = True
            break

    if smoothener_found:
        offsets1[1] = p0
        offsets2[0] = p2
        bezier1 = narrow_bezier(bezier1, offsets1)
        bezier2 = narrow_bezier(bezier2, offsets2)
        return {"curve1":[bezier1], "smoothener": [[p0, V, p2]], "curve2": [bezier2]}

    return False  # corner is impossible to smoothen

def cut_gap(gap, shape, mr, step = False):
    """
    Given gap line and shape, rearranges them to have shared point in the middle,
    uses smoothen_lines or smoothen_combined to cut gap line, in order to establish
    e1-CP : CP-e2 ratio. Returns distance by which gap was cut (always positive)
    Uses: .get_k(), .distance(), .smoothen_lines(), .smoothen_combined()

    Input: gap ([xy, xy]),
            shape ([xy, xy, xy] OR [xy, xy],  # NO OFFSETS!
            mr (float)
            step (float OR False)
    Output: gap ([xy, xy])
    """
    gap = [xy.copy() for xy in gap]
    shape = [xy.copy() for xy in shape]

    if gap[0] == shape[1]:
        gap, shape = gap[::-1], shape[::-1]   # reverse lines - put shared point in the middle

    # extend gap on both ends to allow cut overreach, DON'T FLIP POINTS
    extended_gap = extend_abiscissa(gap, 100)

    if len(shape) == 3:
        cut_gap = smoothen_combined(extended_gap, [shape], step, mr)
    else:
        cut_gap = smoothen_lines(extended_gap, shape, mr)

    assert cut_gap, "mr impossible for given road network ~ abstract_geometry.cut_gap()"
    return distance(*cut_gap["smoothener"][0][:2])

def biaxial_cutter(shape1, shape2, mr, step = False):
    """
    Takes in two properly oriented shapes (lines or beziers) with a gap, establishes
    middle point between them, asymetrically cuts them to create 2 smootheners, obeying
    mimnimal radius.
    Uses: .distance(), .connect_line(), .connect_bezier() .narrow_bezier(), .get_k(), .cut_gap()

    Input: shape1 ([xy, xy]) OR ([[xy, xy, xy, offs_dict]]),
           shape1 ([xy, xy]) OR ([[xy, xy, xy, offs_dict]]),
           mr (float)
    Output: {
        "line1" : [xy, xy] OR "curve1" : [xy, xy, xy]
        "smoothener1" : [xy, xy, xy],
        "smoothener2" : [xy, xy, xy],
        "line2" : [xy, xy] OR "curve2" : [xy, xy, xy]
    }  (dict) OR False  (False, if no biaxal of given curvature possible)
    """
    # HERE BEGINS biaxial_cutter; protect mutable inputs, establish gap
    input_shapes = [shape1, shape2]
    shapes = []
    gap = []
    for i in range(2):
        if isinstance(input_shapes[i][0][-1], dict):  # curve with offsets
            assert step, "Cutting curve, its step must be given ~ abstract_geometry.biaxial_cutter()"
            offsets = input_shapes[i][0][-1]["Offsets"]
            shape = narrow_bezier(input_shapes, offsets)
        elif isinstance(input_shapes[i][0][-1], list):  # curve, no offsets
            assert step, "Cutting curve, its step must be given ~ abstract_geometry.biaxial_cutter()"
            shape = [xy.copy() for xy in input_shapes[i][0]]
        else:
            assert len(input_shapes[i]) == 2, "Invalid argument (non-nested bezier?) ~ abstract_geometry.biaxial_cutter()"
            shape = [xy.copy() for xy in input_shapes[i]]

        shapes.append(shape)
        gap.append(shape[i-1])

    # establish e1-CP : CP-e2 ratio as the ratio of distances cut by smootheners:
    e1, e2 = gap
    assert e1 != e2, "No gap to bridge with biaxial cutter. ~ abstract_geometry.biaxial_cutter()"
    cut_distances = []
    for shape in shapes:
        cut_distances.append(cut_gap(gap, shape, mr, step))

    gap_length = distance(*gap)
    e1CP_prop = cut_distances[0] / cut_distances[1]  # e1-CP proportion; gap split by cp into e1CP : 1 parts
    CP_ratio = e1CP_prop / (e1CP_prop + 1)

    gap_x_shift = gap[1][0] - gap[0][0]
    gap_y_shift = gap[1][1] - gap[0][1]
    CP = [
        gap[0][0] + gap_x_shift * CP_ratio,
        gap[0][1] + gap_y_shift * CP_ratio
    ]
    # construct smooth biaxial pair:
    keys = [
        "line" if len(shapes[0]) == 2 else "curve",
        "line" if len(shapes[1]) == 2 else "curve"
    ]
    connector1 = connect_line(CP, shapes[0][::-1], mr) if keys[0] == "line" else connect_bezier(CP, shapes[0][::-1], step, mr)
    connector2 = connect_line(CP, shapes[1], mr) if keys[1] == "line" else connect_bezier(CP, shapes[1], step, mr)
    if connector1 and connector2:
        return {
            keys[0] + "1": connector1[keys[0]][::-1],
            "smoothener1": connector1["smoothener"][::-1],
            "smoothener2": connector2["smoothener"],
            keys[1] + "2": connector2[keys[1]]
        }
    return False  # all in vain, no biaxial smoothener found

def biaxial_connector(line1, line2, mr):
    """
    Takes in two properly oriented lines with a gap, attempts to smoothly breach the gap
    with a biaxial connection: 2 bezier curves of smallest possible curvature. If their
    curvature is indeed small enough (Ie. with higher radius than mr).
    Returns back the two beziers, properly oriented.
    Does not alter input lines' length, so can be also used on beziers.
    Uses: .distance(), .angle(), .cut_gap(), .extended_intersection(), .deduce_p1s(), .check_curvature()

    Input: line1 ([xy, xy]),  line2 ([xy, xy]), mainimal_radius (float), allow_cut (bool)
    Output: {
        "smoothener1" : [xy, xy, xy],
        "smoothener2" : [xy, xy, xy]
    }  (dict) OR False  (False, if no biaxal of given curvature possible)
    """
    assert len(line1) == len(line2) == 2, "Connector links 2 lines [xy, xy]. ~ abstract_geometry.biaxial_connector()"
    line1 = [xy.copy() for xy in line1]
    line2 = [xy.copy() for xy in line2]
    lines = [line1, line2]
    gap = [line1[1], line2[0]]

    # establish e1-CP : CP-e2 ratio as the ratio of distances cut by smootheners:
    assert gap[0] != gap[1], "No gap to bridge with biaxial cutter. ~ abstract_geometry.biaxial_cutter()"
    cut_distances = []
    for line in lines:
        cut_distances.append(cut_gap(gap, line, mr))

    gap_length = distance(*gap)
    e1CP_prop = cut_distances[0] / cut_distances[1]  # e1-CP proportion; gap split by cp into e1CP : 1 parts
    CP_ratio = e1CP_prop / (e1CP_prop + 1)

    gap_x_shift = gap[1][0] - gap[0][0]
    gap_y_shift = gap[1][1] - gap[0][1]
    CP = [
        gap[0][0] + gap_x_shift * CP_ratio,
        gap[0][1] + gap_y_shift * CP_ratio
    ]

    # gather all possible p1s and q1s from both lines, order them ascendingly by beta deviation (by gap tilt):
    tilts = {}  # format: tilt : (p1, line_index) (0 or 1)
    tilt_keys = []
    for i in range(2):
        line = lines[i][::1-2*i]  # reverse second line
        p1s = deduce_p1s(mr, *line, CP)
        if not p1s:
            continue
        # insert-sort computed tilts in proper order:
        for p1 in p1s:
            beta = angle(line[1], CP, p1)
            if tilt_keys == []:
                tilt_keys = [beta]
                tilts[beta] =  (p1, i)

            j = len(tilt_keys)
            while(beta < tilt_keys[j-1] and j > 0):
                j -= 1

            if j == 0 or tilt_keys[j-1] != beta:  # duplicates protection
                tilt_keys.insert(j, beta)
                tilts[beta] = (p1, i)

    sorted_p1s = [tilts[key] for key in tilt_keys]  # format: (p1, line_index)
    for p1_pair in sorted_p1s:
        p1, line_i = p1_pair
        other_e0e1 = lines[1-line_i][::2*line_i-1]
        other_p1 = extended_intersection([other_e0e1, [p1, CP]])
        if not other_p1:  # other_p1 found?
            continue
        elif not (((other_e0e1[0][0] < other_e0e1[1][0] < other_p1[0]) or \
                   (other_e0e1[0][0] > other_e0e1[1][0] > other_p1[0])) and \
                  ((other_e0e1[0][1] < other_e0e1[1][1] < other_p1[1]) or \
                   (other_e0e1[0][1] > other_e0e1[1][1] > other_p1[1]))):  # other_p1 in right direction?
            continue
        elif not check_curvature(mr, other_e0e1[1], other_p1, CP):  # deduced smoothener mr-okay?
            continue

        p1s = [other_p1, p1] if line_i else [p1, other_p1]
        return {
            "smoothener1": [line1[1], p1s[0], CP],
            "smoothener2": [CP, p1s[1], line2[0]]
        }
    return False  # no double smoothener possible
