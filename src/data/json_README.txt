Traffic networks are understood as graph data structures.
The conventional structure of a network layout JSON goes as follows:

{
  "Vertices": {
    "A": {"Coordinates": [0,0], "Neighbours": ["B"]},
    "B": {"Coordinates": [40,0], "Neighbours": ["A","C","G"]},
    "C": {"Coordinates": [180,0], "Neighbours": ["B","D","H"]},
    "D": {"Coordinates": [220,0], "Neighbours": ["C","E","I"]},
    ...
    },

  "Edges": [
    {"Vertices": ["A","B"], "Cost": 2, "Street": "Hlavní", "Width": 2},
    {"Vertices": ["B","C"], "Cost": 7, "Street": "Hlavní", "Width": 2}},
    {"Vertices": ["B","G"], "Cost": 3, "Street": "Vysoká", "Shape": [[30,100], [20,120]]},
    {"Vertices": ["C","D"], "Cost": 2, "Street": "Hlavní", "Width": -0.5}},
    ...
  ],

  "Rails": {
    "Vertices": {
      "A1": {"Coordinates": [0,0], "Neighbours": ["B1"], "NeighbourVertices": ["B"]},
      "B1": {"Coordinates": [0,0], "Neighbours": ["B2"], "NeighbourVertices": ["B"]},
      "A3": {"Coordinates": [0,0], "Neighbours": ["B1", "B2"], "NeighbourVertices": ["B"]},
      "A4": {"Coordinates": [0,0], "Neighbours": ["B1", "B2"], "NeighbourVertices": ["B"]},
      "A1": {"Coordinates": [0,0], "Neighbours": ["B1", "B2"], "NeighbourVertices": ["B"]},
      ...
      },

    "Edges": [
      {"Vertices": ["A","B"], "Cost": 2, "Street": "Hlavní", "Width": 2},
      {"Vertices": ["B","C"], "Cost": 7, "Street": "Hlavní", "Width": 2}},
      {"Vertices": ["B","G"], "Cost": 3, "Street": "Vysoká",
        "Shape": [[30,100], [20,120]]
      },
      {"Vertices": ["C","D"], "Cost": 2, "Street": "Hlavní", "Width": -0.5,
        "Shape": [[180,0], [[190,0],[200,30], [220,0],
          {"Offsets": [False, [215,0]]}
        ]]
      },
      ...
    ]
  }
}


  "Vertices" represent intersections (graph nodes). In the inevitable case
there's more than 26, the naming convention adds as if letters were numbers:
A, B, C, ... Z, AA, AB, AC, ... AZ, BA, BB, BC ... ZZ, AAA, AAB, AAC ...
By the nature of processing algorithms, "Vertices" MAY NOT be added ad-hoc,
(The "Vertices" object is a primary for DataProcessors).
  "Coordinates" tell the x and y distances from topleft corner, in meters!
  "Neighbours" are a list of all connected vertices connected to this node.


  "Edges" represent chunks of road defined by their border Vertices. The "Cost"
is now established by means of computing distance between intersections. That is
not ideal, I hope to weigh-in road's top speed, traffic magnitude, and curvature.

  The optional "Width" attribute is the number of parallel roads on the edge -
the number of vehicles that can reasonably drive next to each other (applies
especially to motorways). SPECIAL CASE: A one way road is 0.5 if it's in
alphabetical A -> Z direction, -0.5 if it's not. The default, assumed value of 1
means a standard two-way road for 1 vehicle in each direction.

  The optional "Shape" attribute determines the exact shape of the road. Every
possible road shape is broken down into a sequence of lines and Bezier curves
(in the direction of A -> Z alphabetical order). An xy tuplet, eg. [30, 25] means
a point defining a line, array of xy tuplets, eg [[0, 40], [100, 50], [20, 25]],
means a Bezier curve. The assumed, default shape is a straight line.

  "Street"s are a bit of a cherry on top for now.


  "Rails" contains a complex structure of trails the cars are actually driving
  -> "Vertices" mean intersection points. Their "Coordinates" are simply rail
      crossings belonging to 1 vertice. It is assumed, that half-gauge is short enough
      not to push any two vertices' points over each other. "NeighbourVertices" are graph
      vertices the rails from a given intersection point lead to (see detailed drawing
      in notebook). CRUCIALLY, "NeighbourVertices" are ALWAYS in this order:
      1, left-of-edge rail, 2, right-of-edge rail (of next counter-clockwise neighbour)

  -> "Edges" are ordered [start, end], so their order depends on the country we drive in
      A "Rails" dict without "Edges" is considered invalid.

# # # ----- RIGID DATA STANDARDIZATION PRINCIPLES ----- # # #
- All coordinates are in meters relative to bottom-left (south-west) corner
- "Vertices" are the primary graph material
- "Edges" are secondary graph material, necessary to render graph
- "Edges": "Vertices" are ordered in direction (Ie. depend on drive_right)
- "Edges": "Shape" is a primary shape material
- "Edges": "Shape": [<start-xy-list>, <xy-list>, ..., <end-xy-list>]
- "Edges": "Shape": [<list>[<start-xy>, <xy>, ..., <end-xy>]<list>] is a Bezier curve defined by list of control points
- "Edges": ""Shape": [<list>[<start-xy>, <xy>, ..., <dict>"Offsets":[<xy or False>, <xy or False>]<dict>]<list>]
  is a special element of bezier control point array - it holds bezier's starting and end points in case
  the curve doesn't start or end in its control point (this also applies also for Rails Edges Shape)
- "Rails" are written so that their rendering process is identical to graph's
- "Rails": "Edges": "Vertices" are ordered in direction (Ie. depend on drive_right)
