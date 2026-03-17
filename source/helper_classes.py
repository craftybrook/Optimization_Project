import numpy as np
"""
These are the smaller less functional classes used in the the SensitivtyAnalysis class
"""

class Panel:
    def __init__(self, id, nodes):
        ## a panel will hold its own nodes. I will check each panel against all other panels to see if it shares nodes with other panels
        ## If it does share nodes we will make a hinge there. If the panel already has 4 hinges, we won't check it against other panels. waste of compute
        self.id = id
        self.nodes = nodes

    def get_nodes(self):
        return self.nodes

class Node:
    def __init__(self, id, x,y,z):
        # NOTE: These are the vertex coordinates of the pattern in the deployed state
        self.id = id
        self.coordinates = np.array([x,y,z], dtype=float)

class BarElement:
    def __init__(self,node_a, node_b):
        self.nodes = [node_a, node_b]
        
    def get_compatibility_matrix_row(self,total_DOFs):
        # extract the node coordinates
        point1 = self.nodes[0].coordinates
        point2 = self.nodes[1].coordinates

        # Calculate length between the nodes (length of the bar)
        vector_between_nodes = point2 - point1
        length_of_bar = np.linalg.norm(vector_between_nodes)

        # The direction cosines are the normalized components
        direction_cosine_x, direction_cosine_y, direction_cosine_z = vector_between_nodes/ length_of_bar

        # Intialize the row vector that will get put into the compatibility matrix
        row_vector = np.zeros(total_DOFs)

        # Node 1 indices
        # The syntax "* 3" is how we skip through the array to find the exact "starting slot" for a specific node.
        # The "index_1 : index_1 + 3" put the direction cosines for this bar in the right spot of the matrix
        index_1 = self.nodes[0].id * 3
        row_vector[index_1 : index_1 + 3] = [-direction_cosine_x, - direction_cosine_y, -direction_cosine_z]

        # Node 2 indices
        index_2 = self.nodes[1].id * 3
        row_vector[index_2 : index_2 + 3] = [direction_cosine_x, direction_cosine_y, direction_cosine_z]

        
        # Return the row for the compatibility matrix
        return row_vector

class HingeElement:
    def __init__(self, wing_nodes_1, node_j, node_k, wing_nodes_2, fold_assignment="U"):
        """
        wing_nodes_1 : list of Node objects — all non-hinge nodes in panel 1
        wing_nodes_2 : list of Node objects — all non-hinge nodes in panel 2
        node_j, node_k : the two nodes that define the shared hinge axis

        The Jacobian is computed via the centroid of each panel's non-hinge nodes.
        This makes the formula independent of which specific wing node is chosen,
        works correctly for triangles (m=1), quads (m=2), and any polygon.
        """
        # Compute centroids of each panel's non-hinge nodes
        c1 = np.mean([n.coordinates for n in wing_nodes_1], axis=0)
        c2 = np.mean([n.coordinates for n in wing_nodes_2], axis=0)

        e      = node_k.coordinates - node_j.coordinates
        r_jc1  = c1 - node_j.coordinates
        r_jc2  = c2 - node_j.coordinates

        triple = np.dot(np.cross(e, r_jc1), r_jc2)

        if abs(triple) < 1e-10:
            # Degenerate case: all nodes coplanar (e.g. flat/unfolded pattern).
            # Fall back to global +z reference: enforce n1 = cross(r_jc1, e) points in +z.
            n1 = np.cross(r_jc1, e)
            if np.dot(n1, np.array([0.0, 0.0, 1.0])) < 0:
                wing_nodes_1, wing_nodes_2 = wing_nodes_2, wing_nodes_1
        elif triple < 0:
            wing_nodes_1, wing_nodes_2 = wing_nodes_2, wing_nodes_1

        self.wing_nodes_1 = wing_nodes_1   # list of non-hinge nodes in panel 1
        self.wing_nodes_2 = wing_nodes_2   # list of non-hinge nodes in panel 2
        self.node_j = node_j
        self.node_k = node_k
        self.fold_assignment = fold_assignment

        # Keep node_i / node_l as the first element of each list for any code
        # that still references them (e.g. tests, dihedral angle calculation).
        self.node_i = wing_nodes_1[0]
        self.node_l = wing_nodes_2[0]

    def calculate_vectors(self):
        """
        Computes the hinge geometry using the centroid of each panel's
        non-hinge nodes.  Called by both get_jacobian_row and
        calculate_dihedral_angle.
        """
        self.hinge_line_vector   = self.node_k.coordinates - self.node_j.coordinates
        self.length_of_hinge_line = np.linalg.norm(self.hinge_line_vector)

        if self.length_of_hinge_line < 1e-12:
            raise ValueError("Degenerate hinge: coincident nodes")

        # Centroid of each panel's non-hinge nodes
        self.c1 = np.mean([n.coordinates for n in self.wing_nodes_1], axis=0)
        self.c2 = np.mean([n.coordinates for n in self.wing_nodes_2], axis=0)

        # Vectors from j to each centroid
        self.r_jc1 = self.c1 - self.node_j.coordinates
        self.r_jc2 = self.c2 - self.node_j.coordinates

        # Panel normals computed from the centroids — same direction as the
        # true panel normal for any flat polygon, scale proportional to the
        # "average triangle" area.
        self.panel_1_normal_vector = np.cross(self.r_jc1, self.hinge_line_vector)
        self.panel_2_normal_vector = np.cross(self.hinge_line_vector, self.r_jc2)

    def calculate_dihedral_angle(self):
        """The dihedral angle is the angle between the two panel planes.
        Uses centroid normals — correct for any flat polygon."""
        self.calculate_vectors()

        n1    = self.panel_1_normal_vector / np.linalg.norm(self.panel_1_normal_vector)
        n2    = self.panel_2_normal_vector / np.linalg.norm(self.panel_2_normal_vector)
        e_hat = self.hinge_line_vector     / np.linalg.norm(self.hinge_line_vector)

        y = np.dot(np.cross(n1, n2), e_hat)
        x = np.dot(n1, n2)

        return np.arctan2(y, x)

    def get_jacobian_row(self, total_DOFs):
        """
        Computes ∂θ/∂(every node DOF) using the centroid-based Schenk & Guest formula.

        Key idea
        --------
        Instead of one wing node per panel, we use the centroid of ALL non-hinge
        nodes in that panel as a virtual wing point.  The Schenk-Guest formula is
        applied exactly as written, but using that centroid.  Because the centroid
        moves as the average of its constituent nodes, the resulting gradient is
        then distributed equally back to each actual non-hinge node:

            gradient_wi  =  gradient_centroid / m        (m = number of non-hinge nodes)

        For triangular panels (m = 1) this reduces exactly to the original formula.
        For quad or higher-order panels the result is independent of which node
        is arbitrarily labeled the "wing node", making it symmetric by construction.
        """
        self.calculate_vectors()

        n1_sq = np.dot(self.panel_1_normal_vector, self.panel_1_normal_vector)
        n2_sq = np.dot(self.panel_2_normal_vector, self.panel_2_normal_vector)

        # Safety check for degenerate (zero-area) panels
        if n1_sq < 1e-12 or n2_sq < 1e-12:
            return np.zeros(total_DOFs)

        L = self.length_of_hinge_line

        # --- Schenk & Guest (2011) gradients at the virtual centroid wing points ---
        gradient_c1 = (L / n1_sq) * self.panel_1_normal_vector
        gradient_c2 = (L / n2_sq) * self.panel_2_normal_vector

        # Projection of each centroid along the hinge axis (alpha factors)
        L_sq     = L * L
        alpha_c1 = np.dot(self.r_jc1, self.hinge_line_vector) / L_sq
        alpha_c2 = np.dot(self.r_jc2, self.hinge_line_vector) / L_sq

        # Gradients at the hinge-axis nodes (unchanged from original formula)
        gradient_j = (alpha_c1 - 1) * gradient_c1 + (alpha_c2 - 1) * gradient_c2
        gradient_k = (-alpha_c1)    * gradient_c1 + (-alpha_c2)    * gradient_c2

        # --- Assemble the Jacobian row ---
        row = np.zeros(total_DOFs)

        def stamp(node_id, vector):
            idx = node_id * 3
            row[idx : idx+3] += vector   # += so multiple stamps to same node accumulate

        # Distribute panel-1 centroid gradient equally among all panel-1 wing nodes
        m1 = len(self.wing_nodes_1)
        for w in self.wing_nodes_1:
            stamp(w.id, gradient_c1 / m1)

        # Distribute panel-2 centroid gradient equally among all panel-2 wing nodes
        m2 = len(self.wing_nodes_2)
        for w in self.wing_nodes_2:
            stamp(w.id, gradient_c2 / m2)

        # Hinge-axis nodes get their gradient directly
        stamp(self.node_j.id, gradient_j)
        stamp(self.node_k.id, gradient_k)

        return row

# class HingeElement:
#     def __init__(self,node_i, node_j, node_k, node_l,fold_assignment="U"):
#         """By convention: j and k are the SHARED nodes (the hinge line)
#              and l are the unique nodes on either side"""
        
#        # Ensure consistent hinge orientation (geometry-based)
#         e = node_k.coordinates - node_j.coordinates
#         r_ji = node_i.coordinates - node_j.coordinates
#         r_jl = node_l.coordinates - node_j.coordinates

#         triple = np.dot(np.cross(e, r_ji), r_jl)

#         if abs(triple) < 1e-10:
#             # Degenerate case: all nodes are coplanar (e.g. flat/unfolded pattern).
#             # The triple product is identically zero, so the standard check cannot
#             # determine orientation.  Fall back to a global reference: enforce that
#             # n1 = cross(r_ji, e) points in the +z direction.
#             n1 = np.cross(r_ji, e)
#             if np.dot(n1, np.array([0.0, 0.0, 1.0])) < 0:
#                 node_i, node_l = node_l, node_i
#         elif triple < 0:
#             node_i, node_l = node_l, node_i

#         self.node_i = node_i
#         self.node_j = node_j
#         self.node_k = node_k
#         self.node_l = node_l

#         self.fold_assignment = fold_assignment # a hinge will be assigned Mountian or Valley
        
#     def calculate_vectors(self):
#         """ 
#         calculates/intializes several vectors used in this class to reduce duplicate code
#         just helper fucntion
#         """

#         # Vector for the hinge line (intersection between the panels)
#         self.hinge_line_vector = self.node_k.coordinates - self.node_j.coordinates
#         self.length_of_hinge_line = np.linalg.norm(self.hinge_line_vector)

#         if self.length_of_hinge_line < 1e-12:
#             raise ValueError("Degenerate hinge: coincident nodes")

#         # Vectors from the hinge to the outer nodes 
#         self.r_ji = self.node_i.coordinates - self.node_j.coordinates # (j -> i)
#         self.r_jl = self.node_l.coordinates - self.node_j.coordinates # (j -> l)

#         # Get vectors normal to each panel
#         self.panel_1_normal_vector = np.cross(self.r_ji, self.hinge_line_vector)
#         self.panel_2_normal_vector = np.cross(self.hinge_line_vector, self.r_jl)

#     def calculate_dihedral_angle(self): # used this to help me numerically validate the Jacobian calculations, not used in the final code
#         """The dihedral angle is the angle between two planes, 
#             we find this by finding the angle between the vectors normal to each plane"""
        
#         self.calculate_vectors()

#         n1 = self.panel_1_normal_vector / np.linalg.norm(self.panel_1_normal_vector)
#         n2 = self.panel_2_normal_vector / np.linalg.norm(self.panel_2_normal_vector)

#         e_hat = self.hinge_line_vector / np.linalg.norm(self.hinge_line_vector)

#         y = np.dot(np.cross(n1, n2), e_hat)
#         x = np.dot(n1, n2)

#         return np.arctan2(y, x)

#     def get_jacobian_row(self, total_DOFs):
#         """Mathematically, the Jacobian (J) answers:
#         "If I wiggle Node i in some direction, exactly how much does the fold angle θ change?"""
        
#         self.calculate_vectors()

#         # Calculate squared magnitudes (needed for the denominator)
#         panel_1_normal_vector_squared = np.dot(self.panel_1_normal_vector, self.panel_1_normal_vector)
#         panel_2_normal_vector_squared = np.dot(self.panel_2_normal_vector, self.panel_2_normal_vector)

#         # Safety check for zero-area triangles (collinear nodes)
#         if panel_1_normal_vector_squared < 1e-12 or panel_2_normal_vector_squared < 1e-12:
#             return np.zeros(total_DOFs)
        
#         # Calculate gradients (The "sensitivity" vectors)
#         # [cite_start]Formulas derived from Schenk & Guest (2011)

#         # Gradient for nodes i & l
#         gradient_i = (self.length_of_hinge_line / panel_1_normal_vector_squared) * self.panel_1_normal_vector
#         gradient_l = (self.length_of_hinge_line / panel_2_normal_vector_squared) * self.panel_2_normal_vector

#         # Projection factors: How far along the hinge are node i and l?
#         alpha_i = np.dot(self.r_ji, self.hinge_line_vector) / (self.length_of_hinge_line * self.length_of_hinge_line)
#         alpha_l = np.dot(self.r_jl, self.hinge_line_vector) / (self.length_of_hinge_line * self.length_of_hinge_line)

#         # gradient for j & k
#         gradient_j = (alpha_i - 1) * gradient_i + (alpha_l - 1) * gradient_l
#         gradient_k = (-alpha_i) * gradient_i - alpha_l * gradient_l

#         # intialize empty row vector to populate with gradients
#         row = np.zeros(total_DOFs)

#         # Helper function  to stamp 3 values at a time into the correct slots
#         def stamp(node_id, vector):
#             idx = node_id * 3
#             row[idx : idx+3] = vector

#         stamp(self.node_i.id, gradient_i)
#         stamp(self.node_j.id, gradient_j)
#         stamp(self.node_k.id, gradient_k)
#         stamp(self.node_l.id, gradient_l)

#         return row