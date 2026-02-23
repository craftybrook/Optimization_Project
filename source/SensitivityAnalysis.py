import numpy as np
import sympy as sp
import itertools
import json
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from scipy.linalg import eigh
from matplotlib import animation

#  IMPORT BLOCK
try:
    # This works when running from the ROOT directory (e.g., main.py)
    from source.helper_classes import *
except ModuleNotFoundError:
    # This works when running from INSIDE the source directory (e.g., test files)
    from helper_classes import *

"""
This is the meat of this script
Febuaray 2026
Jake Sutton
"""

class SensitivityModel:
    def __init__(self, fold_file_path):
        """ Upon initializing this class makes the origami pattern, then adds the bars between
        nodes in a panel to make it rigid, and also slaps on some hinges. Telling it where the hignes
        are is helpful for calculatring the dihedral angle jacobian. """
        self.coordinates, self.panel_indices, self.crease_info = self.extract_pattern_data_from_fold_file(fold_file_path)

        self.nodes, self.panels = self.generate_geometry(self.coordinates, self.panel_indices)

        self.bars = self.generate_bars()
        self.hinges = self.generate_hinges()
        
        
    def analyze_sensitivity(self):
        # 1. Build Matrices
        dihedral_jacobian = self.build_dihedral_jacobian()
        constraint_matrix = self.build_constraint_matrix()

        # 2. Solve SVD
        _, singular_values, Vh = np.linalg.svd(constraint_matrix)

        # 3. Analyze the Null Space (S ≈ 0)
        print("\n--- NULL SPACE ANALYSIS (S ≈ 0) ---")
        print(f"{'Mode Idx':<10} | {'Singular Value (S)':<20} | {'Folding Magnitude':<20} | {'Type'}")
        print("-" * 75)

        best_sensitivity = None
        max_folding = -1.0
        
        # We check every single mode to find the zeros
        # singular_values is sorted High -> Low, so the zeros are at the end.
        for i in range(len(singular_values)):
            s_val = singular_values[i]
            
            # THRESHOLD: Only look at modes that are effectively zero energy
            if s_val < 1e-9:
                current_nullspace_vector = Vh[i, :]
                
                # Calculate how much this mode folds the hinges
                current_crease_changes = dihedral_jacobian @ current_nullspace_vector
                current_total_folding = np.sum(np.abs(current_crease_changes))
                
                # Classify the mode
                if current_total_folding < 1e-5:
                    mode_type = "Rigid Body (Motion without Folding)"
                else:
                    mode_type = "*** MECHANISM *** (Valid Folding)"
                    
                    # Track the best mechanism
                    if current_total_folding > max_folding:
                        max_folding = current_total_folding
                        best_sensitivity = current_crease_changes

                print(f"{i:<10} | {s_val:.4e}           | {current_total_folding:.6f}             | {mode_type}")

        print("-" * 75)

        if best_sensitivity is None:
             print("WARNING: No mechanism detected in the Null Space.")
             return np.zeros(len(self.hinges))

        # 4. Pin sign convention using fold assignments from the .fold file.
        #    SVD eigenvectors are defined up to a global sign flip (+v or -v).
        #    We resolve this ambiguity by requiring mountain folds (M) to be
        #    positive — i.e. they activate in the mountain direction.
        mountain_indices = [i for i, h in enumerate(self.hinges) if h.fold_assignment == 'M']
        if len(mountain_indices) > 0:
            if np.sum(best_sensitivity[np.array(mountain_indices)]) < 0:
                print("FLIPPING SENSITIVITY VECTOR TO MATCH MOUNTAIN FOLD CONVENTION...(we want mountains to be positive)")
                best_sensitivity = -best_sensitivity

        self.mountain_valley_check(best_sensitivity)

        # --- This prints all the innards ---
        self.print_system_matrices(dihedral_jacobian, constraint_matrix, singular_values, Vh, best_sensitivity)
        mechanism_mode_index = 7 # Based on your table output
        mechanism_vector = Vh[mechanism_mode_index, :] # Grab that specific row
        
        return best_sensitivity

    def print_system_matrices(self, dihedral_jacobian, constraint_matrix, singular_values, Vh, sensitivity_vector):
        """
        Prints the core matrices with Node DOF column labels and respective row labels.
        Replaces near-zero values with '0.0' to highlight matrix sparsity.
        """
        # 1. Generate Column Labels (Node DOFs: N0_x, N0_y, N0_z, ...)
        col_labels = []
        for n in self.nodes:
            col_labels.extend([f"N{n.id}_x", f"N{n.id}_y", f"N{n.id}_z"])

        # 2. Generate Row Labels
        bar_labels = [f"Bar {i}" for i in range(len(self.bars))]
        hinge_labels = [f"Hinge {i}" for i in range(len(self.hinges))]
        
        # Helper function for clean, aligned printing
        def print_labeled_matrix(name, matrix, r_labels, c_labels):
            print(f"\n--- {name} ---")
            if len(matrix) == 0:
                print("Matrix is empty.")
                return
                
            # Dynamic spacing based on label length
            label_width = max([len(str(lbl)) for lbl in r_labels] + [10])
            col_width = 8
            
            # Print Header
            header = f"{'':>{label_width}} | " + " | ".join([f"{col:>{col_width}}" for col in c_labels])
            print(header)
            print("-" * len(header))
            
            # Print Rows
            for i, row in enumerate(matrix):
                r_lbl = r_labels[i]
                # Replace near-zeros with "0.0" for visual clarity
                row_str = " | ".join([f"{val:>{col_width}.4f}" if abs(val) > 1e-9 else f"{'0.0':>{col_width}}" for val in row])
                print(f"{r_lbl:>{label_width}} | {row_str}")

        # --- Execute Printing ---
        print("\n" + "="*100)
        print("LABELED SYSTEM MATRICES".center(100))
        print("="*100)

        print_labeled_matrix("CONSTRAINT MATRIX (Bars)", constraint_matrix, bar_labels, col_labels)
        print_labeled_matrix("DIHEDRAL JACOBIAN (Hinges)", dihedral_jacobian, hinge_labels, col_labels)

        # For Vh (Null Space), we usually only care about the bottom rows where S ≈ 0
        null_space_rows = Vh[-9:, :] if len(Vh) >= 5 else Vh
        vh_labels = [f"Mode {len(Vh) - len(null_space_rows) + i}" for i in range(len(null_space_rows))]
        print_labeled_matrix("NULL SPACE MODES (Last rows of Vh)", null_space_rows, vh_labels, col_labels)

        print("\n--- SELECTED SENSITIVITY VECTOR ---")
        if sensitivity_vector is not None:
            # Print vertically so it's easy to read against the specific hinges
            print(f"{'Hinge':>10} | {'Sensitivity (rad)':>18}")
            print("-" * 31)
            for i, val in enumerate(sensitivity_vector):
                print(f"Hinge {i:>4} | {val:>18.6f}")
        else:
            print("None (No valid mechanism found).")

        print("="*100 + "\n")    
   
    def mountain_valley_check(self, sensitivity_vector):
        # 5. Validate: every hinge's sensitivity sign should match its .fold assignment.
        #    Mountain (M) → positive,  Valley (V) → negative.
        print("\n--- FOLD ASSIGNMENT VALIDATION ---")
        all_match = True
        for i, h in enumerate(self.hinges):
            s_val = sensitivity_vector[i]
            if h.fold_assignment == 'M':
                match = s_val >= 0
            elif h.fold_assignment == 'V':
                match = s_val <= 0
            else:
                continue  # skip unassigned hinges
            status = "✓" if match else "✗ MISMATCH"
            if not match:
                all_match = False
            print(f"  H{i} ({h.fold_assignment}): s = {s_val:+.6f}  {status}")
        if all_match:
            print("  All folds are consistent with .fold assignments.")
        else:
            print("  WARNING: Some folds are inconsistent with .fold assignments!")
        print("-" * 40)
            
    def build_dihedral_jacobian(self):
        number_of_hinges = len(self.hinges)
        number_of_nodes = len(self.nodes)
        total_DOFs = 3 * number_of_nodes

        dihedral_jacobian = np.zeros((number_of_hinges, total_DOFs))

        for i, hinge in enumerate(self.hinges):
            dihedral_jacobian[i,:] = hinge.get_jacobian_row(total_DOFs)

        return dihedral_jacobian
    
    def build_constraint_matrix(self):
        """
        Builds the constraint matrix (from the bars)
        """
        number_of_bars = len(self.bars)
        number_of_nodes = len(self.nodes)
        total_DOFs = 3 * number_of_nodes

        constraint_matrix = np.zeros((number_of_bars, total_DOFs))

        for i, bar in enumerate(self.bars):
            constraint_matrix[i, :] = bar.get_compatibility_matrix_row(total_DOFs)

        return constraint_matrix

    def extract_pattern_data_from_fold_file(self, fold_file_path):
        """
        Parses a .fold file and yanks the data from it. 

        Returns a type list of [x,y,z] coords
        panel_indices type list
        crease_lines (set)
        """
        with open(fold_file_path, 'r') as file:
            data = json.load(file)

        # extract coordinates
        coordinates = data['vertices_coords']
        #if z-coord is missing, add zero for z coord
        if len(coordinates[0]) == 2:
            coordinates = [[c[0], c[1], 0.0] for c in coordinates]

        # extract panel indices
        panel_indices = data['faces_vertices']

        # extract crease lines
        # M = mountian V = valley B = boundry U = unassigned
        crease_info = {}
        if 'edges_vertices' in data and 'edges_assignment' in data:
            for edge, assignment in zip(data['edges_vertices'], data['edges_assignment']):
                # Filter for Mountains and Valleys only
                if assignment in ['M', 'V']: 
                    # Sort indices so (1,2) is the same as (2,1)
                    u, v = sorted(edge)
                    crease_info[(u, v)] = assignment

                    """
                    crease_info is a dictionary that looks like this:
                    {
                    (13, 14): "M",  # From Index 1
                    (3, 15): "M",   # From Index 2
                    ...
                    (11, 21): "V"   # From Index 50
                    }
                    """

        return coordinates, panel_indices, crease_info

    def generate_geometry(self,coordinates, panel_indices):
        """ Generates the node objects and the panel obejects.
        These are simple object. See helper_classes to look at them."""
        
        nodes = self.generate_nodes(coordinates)
        panels = self.generate_panels(nodes,panel_indices)

        return nodes, panels
    
    def generate_panels(self, nodes, panel_indices):
        # This loop assigns nodes to different panels
        panels = []
        for i, idxs in enumerate(panel_indices):
            """ We look up the Node objects using the indices provided.
            If panel 1 uses node index 2, and panel 2 uses node index 2,
            they both get the EXACT SAME Node object from memory. 
            This makes sure that if Node 2 moves, it moves for both panels"""

            p_nodes = [nodes[k] for k in idxs]
            panels.append(Panel(i, p_nodes))

            
        return panels

    def generate_nodes(self, coordinates):
        """ Coordinates: List of [X,Y,Z] for every unique vertex (node)
        panel_indices: List of lists, e.g., [[0,1,2], [0,2,3,4]]
        This handles n-sides polygon panels"""

        # This loop creates all the nodes with the provided coordinates
        nodes = []
        count = 0
        for coordinate_list in coordinates:
            x = coordinate_list[0]
            y = coordinate_list[1]
            z = coordinate_list[2]

            new_node = Node(count, x ,y ,z)
            nodes.append(new_node)
            count += 1

        return nodes

    def generate_bars(self):
        """ 
        Creates a rigid "truss" for every panel, regardless of how many sides the panel has.
        It makes a bar between a node and every other node. 4 nodes = 6 bars, 3 nodes = regular trianlge
        
        WARNING: This creates non-deterministic panels. If this program were to be scaled/edited to analyze 
        panel bending/shearing, this function would need to be edited to generate deterministic panels. 

        """

        unique_edges = set()
        bars = []

        for panel in self.panels:
            # Connect every node to every other node in this specific panel
            for node_a, node_b in itertools.combinations(panel.nodes,2):

                # Sort IDs to ensure Edge(1,2) = Edge (2,1)
                edge_id = tuple(sorted((node_a.id, node_b.id)))

                if edge_id not in unique_edges:
                    unique_edges.add(edge_id)
                    #Add the rigid bar
                    bars.append(BarElement(node_a, node_b))

        return bars

    def generate_hinges(self):
        """ Detects shared edges and creates a hinge element. """
        # TODO: make it so that it only puts a hinge where we tell it there is a hinge, not at every shared edge

        hinges = []

        # map edges to panels
        edge_to_panels = {}

        # this loops checks to see if an edge has panels touching it already
        for panel in self.panels:
            number_nodes = len(panel.nodes)
            for i in range(number_nodes):
                node1 = panel.nodes[i]
                node2 = panel.nodes[(i+1) % number_nodes] #wrap around

                edge_key = tuple(sorted((node1.id, node2.id)))
                if edge_key not in edge_to_panels:
                    edge_to_panels[edge_key] = []
                edge_to_panels[edge_key].append(panel)

        for edge_key, panel_list in edge_to_panels.items():
            count = len(panel_list)

            if count > 2: # if there is more than 2 panels on an edge, something is wrong
                panel_ids = [panel.id for panel in panel_list]
                raise ValueError(f"TOPOLOGY ERROR: Edge between Nodes {edge_key} is shared by {count} panels "
                    f"(Panels: {panel_ids}).\n"
                    "Real origami edges can only connect 2 panels. "
                    "Check your input indices for overlapping panels."
                )
            # if only 1 panel, its a free edge and no hinge is needed there
            if count < 2:
                continue

            # Default assignment if no dictionary is provided
            assignment = 'U' 

            # If we have the dictionary from the .fold file...
            if hasattr(self, 'crease_info') and self.crease_info is not None:
                # Check if this edge exists in our "Valid Creases" list
                if edge_key in self.crease_info:
                    assignment = self.crease_info[edge_key] # assignment should be grabbing an "M" or a "V", and then assiging it to the hinge.
                else:
                    # If it's NOT in the dictionary, it's likely a Boundary ('B')
                    # that we filtered out in the parser. Skip it!
                    continue

            # This logic below is if there are just 2 panels, we create a hinge
            panel1 = panel_list[0]
            panel2 = panel_list[1]

            # Identify Axis Nodes (j, k)
            # Find the actual Node objects in panel_1 matching the IDs in edge_key
            node_j = next(n for n in panel1.nodes if n.id == edge_key[0])
            node_k = next(n for n in panel1.nodes if n.id == edge_key[1])

            # Identify "Wing" Nodes (i, l) - any node NOT on the axis
            node_i = next(n for n in panel1.nodes if n.id not in edge_key)
            node_l = next(n for n in panel2.nodes if n.id not in edge_key)

            hinges.append(HingeElement(node_i, node_j, node_k, node_l, assignment))
        
        return hinges
    
    def plot_pattern(self, sensitivity_vector=None, show_node_labels=True, show_hinge_labels=True, title="Pattern", normalize=False):
        plt.close('all') 
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- Process Data ---
        plot_sens = np.zeros(len(self.hinges))
        
        if sensitivity_vector is not None:
            # Flatten but KEEP THE SIGNS
            plot_sens = np.array(sensitivity_vector).flatten()
            
        # Determine the maximum absolute value to center the colorbar
        max_abs_val = np.max(np.abs(plot_sens))
        if max_abs_val < 1e-12: max_abs_val = 1.0

        # --- Normalization Strategy ---
        if normalize:
            # Scale -1.0 to +1.0
            plot_sens = plot_sens / max_abs_val
            limit = 1.0
        else:
            # Keep raw values
            limit = max_abs_val

        # --- REVERSED COLOR MAP (Negative=Red, Positive=Blue) ---
        cmap = plt.cm.coolwarm_r  # <--- The "_r" reverses the gradient
        norm = plt.Normalize(-limit, limit)

        # --- Plot Nodes & Bars ---
        xs = [n.coordinates[0] for n in self.nodes]
        ys = [n.coordinates[1] for n in self.nodes]
        zs = [n.coordinates[2] for n in self.nodes]
        ax.scatter(xs, ys, zs, c='black', s=20, alpha=0.4)

        if show_node_labels:
            for n in self.nodes:
                ax.text(n.coordinates[0], n.coordinates[1], n.coordinates[2], 
                        f"{n.id}", fontsize=8, color='grey')

        for bar in self.bars:
            p1, p2 = bar.nodes[0].coordinates, bar.nodes[1].coordinates
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color='black', alpha=0.3, linewidth=1)

        # --- Plot Hinges ---
        for h_id, h in enumerate(self.hinges):
            p_j, p_k = h.node_j.coordinates, h.node_k.coordinates
            
            raw_val = plot_sens[h_id]
            
            # 1. Get Color (Now inverted: Neg=Red, Pos=Blue)
            color = cmap(norm(raw_val))

            # 2. Constant Width
            width = 3.0

            # 3. Plot
            ax.plot([p_j[0], p_k[0]], [p_j[1], p_k[1]], [p_j[2], p_k[2]], 
                    color=color, linestyle='-', linewidth=width, alpha=0.9)

            # 4. Label
            if show_hinge_labels and abs(raw_val) > (0.1 * limit):
                mid = (p_j + p_k) / 2
                
                label_text = f"H{h_id}"
                if not normalize:
                    label_text += f"\n{raw_val:.2f}"
                
                ax.text(mid[0], mid[1], mid[2], label_text, 
                        color=color, fontsize=9, fontweight='bold',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="white")])

        # --- Formatting ---
        all_coords = np.array([xs, ys, zs])
        max_range = np.ptp(all_coords, axis=1).max() / 2.0
        mid_x, mid_y, mid_z = np.mean(all_coords[0]), np.mean(all_coords[1]), np.mean(all_coords[2])
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_title(title)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        
        label = 'Normalized Mode (-1 to +1)' if normalize else 'Hinge Motion (Radians)'
        cbar.set_label(label, rotation=270, labelpad=15)

        plt.show()

    