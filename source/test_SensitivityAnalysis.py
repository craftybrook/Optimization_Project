import unittest
import numpy as np
# Assuming your main file is named SensitivityAnalysis.py
from SensitivityAnalysis import *
import sys
import os
from helper_classes import *
import json



# --- HELPER FUNCTION FOR TESTS ---
def create_temp_fold_file(filename, coords, faces, edge_vertices=None, edge_assignments=None):
    """
    Helper to write a temporary .fold JSON file for testing.
    """
    data = {
        "file_spec": 1.1,
        "vertices_coords": coords,
        "faces_vertices": faces,
        "edges_vertices": edge_vertices if edge_vertices else [],
        "edges_assignment": edge_assignments if edge_assignments else []
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def wrap_angle_difference(delta):
    return (delta + np.pi) % (2*np.pi) - np.pi


# --- TEST SUITE 1: PHYSICS KERNELS (Bars & Hinges) ---
class TestElementPhysics(unittest.TestCase):
    
    def setUp(self):
        """Set up nodes and elements for physics testing."""
        # 1. Setup for Bar Tests
        self.n1 = Node(0, 0, 0, 0)
        self.n2 = Node(1, 10, 0, 0) # Bar 10 units long on X-axis
        self.bar = BarElement(self.n1, self.n2)

        # 2. Setup for Hinge Tests
        # Create 4 nodes forming two triangles in the XY plane
        self.ni = Node(0, 0, 1, 0)   # Wing 1 (top)
        self.nj = Node(1, 0, 0, 0)   # Hinge Start
        self.nk = Node(2, 5, 0, 0)   # Hinge End
        self.nl = Node(3, 8, -1, 0)  # Wing 2 (bottom)
        
        # NOTE: Updated to include 'M' assignment
        self.hinge = HingeElement(self.ni, self.nj, self.nk, self.nl, "M")

    # --- BAR TESTS ---
    def test_bar_direction_cosines(self):
        """Verify the bar identifies its orientation correctly."""
        row = self.bar.get_compatibility_matrix_row(total_DOFs=6)
        # Check Node 1 (Tail) [-1, 0, 0]
        np.testing.assert_array_almost_equal(row[0:3], [-1.0, 0.0, 0.0])
        # Check Node 2 (Head) [1, 0, 0]
        np.testing.assert_array_almost_equal(row[3:6], [1.0, 0.0, 0.0])

    def test_bar_rigid_body_mode(self):
        """Moving the whole bar shouldn't create strain."""
        row = self.bar.get_compatibility_matrix_row(6)
        # Move both nodes 5 units in Y 
        u_vector = np.array([0, 5, 0, 0, 5, 0])
        stretch = np.dot(row, u_vector)
        self.assertAlmostEqual(stretch, 0.0, places=12)

    # --- HINGE TESTS ---
    def test_hinge_flat_angle(self):
        """A flat assembly should return ~0.0 angle."""
        angle = self.hinge.calculate_dihedral_angle()
        self.assertAlmostEqual(angle, 0.0, places=7)

    def test_hinge_90_deg_fold(self):
        """Rotate one wing to 90 degrees and verify."""
        original = self.ni.coordinates.copy()
        # Move node 'ni' to be on Z-axis
        self.ni.coordinates = np.array([0, 0, 1]) 
        angle = self.hinge.calculate_dihedral_angle()
        
        self.assertAlmostEqual(abs(angle), np.pi/2, places=7)
        self.ni.coordinates = original # Reset

    def test_jacobian_analytical_vs_numerical(self):
        """
        GOLDEN TEST: Verifies the analytical Jacobian gradients (1/h * n)
        against a brute-force Finite Difference calculation.
        """
        total_dofs = 12 
        numerical_J = np.zeros(total_dofs)
        nodes = [self.ni, self.nj, self.nk, self.nl]
        
        for node in nodes:
            original_coords = node.coordinates.copy()
            epsilon = 1e-6 
            
            for axis in range(3): 
                # Perturb +
                node.coordinates[axis] = original_coords[axis] + epsilon
                angle_plus = self.hinge.calculate_dihedral_angle()
                
                # Perturb -
                node.coordinates[axis] = original_coords[axis] - epsilon
                angle_minus = self.hinge.calculate_dihedral_angle()
                
                # Central Difference
                delta = wrap_angle_difference(angle_plus - angle_minus)
                gradient = delta / (2 * epsilon)
                
                idx = node.id * 3 + axis
                numerical_J[idx] = gradient
                
                node.coordinates[axis] = original_coords[axis] # Reset

        analytical_J = self.hinge.get_jacobian_row(total_dofs)

        np.testing.assert_allclose(
            analytical_J,
            numerical_J,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Analytical Jacobian does not match Finite Difference"
        )


# --- TEST SUITE 2: FILE PARSING & MODEL ASSEMBLY ---
class TestModelAssembly(unittest.TestCase):
    
    def setUp(self):
        self.test_file = "test_pattern.fold"

    def tearDown(self):
        """Delete the temp file after every test."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_load_basic_triangle(self):
        """Verify model loads nodes and panels from file correctly."""
        coords = [[0,0,0], [1,0,0], [0,1,0]]
        faces = [[0,1,2]]
        create_temp_fold_file(self.test_file, coords, faces)

        model = SensitivityModel(self.test_file)
        
        self.assertEqual(len(model.nodes), 3)
        self.assertEqual(len(model.panels), 1)
        self.assertEqual(len(model.bars), 3) # A triangle has 3 bars

    def test_hinge_filtering_logic(self):
        """
        CRITICAL TEST: Verify that the model uses 'edges_assignment' 
        to filter out Boundaries (B) and keep Mountains (M).
        """
        # Two triangles sharing edge 1-2
        # Nodes: 0, 1, 2, 3
        coords = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
        faces = [[0,1,2], [1,2,3]]
        
        # Edge (1,2) is the shared edge.
        # We define it as 'M' (Mountain). 
        # Note: .fold parser sorts indices, so order [1,2] vs [2,1] doesn't matter.
        edges_v = [[1,2]] 
        edges_a = ["M"] 

        create_temp_fold_file(self.test_file, coords, faces, edges_v, edges_a)
        model = SensitivityModel(self.test_file)

        # Should find exactly 1 hinge
        self.assertEqual(len(model.hinges), 1)
        # Should have assignment 'M'
        self.assertEqual(model.hinges[0].fold_assignment, "M")

    def test_boundary_ignore_logic(self):
        """
        Verify that if an edge is marked 'B' (Boundary), 
        the model DOES NOT create a hinge there, even if panels touch.
        """
        coords = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
        faces = [[0,1,2], [1,2,3]] # Shares edge 1-2
        
        # Explicitly mark the shared edge as Boundary
        edges_v = [[1,2]]
        edges_a = ["B"] 

        create_temp_fold_file(self.test_file, coords, faces, edges_v, edges_a)
        model = SensitivityModel(self.test_file)

        # Logic check:
        # Parser filters out 'B', so generate_hinges sees no entry in dictionary.
        # It should skip creating the hinge.
        self.assertEqual(len(model.hinges), 0, "Should not create hinges on Boundary edges")

    def test_unassigned_assignment(self):
        """
        If edge info is missing entirely from the file, but topology exists,
        what should happen? 
        Current Logic: It defaults to 'U' if no dict exists, OR skips if dict exists but edge is missing.
        Let's test the 'Empty Edge List' case (Simulating a barebones .fold file).
        """
        coords = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
        faces = [[0,1,2], [1,2,3]]
        
        # No edge data provided
        create_temp_fold_file(self.test_file, coords, faces)
        model = SensitivityModel(self.test_file)
        
        # The parser will return an empty dict for crease_info.
        # The generate_hinges loop checks `if hasattr(self, 'crease_info')`.
        # If the list is empty, it will default to skipping everything (strict mode)
        # OR defaulting to 'U'. 
        
        # Based on your current code: 
        # "if edge_key in self.crease_info: ... else: continue"
        # So without edge data, it finds NO hinges.
        self.assertEqual(len(model.hinges), 0)

    def test_topology_error_detection(self):
        """Test the 3-panel overlap error."""
        coords = [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1]]
        # 3 Panels sharing edge 0-1
        faces = [
            [0,1,2],
            [0,1,3],
            [0,1,4] 
        ]
        create_temp_fold_file(self.test_file, coords, faces)

        with self.assertRaises(ValueError):
            SensitivityModel(self.test_file)

if __name__ == '__main__':
    unittest.main()