import numpy as np
from math import cos, sin, radians, pi
from source.SensitivityAnalysis import SensitivityModel



if __name__ == "__main__":
    # =========================
    # NODE COORDINATES (ALL FLAT)
    # =========================
    coords = [
        [0.0, 0.0, 0.0],    # 0  center
        [1.0, 0.0, 0.0],    # 1
        [0.5, 0.866, 0.0],  # 2 (Fixed 0.87 to 0.866 for better precision)
        [-0.5, 0.866, 0.0], # 3
        [-1.0, 0.0, 0.0],   # 4
        [-0.5, -0.866, 0.0],# 5
        [0.5, -0.866, 0.0], # 6
        # Outer ring
        [2.0, 0.0, 0.0],    # 7
        [1.0, 1.732, 0.0],  # 8
        [-1.0, 1.732, 0.0], # 9
        [-2.0, 0.0, 0.0],   # 10
        [-1.0, -1.732, 0.0],# 11
        [1.0, -1.732, 0.0], # 12
    ]

    # =========================
    # PANEL DEFINITIONS
    # =========================
    indices = [
        # ---- Inner star (6 triangles) ----
        [1, 2, 0], [0, 2, 3], [0, 3, 4], 
        [0, 4, 5], [0, 5, 6], [0, 6, 1],
        # ---- Outer ring (6 quads) ----
        [1, 7, 8, 2], [2, 8, 9, 3], [3, 9, 10, 4], 
        [4, 10, 11, 5], [5, 11, 12, 6], [6, 12, 7, 1],
    ]

    model = SensitivityModel(coords, indices)

    print("\n==== FLAT COMPLEX ORIGAMI PATCH ====")
    print(f"Nodes:   {len(model.nodes)}")
    print(f"Panels:  {len(model.panels)}")
    print(f"Bars:    {len(model.bars)}")
    print(f"Hinges:  {len(model.hinges)} ")

    # 1. Run Analysis (Pinning included inside the class now)
    sensitivity_results = model.analyze_sensitivity()

    # 2. Print Top Active Hinges
    print("\nTop Active Hinges:")
    sorted_indices = np.argsort(np.abs(sensitivity_results))[::-1]
    for i in sorted_indices[:5]:
        print(f"Hinge {i}: {abs(sensitivity_results[i]):.4f}")

    # 3. Plot
    model.plot_pattern(
            sensitivity_vector=sensitivity_results,
            show_node_labels=False,  
            show_hinge_labels=True,   
            title="Kinematic Sensitivity (Red = High Motion)"
        )