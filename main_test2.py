import numpy as np
import matplotlib.pyplot as plt

from source.SensitivityAnalysis import SensitivityModel
from source.Bloom_Yoshimura import Bloom_Yoshimura
from source.visualization import plot_sensitivity_violin, plot_fold_pattern

# --- Pattern Generation Setup ---
def set_up_bloom(m=5,h=1,s=1,file_name=None, show_plot=None, Show_Origin=1, Show_Points=1, Show_facets=0, Show_Lines=1, Line_Width=1, Line_Style=1, Invert_Creases=0):
    Show_Origin = bool(Show_Origin)
    Show_Points = bool(Show_Points)
    Show_facets = bool(Show_facets)
    Show_Lines = bool(Show_Lines)
    Line_Width = float(Line_Width)
    Line_Style = bool(Line_Style)
    Invert_Creases = bool(Invert_Creases)

    bloom = Bloom_Yoshimura(m,h,s) # M,H,S
    bloom.plot_origin = Show_Origin
    bloom.plot_points = Show_Points
    bloom.plot_facets = Show_facets

    bloom.plot_lines = Show_Lines
    bloom.line_width = Line_Width
    bloom.line_style = Line_Style
    bloom.crease_is_invert = Invert_Creases
    

    bloom.graph()
    if show_plot is not None:
        plt.close('all')

    bloom.export_to_fold(filename=file_name) # type: ignore

#%% Bloom Yoshimura Y6_1
filename_Y6_1 = "Y6_1.fold"
set_up_bloom(m=6,h=1,s=1,file_name=filename_Y6_1)

# 1. FIRST, load the uncut model to see what edges actually exist
model_uncut = SensitivityModel(filename_Y6_1)  # <-- using the correct file!

print("--- VALID INTERNAL EDGES YOU CAN CUT ---")
for i, h in enumerate(model_uncut.hinges):
    print(f"Hinge {i}: ({h.node_j.id}, {h.node_k.id})")
print("----------------------------------------\n")


# 2. Pick two edges from the list printed above and put them here:
# (Replace these with real pairs printed to your console)
inactive_hinges = [(1, 4), (2, 4)] 

# 3. NOW run the model with the cuts
model_cut = SensitivityModel(filename_Y6_1, cut_edges=inactive_hinges)

print(f"Hinges without cuts: {len(model_uncut.hinges)}")
print(f"Hinges with cuts: {len(model_cut.hinges)}")

# Run the analysis normally
model_cut.analyze_sensitivity(show_plot='yes')

plt.close('all')

# Get the standard deviation of the fold magnitudes
#std_dev = model_cut.get_sensitivity_standard_deviation()

# Animate
#model_cut.animate_nonlinear_folding()