import numpy as np
import matplotlib.pyplot as plt

from source.SensitivityAnalysis import SensitivityModel
from source.Bloom_Yoshimura import Bloom_Yoshimura
from source.visualization import plot_sensitivity_violin





# --- Pattern Generation Setup ---
def set_up_bloom(m=5,h=1,s=1,file_name=None, show_plot = None,Show_Origin=1, Show_Points=1, Show_facets=0, Show_Lines=1, Line_Width=1, Line_Style=1, Invert_Creases=0):
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

    bloom.export_to_fold(filename=file_name)


# --- Main Execution ---
if __name__ == "__main__":
    try:
        filename = "birdsfoot4.fold"
        birdsfoot_model = SensitivityModel(filename)
        birdsfoot_model.analyze_sensitivity(show_plot='yes')
        birdsfoot_model.check_integration_rigidity(num_steps=1000, step_size= 0.001)

        
        # filename_r1_h3_m5_flasher = "r1_h3_m5_flasher.fold"
        # r1_h3_m5_flasher = SensitivityModel(filename_r1_h3_m5_flasher)
        # r1_h3_m5_flasher.analyze_sensitivity(show_plot='yes')
        # plt.close('all') 

        # filename_r1_h2_m6_flasher = "r1_h2_m6_flasher.fold"
        # r1_h2_m6_flasher = SensitivityModel(filename_r1_h2_m6_flasher)
        # r1_h2_m6_flasher.analyze_sensitivity(show_plot='no')
        # plt.close('all') 

        # filename_r1_h2_m5_flasher = "r1_h2_m5_flasher.fold"
        # r1_h2_m5_flasher = SensitivityModel(filename_r1_h2_m5_flasher)
        # r1_h2_m5_flasher.analyze_sensitivity(show_plot='yes')
        # r1_h2_m5_flasher.animate_nonlinear_folding()
        # plt.close('all')

        # filename_r1_h1_m6_flasher = "r1_h1_m6_flasher.fold"
        # r1_h1_m6_flasher = SensitivityModel(filename_r1_h1_m6_flasher)
        # r1_h1_m6_flasher.analyze_sensitivity(show_plot='yes')
        # plt.close('all') 
        
        # filename_Y6_1 = "Y6_1.fold"
        # set_up_bloom(m=6,h=1,s=1,file_name=filename_Y6_1) 
        # Y6_1 = SensitivityModel(filename_Y6_1)
        # Y6_1.analyze_sensitivity(show_plot='yes')
        # Y6_1.animate_nonlinear_folding()
        # plt.close('all') 
        
        # filename_Y5_1 = "Y5_1.fold"
        # set_up_bloom(m=5,h=1,s=1,file_name=filename_Y5_1) 
        # Y5_1 = SensitivityModel(filename_Y5_1)
        # Y5_1.analyze_sensitivity(show_plot='no')
        # plt.close('all')

        # filename_Y6_2 = "Y6_2.fold"
        # set_up_bloom(m=6,h=2,s=1,file_name=filename_Y6_2) 
        # Y6_2 = SensitivityModel(filename_Y6_2)
        # Y6_2.analyze_sensitivity(show_plot='no')
        # Y6_2.check_integration_rigidity(num_steps=100, step_size= 0.01)
        # plt.close('all') 

        # # --- Data Collection ---
        # sensitivities = {
        #     "Y5_1" : Y5_1.best_sensitivity,
        #     "Y6_1" : Y6_1.best_sensitivity,
        #     "Y6_2" : Y6_2.best_sensitivity,
        #     "r1_h3_m5_flasher" : r1_h3_m5_flasher.best_sensitivity,
        #     "r1_h2_m6_flasher" : r1_h2_m6_flasher.best_sensitivity,
        #     "r1_h2_m5_flasher" : r1_h2_m5_flasher.best_sensitivity,
        #     "r1_h1_m6_flasher" : r1_h1_m6_flasher.best_sensitivity,
        # }

        

        # # --- Generate Violin Plot ---
        # fig, ax = plot_sensitivity_violin(sensitivities)
        # fig.savefig("sensitivity_violin.pdf", bbox_inches="tight")
        # plt.show() 
        
    except FileNotFoundError as e:
        # Changed this to dynamically print the file that caused the error
        print(f"ERROR: Could not find file. Check your path. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")