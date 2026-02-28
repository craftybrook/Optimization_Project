from source.SensitivityAnalysis import SensitivityModel
from source.Bloom_Yoshimura import Bloom_Yoshimura

def set_up_bloom(m=5,h=1,s=1, Show_Origin=1, Show_Points=1, Show_facets=0, Show_Lines=1, Line_Width=1, Line_Style=1, Invert_Creases=0):
    ''' ENTER YOUR PREFERENCES HERE: '''
    # m = ["     7     "] # example values: 5, 6, 8, 12 #
    # h = ["     1     "] # example values: 1, 2, 3 #
    # s = ["     1     "] # example values: 1, 4, 50, 27.5, 1001 #
    # Show_Origin = ["     1     "] # enter 1 to show origin or 0 to hide origin.
    # Show_Points = ["     1     "] # enter 1 to show points or 0 to hide points.
    # Show_facets = ["     0     "] # enter 1 to show facets or 0 to hide facets.
    # Show_Lines = ["     1     "] # enter 1 to show lines or 0 to hide lines.
    # Line_Width = ["     1     "]  # example values: 0.7, 2.5, 0.625, 5 #
    # Line_Style = ["     1     "]  # enter 1 for colored lines and 0 for black lines.
    # Invert_Creases = ["      0      "]  # enter 1 to invert creases or 0 for no change.
    ''' END OF PREFERENCES '''
    # m = int(m[0])
    # h = int(h[0])
    # s = float(s[0])
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
    bloom.export_to_fold()

if __name__ == "__main__":
    # 1. Define the path to your .fold file
    # Make sure this file is in the same folder, or provide the full path
    filename = "BirdsFoot3.fold" 

    # set_up_bloom(m=5,h=1,s=1) # Generates bloom_yoshimura.fold in the current directory
    # filename = "bloom_yoshimura.fold"

    # filename = "flasher.fold"

    try:
        print(f"Loading {filename}...")
        
        # 2. Initialize the Model

        model = SensitivityModel(filename)
        print(f"Model Loaded: {len(model.nodes)} Nodes, {len(model.hinges)} Hinges detected.")

        # 3. Run Analysis
        print("Solving SVD...")
        sensitivity = model.analyze_sensitivity()

        # 4. Visualize
        print("Plotting results...")
        # model.plot_pattern(sensitivity, title=f"Sensitivity Analysis: {filename}")
        

    except FileNotFoundError:
        print(f"ERROR: Could not find file '{filename}'. Check your path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")