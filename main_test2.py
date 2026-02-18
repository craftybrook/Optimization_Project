from source.SensitivityAnalysis import SensitivityModel

if __name__ == "__main__":
    # 1. Define the path to your .fold file
    # Make sure this file is in the same folder, or provide the full path
    filename = "BirdsFoot.fold" 

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
        model.plot_pattern(sensitivity, title=f"Sensitivity Analysis: {filename}")

    except FileNotFoundError:
        print(f"ERROR: Could not find file '{filename}'. Check your path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")