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
        """
        Identifies the physical folding mechanism via SVD of the mechanism
        subspace matrix A = J @ Q^T. Automatically corrects arbitrarily 
        scrambled hinge coordinate systems by referencing the .fold M/V assignments.
        """
        # 1. Build Matrices
        dihedral_jacobian = self.build_dihedral_jacobian()
        constraint_matrix = self.build_constraint_matrix()

        # 2. Solve SVD on Constraint Matrix
        _, singular_values, Vh = np.linalg.svd(constraint_matrix)
        n_dof = Vh.shape[0]

        # 3. Enumerate all null space modes, classify as RBM vs Mechanism
        mechanism_indices = []   

        for i in range(n_dof):
            s_val = singular_values[i] if i < len(singular_values) else 0.0 
            if s_val < 1e-9:
                v             = Vh[i, :]
                fold_changes  = dihedral_jacobian @ v
                total_folding = np.sum(np.abs(fold_changes))

                # If the nodes move but the hinges don't fold, it's a rigid body/junk mode
                if total_folding >= 1e-5:
                    mechanism_indices.append(i)

        if not mechanism_indices:
            print("WARNING: No mechanism detected in the Null Space.")
            return np.zeros(len(self.hinges))

        # 4. Build mechanism subspace matrix Q and Fold matrix A
        Q = Vh[mechanism_indices, :]
        A = dihedral_jacobian @ Q.T

        # 5. Build target vector from fold assignments
        target_fold_vector = np.zeros(len(self.hinges))
        for i, h in enumerate(self.hinges):
            if h.fold_assignment == 'M':
                target_fold_vector[i] = +1.0
            elif h.fold_assignment == 'V':
                target_fold_vector[i] = -1.0
                
        print("\nTarget fold vector (t):")
        for i, val in enumerate(target_fold_vector):
            assignment = self.hinges[i].fold_assignment
            print(f"  Hinge {i:>4} ({assignment}): t = {val:+.1f}")

        # 6. INITIAL SVD of A
        U_sv, S_sv, Vt_sv = np.linalg.svd(A, full_matrices=False)

        # Select the mode closest to our M/V target (using absolute cosine similarity)
        if np.linalg.norm(target_fold_vector) > 1e-12:
            best_r        = 0
            best_cos      = -1.0
            rel_threshold = 1e-3 * S_sv[0]   
            for r in range(len(S_sv)):
                if S_sv[r] < rel_threshold:  
                    continue
                cos = np.dot(U_sv[:, r], target_fold_vector) / (np.linalg.norm(U_sv[:, r]) * np.linalg.norm(target_fold_vector))
                if abs(cos) > best_cos:      
                    best_cos = abs(cos)
                    best_r   = r
        else:
            best_r = 0                       

        best_sensitivity = U_sv[:, best_r] * S_sv[best_r]
        v_dominant = Q.T @ Vt_sv[best_r, :]

        # Fix the global sign for the initial pass
        if np.linalg.norm(target_fold_vector) > 1e-12 and np.dot(best_sensitivity, target_fold_vector) < 0:
            best_sensitivity = -best_sensitivity
            v_dominant       = -v_dominant

        # =====================================================================
        # THE FIX: HINGE AUTO-CALIBRATION (SWAP AND RERUN)
        # =====================================================================
        print("\nChecking for scrambled hinge orientations based on M/V assignments...")
        mismatches_found = False
        
        for i, h in enumerate(self.hinges):
            s_val = best_sensitivity[i]
            
            # If math says negative, but the .fold assignment is Mountain (+)
            if h.fold_assignment == 'M' and s_val < -1e-5:
                h.wing_nodes_1, h.wing_nodes_2 = h.wing_nodes_2, h.wing_nodes_1
                h.node_i, h.node_l = h.node_l, h.node_i
                mismatches_found = True
                
            # If math says positive, but the .fold assignment is Valley (-)
            elif h.fold_assignment == 'V' and s_val > 1e-5:
                h.wing_nodes_1, h.wing_nodes_2 = h.wing_nodes_2, h.wing_nodes_1
                h.node_i, h.node_l = h.node_l, h.node_i
                mismatches_found = True

        if mismatches_found:
            print("Mismatches found! Swapping internal panel definitions and rerunning Dihedral Jacobian...")
            
            # Rebuild only the Jacobian and A matrix (Constraint matrix C hasn't changed!)
            dihedral_jacobian = self.build_dihedral_jacobian()
            A = dihedral_jacobian @ Q.T
            
            # Rerun SVD on the newly aligned A matrix
            U_sv, S_sv, Vt_sv = np.linalg.svd(A, full_matrices=False)
            
            # Re-evaluate best mode alignment
            if np.linalg.norm(target_fold_vector) > 1e-12:
                best_r = 0
                best_cos = -1.0
                rel_threshold = 1e-3 * S_sv[0]
                for r in range(len(S_sv)):
                    if S_sv[r] < rel_threshold:
                        continue
                    cos = np.dot(U_sv[:, r], target_fold_vector) / (np.linalg.norm(U_sv[:, r]) * np.linalg.norm(target_fold_vector))
                    if abs(cos) > best_cos:
                        best_cos = abs(cos)
                        best_r = r
            else:
                best_r = 0

            # Grab the newly corrected vectors
            best_sensitivity = U_sv[:, best_r] * S_sv[best_r]
            v_dominant = Q.T @ Vt_sv[best_r, :]
            
            # Fix global sign one last time
            if np.linalg.norm(target_fold_vector) > 1e-12 and np.dot(best_sensitivity, target_fold_vector) < 0:
                best_sensitivity = -best_sensitivity
                v_dominant = -v_dominant
                
            print("Rerun complete. Hinges are now permanently aligned to the .fold file.")
        else:
            print("No scrambled orientations found. Initial pass is perfectly aligned.")
        # =====================================================================

        # Report singular value spectrum, marking the selected mode
        print(f"\nMechanism subspace singular values (fold efficiency per unit displacement):")
        for r, sv in enumerate(S_sv):
            marker = f"  ← selected (best M/V alignment, rank {r})" if r == best_r else ""
            print(f"  σ_{r} = {sv:.6f}{marker}")

        # Report alignment quality
        norm_s = np.linalg.norm(best_sensitivity)
        norm_t = np.linalg.norm(target_fold_vector)
        if norm_s > 1e-12 and norm_t > 1e-12:
            cos_sim = np.dot(best_sensitivity, target_fold_vector) / (norm_s * norm_t)
            quality = 'excellent' if cos_sim > 0.99 else \
                      'good'      if cos_sim > 0.90 else \
                      'moderate'  if cos_sim > 0.50 else 'poor'
            print(f"M/V alignment score (cosine similarity with target): "
                  f"{cos_sim:.6f}  ({quality})")

        # 7. Validate and report
        self.mountain_valley_check(best_sensitivity)
        self.print_system_matrices(dihedral_jacobian, constraint_matrix, singular_values, Vh,
                                   best_sensitivity,
                                   mechanism_indices=mechanism_indices,
                                   Q=Q, A=A,
                                   U_sv=U_sv, S_sv=S_sv, Vt_sv=Vt_sv,
                                   v_dominant=v_dominant,
                                   t=target_fold_vector,
                                   chosen_mode_idx=best_r)
        
        self.plot_pattern_vector(best_sensitivity, nodal_vectors=v_dominant,
                                 title="Dominant Folding Mechanism (Sensitivity Vector)",
                                 normalize=True)

        return best_sensitivity    
    
    def dead_analyze_sensitivity(self): # old function, may revery back to it. 
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
        best_mode_idx = None
        n_sv = len(singular_values)
        n_dof = Vh.shape[0]

        
        # We check every single mode to find the zeros
        # singular_values is sorted High -> Low, so the zeros are at the end.
        for i in range(n_dof):
            s_val = singular_values[i] if i < n_sv else 0.0 
            
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
                        best_mode_idx = i

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
        self.print_system_matrices(dihedral_jacobian, constraint_matrix, singular_values, Vh, best_sensitivity, chosen_mode_idx=best_mode_idx)
        
        
        return best_sensitivity

    def print_system_matrices(self, dihedral_jacobian, constraint_matrix, singular_values, Vh,
                              sensitivity_vector, mechanism_indices=None,
                              Q=None, A=None, U_sv=None, S_sv=None, Vt_sv=None,
                              v_dominant=None, t=None, chosen_mode_idx=None):
        """
        Comprehensive diagnostic report of every matrix and intermediate result
        produced by analyze_sensitivity, printed in pipeline order.

        Sections
        --------
        [1] Constraint Matrix  C
        [2] Dihedral Jacobian  J
        [3] Singular value spectrum of C  (full, classified)
        [4] Complete null space of C      (RBM + Mechanism rows)
        [5] Mechanism null space basis  Q
        [6] Fold angle matrix  A = J @ Q^T
        [7] SVD of A  (Σ, U, Vt)
        [8] Dominant nodal displacement  v*
        [9] Final sensitivity vector + M/V validation
        """
        W = 110   # report width

        # ── Label setup ───────────────────────────────────────────────────────
        col_labels   = []
        for n in self.nodes:
            col_labels.extend([f"N{n.id}_x", f"N{n.id}_y", f"N{n.id}_z"])
        bar_labels   = [f"Bar {i}"   for i in range(len(self.bars))]
        hinge_labels = [f"Hinge {i}" for i in range(len(self.hinges))]

        n_nodes  = len(self.nodes)
        n_bars   = len(self.bars)
        n_hinges = len(self.hinges)
        n_dofs   = 3 * n_nodes
        n_sv     = len(singular_values)
        n_dof    = Vh.shape[0]
        mech_set = set(mechanism_indices) if mechanism_indices else set()
        k_mech   = len(mech_set)
        null_indices = [i for i in range(n_dof)
                        if (singular_values[i] if i < n_sv else 0.0) < 1e-9]

        # ── Formatting helpers ────────────────────────────────────────────────
        def header(title):
            print("\n" + "═" * W)
            print(f"  {title}")
            print("═" * W)

        def subheader(title):
            bar = "─" * max(0, W - len(title) - 5)
            print(f"\n  ┌─ {title} {bar}")

        def print_matrix(matrix, r_labels, c_labels):
            """Print a 2-D matrix with row and column labels."""
            if len(matrix) == 0:
                print("    (empty)")
                return
            lw = max(len(str(l)) for l in r_labels) + 1
            cw = 9
            # header row
            print("    " + f"{'':>{lw}} ║ " + " │ ".join(f"{c:>{cw}}" for c in c_labels))
            print("    " + "─" * lw + "═╬═" + "═╪═".join("═" * cw for _ in c_labels))
            for lbl, row in zip(r_labels, matrix):
                vals = " │ ".join(
                    f"{'  0.0   ':>{cw}}" if abs(v) < 1e-9 else f"{v:>{cw}.4f}"
                    for v in row
                )
                print(f"    {lbl:>{lw}} ║ {vals}")

        # ══════════════════════════════════════════════════════════════════════
        header("ORIGAMI SENSITIVITY ANALYSIS — FULL DIAGNOSTIC REPORT")
        print(f"  Problem size:  {n_nodes} nodes  │  {n_dofs} DOFs  │  "
              f"{n_bars} bars  │  {n_hinges} hinges")
        print(f"  Null space:    {len(null_indices)} modes (σ < 1e-9)  │  "
              f"{k_mech} mechanism mode(s) selected")

        # ── [1] Constraint Matrix C ───────────────────────────────────────────
        header(f"[1]  CONSTRAINT MATRIX  C     shape: {n_bars} × {n_dofs}")
        print("  One row per bar.  C[i] · v = 0  means bar i doesn't stretch under displacement v.")
        print()
        print_matrix(constraint_matrix, bar_labels, col_labels)

        # ── [2] Dihedral Jacobian J ───────────────────────────────────────────
        header(f"[2]  DIHEDRAL JACOBIAN  J     shape: {n_hinges} × {n_dofs}")
        print("  One row per hinge.  J[i] · v = change in dihedral angle at hinge i.")
        print()
        print_matrix(dihedral_jacobian, hinge_labels, col_labels)

        # ── [3] Singular Value Spectrum of C ──────────────────────────────────
        header("[3]  SINGULAR VALUE SPECTRUM  of  C     (sorted high → low)")
        print("  Rows with σ ≈ 0 span the null space — the only legal nodal movements.")
        print()
        print(f"  {'Idx':>5} │ {'σ':>16} │ {'‖J·v‖₁ fold mag':>18} │  Classification")
        print(f"  {'─'*5}─┼─{'─'*16}─┼─{'─'*18}─┼─{'─'*42}")
        for i in range(n_dof):
            sv     = singular_values[i] if i < n_sv else 0.0
            v      = Vh[i, :]
            fmag   = np.sum(np.abs(dihedral_jacobian @ v))
            if sv >= 1e-9:
                cls    = "Constrained (resisted by bars)"
                fstr   = "—"
                sv_str = f"{sv:>16.6e}"
            else:
                sv_str = f"{sv:>16.2e}" if sv > 0 else f"{'0  (exact)':>16}"
                fstr   = f"{fmag:.6f}"
                if fmag < 1e-5:
                    cls = "NULL — Rigid Body / Spurious z-mode"
                elif i in mech_set:
                    cls = "★ NULL — MECHANISM (selected)"
                else:
                    cls = "NULL — Mechanism (not selected)"
            print(f"  {i:>5} │ {sv_str} │ {fstr:>18} │  {cls}")

        # ── [4] Complete Null Space of C ──────────────────────────────────────
        header(f"[4]  COMPLETE NULL SPACE  of  C     ({len(null_indices)} vectors, σ < 1e-9)")
        print("  Every row satisfies  C · v = 0.  Rows marked MECH are the selected mechanisms.")
        print()
        if null_indices:
            null_labels = []
            for idx in null_indices:
                fmag = np.sum(np.abs(dihedral_jacobian @ Vh[idx, :]))
                tag  = "MECH" if idx in mech_set else "RBM "
                null_labels.append(f"[{tag}] Mode {idx:>2}")
            print_matrix(Vh[null_indices, :], null_labels, col_labels)
        else:
            print("    No null space modes found.")

        # ── [5] Mechanism Null Space Basis Q ──────────────────────────────────
        if Q is not None and mechanism_indices is not None:
            header(f"[5]  MECHANISM NULL SPACE BASIS  Q     shape: {k_mech} × {n_dofs}")
            print("  Q = rows of Vh for mechanism modes only.")
            print("  Each row is a unit nodal-displacement vector that folds at least one hinge.")
            print()
            q_row_labels = [f"Mode {idx:>2}" for idx in mechanism_indices]
            print_matrix(Q, q_row_labels, col_labels)

        # ── [6] Fold Angle Matrix A = J @ Q^T ─────────────────────────────────
        if A is not None and mechanism_indices is not None:
            header(f"[6]  FOLD ANGLE MATRIX  A = J · Qᵀ     shape: {n_hinges} × {k_mech}")
            print("  A[:,r] = hinge fold angles when mechanism mode r is activated at unit amplitude.")
            print("  Large column entries → that mode drives significant folding at those hinges.")
            print()
            a_col_labels = [f"Mode {idx:>2}" for idx in mechanism_indices]
            print_matrix(A, hinge_labels, a_col_labels)

        # ── [7] SVD of A ──────────────────────────────────────────────────────
        if U_sv is not None and S_sv is not None and Vt_sv is not None:
            k = len(S_sv)
            header(f"[7]  SVD  of  A  →  A = U · Σ · Vᵀ     ({k} singular mode(s))")
            print("  σ_r  = fold efficiency of mode r: max fold output per unit null-space displacement.")
            print("  U[:,r] = fold pattern in hinge space  (what you see on the hinges).")
            print("  Vt[r,:] = mixing weights in mechanism space  (how to combine Q rows).")

            # Σ — singular values
            selected_r = chosen_mode_idx if chosen_mode_idx is not None else 0
            subheader("Σ  —  Singular Values  (fold efficiency, dominant first)")
            print(f"    {'Rank':>6} │ {'σ':>14} │  Note")
            print(f"    {'─'*6}─┼─{'─'*14}─┼─{'─'*35}")
            for r, sv in enumerate(S_sv):
                note = f"  ← SELECTED (best M/V alignment)" if r == selected_r else ""
                print(f"    {r:>6} │ {sv:>14.6f} │{note}")

            # U — fold patterns in hinge space
            subheader("U  —  Left Singular Vectors  (fold patterns in hinge space)")
            print("    Column r = fold pattern for the r-th principal mode.")
            print(f"    best_sensitivity = U[:,{selected_r}] · σ_{selected_r}   (selected mode)\n")
            u_col_labels = [f"σ_{r}={S_sv[r]:.3f}" for r in range(k)]
            print_matrix(U_sv, hinge_labels, u_col_labels)

            # Vt — mixing weights in mechanism space
            subheader("Vᵀ  —  Right Singular Vectors  (mixing weights in mechanism space)")
            print("    Row r = weights applied to Q rows to produce the r-th fold pattern.")
            print("    v_dominant = Qᵀ · Vt[0,:]  →  the actual nodal displacement vector.\n")
            vt_row_labels = [f"SVD mode {r}" for r in range(k)]
            q_short_labels = ([f"Q_m{idx}" for idx in mechanism_indices]
                              if mechanism_indices else [f"q{r}" for r in range(Vt_sv.shape[1])])
            print_matrix(Vt_sv, vt_row_labels, q_short_labels)

        # ── [8] Dominant Nodal Displacement v* ────────────────────────────────
        if v_dominant is not None:
            header("[8]  DOMINANT NODAL DISPLACEMENT  v*  =  Qᵀ · Vt[0,:]")
            print("  The physical nodal displacement that produces best_sensitivity.")
            print("  Verify:  C · v* = 0  (all bars satisfied).  J · v* = best_sensitivity.")
            print()
            v3 = np.array(v_dominant).reshape(-1, 3)
            print(f"  {'Node':>6} │ {'dx':>13} │ {'dy':>13} │ {'dz':>13} │ {'‖d‖':>10}")
            print(f"  {'─'*6}─┼─{'─'*13}─┼─{'─'*13}─┼─{'─'*13}─┼─{'─'*10}")
            for i, (dx, dy, dz) in enumerate(v3):
                mag  = float(np.sqrt(dx**2 + dy**2 + dz**2))
                fmtv = lambda x: f"{'  0.0':>13}" if abs(x) < 1e-9 else f"{x:>13.6f}"
                print(f"  {i:>6} │ {fmtv(dx)} │ {fmtv(dy)} │ {fmtv(dz)} │ {mag:>10.6f}")

            # Quick verification: J @ v* should equal best_sensitivity
            if sensitivity_vector is not None:
                jv = dihedral_jacobian @ v_dominant
                max_err = np.max(np.abs(jv - np.array(sensitivity_vector)))
                print(f"\n  Verification  ‖J·v* − best_sensitivity‖∞ = {max_err:.2e}"
                      f"  {'✓ consistent' if max_err < 1e-9 else '⚠ check sign flip'}")

        # ── [9] Final Sensitivity + M/V Validation ────────────────────────────
        _r = chosen_mode_idx if chosen_mode_idx is not None else 0
        header(f"[9]  FINAL SENSITIVITY VECTOR  =  U[:,{_r}] · σ_{_r}  (sign-corrected, best M/V alignment)")
        print("  These are the hinge fold sensitivities for the selected physical mechanism.")
        print()
        print(f"  {'Hinge':>7} │ {'Assign':>7} │ {'Target t':>10} │ {'Sensitivity':>14} │ {'M/V Check':>12}")
        print(f"  {'─'*7}─┼─{'─'*7}─┼─{'─'*10}─┼─{'─'*14}─┼─{'─'*12}")
        all_match = True
        for i, h in enumerate(self.hinges):
            asgn  = h.fold_assignment
            t_val = t[i] if t is not None else float('nan')
            s_val = sensitivity_vector[i] if sensitivity_vector is not None else float('nan')
            if asgn == 'M':
                ok = s_val >= 0
            elif asgn == 'V':
                ok = s_val <= 0
            else:
                ok = None
            check = ("✓" if ok else "✗ MISMATCH") if ok is not None else "— (unassigned)"
            if ok is False:
                all_match = False
            t_str = f"{t_val:>+10.1f}" if t is not None else f"{'N/A':>10}"
            print(f"  {i:>7} │ {asgn:>7} │ {t_str} │ {s_val:>+14.6f} │ {check:>12}")

        if sensitivity_vector is not None and t is not None:
            ns = np.linalg.norm(sensitivity_vector)
            nt = np.linalg.norm(t)
            if ns > 1e-12 and nt > 1e-12:
                cos_sim = np.dot(sensitivity_vector, t) / (ns * nt)
                quality = ('excellent' if cos_sim > 0.99 else
                           'good'      if cos_sim > 0.90 else
                           'moderate'  if cos_sim > 0.50 else 'poor')
                print(f"\n  M/V cosine alignment:  {cos_sim:.6f}  ({quality})")
        verdict = "  ✓ All folds match M/V assignments." if all_match else \
                  "  ⚠ WARNING: One or more folds are inconsistent with M/V assignments."
        print(verdict)

        print("\n" + "═" * W + "\n")
   
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

        # for panel in self.panels:
        #     # Connect every node to every other node in this specific panel
        #     for node_a, node_b in itertools.combinations(panel.nodes,2):

        #         # Sort IDs to ensure Edge(1,2) = Edge (2,1)
        #         edge_id = tuple(sorted((node_a.id, node_b.id)))

        #         if edge_id not in unique_edges:
        #             unique_edges.add(edge_id)
        #             #Add the rigid bar
        #             bars.append(BarElement(node_a, node_b))

        for panel in self.panels:
            nodes = panel.nodes
            n = len(nodes)

            # 1. All perimeter edges
            for i in range(n):
                a, b = nodes[i], nodes[(i + 1) % n]
                edge_id = tuple(sorted((a.id, b.id)))
                if edge_id not in unique_edges:
                    unique_edges.add(edge_id)
                    bars.append(BarElement(a, b))

            # 2. Fan diagonals from node 0 to nodes 2, 3, ..., n-2
            #    (node 0→1 and node 0→n-1 are already perimeter edges)
            for i in range(2, n - 1):
                a, b = nodes[0], nodes[i]
                edge_id = tuple(sorted((a.id, b.id)))
                if edge_id not in unique_edges:
                    unique_edges.add(edge_id)
                    bars.append(BarElement(a, b))

        return bars

    def generate_hinges(self):
        """ Creates a hinge where the .fold file says there should be a hinge. If there is no .fold file,
          it creates a hinge between every edge that has 2 panels on it."""

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

            # Collect ALL non-hinge nodes for each panel (centroid-based Jacobian)
            wing_nodes_1 = [n for n in panel1.nodes if n.id not in edge_key]
            wing_nodes_2 = [n for n in panel2.nodes if n.id not in edge_key]

            hinges.append(HingeElement(wing_nodes_1, node_j, node_k, wing_nodes_2, assignment))
        
        return hinges
    
    def plot_pattern_vector(self, sensitivity_vector=None, nodal_vectors=None, vector_scale=1.0, vector_color='green', show_node_labels=True, show_hinge_labels=True, title="Pattern", normalize=False):
        """
        Plot the origami pattern with:
          - Bars drawn in light grey
          - Hinges color-coded by sensitivity (blue = Mountain/+, red = Valley/-)
          - Raw sensitivity value labeled on every hinge (always shown regardless of normalize)
          - Optional nodal displacement vectors drawn as quiver arrows

        Parameters
        ----------
        sensitivity_vector : array-like, optional
            One value per hinge. Drives hinge color and label.
        nodal_vectors : array-like, optional
            Flat 1D array of length 3*n_nodes (or shape (n_nodes, 3)).
            Drawn as quiver arrows at each node to show the null-space
            displacement direction.
        vector_scale : float
            Arrow length scale for the quiver plot.
        vector_color : str
            Color for the quiver arrows.
        normalize : bool
            If True, the colormap is normalized to [-1, +1].
            Hinge labels always show the original (un-normalized) sensitivity.
        """
        plt.close('all')
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- Sensitivity data ---
        # raw_sens  → original values; always used for hinge labels
        # plot_sens → may be scaled to [-1,+1]; drives the colormap only
        raw_sens = np.zeros(len(self.hinges))
        if sensitivity_vector is not None:
            raw_sens = np.array(sensitivity_vector).flatten()

        max_abs_val = np.max(np.abs(raw_sens))
        if max_abs_val < 1e-12:
            max_abs_val = 1.0

        if normalize:
            plot_sens = raw_sens / max_abs_val
            limit = 1.0
        else:
            plot_sens = raw_sens.copy()
            limit = max_abs_val

        # --- Color map (blue = positive/mountain, red = negative/valley) ---
        cmap = plt.cm.coolwarm_r
        cnorm = plt.Normalize(-limit, limit)

        # --- Nodes & Bars ---
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

        # --- Nodal displacement vectors (quiver) ---
        if nodal_vectors is not None:
            nv = np.array(nodal_vectors)
            # Accept a flat 1D array and reshape it to (N, 3)
            if nv.ndim == 1 and len(nv) == 3 * len(self.nodes):
                nv = nv.reshape(-1, 3)
            if nv.ndim == 2 and len(nv) == len(self.nodes) and nv.shape[1] == 3:
                ax.quiver(xs, ys, zs,
                          nv[:, 0], nv[:, 1], nv[:, 2],
                          color=vector_color,
                          length=vector_scale,
                          normalize=False,
                          arrow_length_ratio=0.15)
            else:
                print(f"Warning: nodal_vectors shape {nv.shape} doesn't match "
                      f"({len(self.nodes)}, 3). Skipping quiver plot.")

        # --- Hinges: color-coded lines + sensitivity labels ---
        for h_id, h in enumerate(self.hinges):
            p_j = h.node_j.coordinates
            p_k = h.node_k.coordinates
            color = cmap(cnorm(plot_sens[h_id]))

            ax.plot([p_j[0], p_k[0]], [p_j[1], p_k[1]], [p_j[2], p_k[2]],
                    color=color, linestyle='-', linewidth=3.0, alpha=0.9)

            if show_hinge_labels:
                mid = (p_j + p_k) / 2
                # Always show the raw value so the user sees the actual physics,
                # even when the colormap has been normalized.
                label_text = f"H{h_id}\n{raw_sens[h_id]:.3f}"
                ax.text(mid[0], mid[1], mid[2], label_text,
                        color=color, fontsize=9, fontweight='bold',
                        path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])

        # --- Axis limits ---
        all_coords = np.array([xs, ys, zs])
        max_range = np.ptp(all_coords, axis=1).max() / 2.0
        mid_x = np.mean(all_coords[0])
        mid_y = np.mean(all_coords[1])
        mid_z = np.mean(all_coords[2])
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_title(title)

        # --- Colorbar ---
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar_label = 'Normalized Mode (-1 to +1)' if normalize else 'Hinge Sensitivity (rad)'
        cbar.set_label(cbar_label, rotation=270, labelpad=15)

        plt.show()
