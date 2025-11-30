import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import json
import time
import bittensor as bt
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import nova_ph2
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening

from molecules import generate_valid_random_molecules_batch

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")


target_models = []
antitarget_models = []

def get_config(input_file: str = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


def initialize_models(config: dict):
    """Initialize separate model instances for each target and antitarget sequence."""
    global target_models, antitarget_models
    target_models = []
    antitarget_models = []
    
    for seq in config["target_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        target_models.append(wrapper)
    
    for seq in config["antitarget_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        antitarget_models.append(wrapper)


# ---------- PARALLEL SCORING FUNCTIONS ----------
def score_single_target_model(model_idx: int, smiles_list: List[str]) -> tuple:
    """Score molecules with a single target model."""
    try:
        model = target_models[model_idx]
        result = model.score_molecules(smiles_list)
        scores = result['predicted_binding_affinity'].tolist()
        smiles_dict = getattr(model, 'smiles_dict', {})
        return (model_idx, scores, smiles_dict)
    except Exception as e:
        bt.logging.error(f"Target model {model_idx} scoring error: {e}")
        return (model_idx, [0.0] * len(smiles_list), {})


def score_single_antitarget_model(model_idx: int, smiles_list: List[str], smiles_dict: dict) -> tuple:
    """Score molecules with a single antitarget model."""
    try:
        model = antitarget_models[model_idx]
        
        # Set smiles data from target models
        model.smiles_list = smiles_list
        model.smiles_dict = smiles_dict
        
        # Create loader and run virtual screening
        model.create_screen_loader(model.protein_dict, model.smiles_dict)
        model.screen_df = virtual_screening(
            model.screen_df, 
            model.model, 
            model.screen_loader,
            os.getcwd(),
            save_interpret=False,
            ligand_dict=model.smiles_dict, 
            device=model.device,
            save_cluster=False,
        )
        
        scores = model.screen_df['predicted_binding_affinity'].tolist()
        return (model_idx, scores)
    except Exception as e:
        bt.logging.error(f"Antitarget model {model_idx} scoring error: {e}")
        return (model_idx, [0.0] * len(smiles_list))


def parallel_score_molecules(smiles_series: pd.Series, config: dict) -> tuple:
    """
    Score molecules against all target and antitarget models in parallel.
    Returns (target_scores, antitarget_scores) as pandas Series.
    """
    global target_models, antitarget_models
    
    smiles_list = smiles_series.tolist()
    n_molecules = len(smiles_list)
    
    if n_molecules == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    target_scores_by_model = {}
    antitarget_scores_by_model = {}
    shared_smiles_dict = {}
    
    try:
        # Step 1: Score all TARGET models in parallel
        with ThreadPoolExecutor(max_workers=len(target_models)) as executor:
            target_futures = {
                executor.submit(score_single_target_model, idx, smiles_list): idx 
                for idx in range(len(target_models))
            }
            
            for future in as_completed(target_futures):
                model_idx = target_futures[future]
                try:
                    idx, scores, smiles_dict = future.result()
                    target_scores_by_model[idx] = scores
                    shared_smiles_dict.update(smiles_dict)
                except Exception as e:
                    bt.logging.error(f"Target model {model_idx} future failed: {e}")
                    target_scores_by_model[model_idx] = [0.0] * n_molecules
        
        # Step 2: Score all ANTITARGET models in parallel
        with ThreadPoolExecutor(max_workers=len(antitarget_models)) as executor:
            antitarget_futures = {
                executor.submit(score_single_antitarget_model, idx, smiles_list, shared_smiles_dict): idx 
                for idx in range(len(antitarget_models))
            }
            
            for future in as_completed(antitarget_futures):
                model_idx = antitarget_futures[future]
                try:
                    idx, scores = future.result()
                    antitarget_scores_by_model[idx] = scores
                except Exception as e:
                    bt.logging.error(f"Antitarget model {model_idx} future failed: {e}")
                    antitarget_scores_by_model[model_idx] = [0.0] * n_molecules
        
        # Step 3: Average scores across models
        if target_scores_by_model:
            target_array = np.array([target_scores_by_model[i] for i in sorted(target_scores_by_model.keys())], 
                                   dtype=np.float32)
            avg_target_scores = target_array.mean(axis=0)
            target_series = pd.Series(avg_target_scores)
        else:
            target_series = pd.Series([0.0] * n_molecules)
        
        if antitarget_scores_by_model:
            antitarget_array = np.array([antitarget_scores_by_model[i] for i in sorted(antitarget_scores_by_model.keys())], 
                                       dtype=np.float32)
            avg_antitarget_scores = antitarget_array.mean(axis=0)
            antitarget_series = pd.Series(avg_antitarget_scores)
        else:
            antitarget_series = pd.Series([0.0] * n_molecules)
        
        return target_series, antitarget_series
        
    except Exception as e:
        bt.logging.error(f"Parallel scoring failed: {e}")
        return pd.Series([0.0] * n_molecules), pd.Series([0.0] * n_molecules)


def build_component_weights(top_pool: pd.DataFrame, rxn_id: int) -> Dict[str, Dict[int, float]]:
    """
    Build component weights based on scores of molecules containing them.
    Returns dict with 'A', 'B', 'C' keys mapping to {component_id: weight}
    """
    weights = {'A': defaultdict(float), 'B': defaultdict(float), 'C': defaultdict(float)}
    counts = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}
    
    if top_pool.empty:
        return weights
    
    # Extract component IDs and scores
    for _, row in top_pool.iterrows():
        name = row['name']
        score = row['score']
        parts = name.split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                weights['A'][A_id] += max(0, score)  # Only positive contributions
                weights['B'][B_id] += max(0, score)
                counts['A'][A_id] += 1
                counts['B'][B_id] += 1
                
                if len(parts) > 4:
                    C_id = int(parts[4])
                    weights['C'][C_id] += max(0, score)
                    counts['C'][C_id] += 1
            except (ValueError, IndexError):
                continue
    
    # Normalize by count and add smoothing
    for role in ['A', 'B', 'C']:
        for comp_id in weights[role]:
            if counts[role][comp_id] > 0:
                weights[role][comp_id] = weights[role][comp_id] / counts[role][comp_id] + 0.1  # Smoothing
    
    return weights


def select_diverse_elites(top_pool: pd.DataFrame, n_elites: int, min_score_ratio: float = 0.7) -> pd.DataFrame:
    """
    Select diverse elite molecules: top by score, but ensure diversity in component space.
    """
    if top_pool.empty or n_elites <= 0:
        return pd.DataFrame()
    
    # Take top candidates (more than needed for diversity filtering)
    top_candidates = top_pool.head(min(len(top_pool), n_elites * 3))
    if len(top_candidates) <= n_elites:
        return top_candidates
    
    # Score threshold: at least min_score_ratio of max score
    max_score = top_candidates['score'].max()
    threshold = max_score * min_score_ratio
    candidates = top_candidates[top_candidates['score'] >= threshold]
    
    # Select diverse set: prefer molecules with different components
    selected = []
    used_components = {'A': set(), 'B': set(), 'C': set()}
    
    # First, add top scorer
    if not candidates.empty:
        top_idx = candidates.index[0]
        top_row = candidates.iloc[0]
        selected.append(top_idx)
        parts = top_row['name'].split(":")
        if len(parts) >= 4:
            try:
                used_components['A'].add(int(parts[2]))
                used_components['B'].add(int(parts[3]))
                if len(parts) > 4:
                    used_components['C'].add(int(parts[4]))
            except (ValueError, IndexError):
                pass
    
    # Then add diverse molecules
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx in selected:
            continue
        
        parts = row['name'].split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                C_id = int(parts[4]) if len(parts) > 4 else None
                
                # Prefer molecules with new components
                is_diverse = (A_id not in used_components['A'] or 
                             B_id not in used_components['B'] or
                             (C_id is not None and C_id not in used_components['C']))
                
                if is_diverse or len(selected) < n_elites * 0.5:  # Always take some top ones
                    selected.append(idx)
                    used_components['A'].add(A_id)
                    used_components['B'].add(B_id)
                    if C_id is not None:
                        used_components['C'].add(C_id)
            except (ValueError, IndexError):
                # If parsing fails, just add it
                if len(selected) < n_elites:
                    selected.append(idx)
    
    # Fill remaining slots with top scorers
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx not in selected:
            selected.append(idx)
    
    return candidates.loc[selected[:n_elites]] if selected else candidates.head(n_elites)


def main(config: dict):
    n_samples = config["num_molecules"] * 5
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score", "Target", "Anti"])
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    iteration = 0
    mutation_prob = 0.1
    elite_frac = 0.25
    prev_avg_score = None
    score_improvement_rate = 0.0
    seen_inchikeys = set()
    start = time.time()

    n_samples_first_iteration = n_samples if config["allowed_reaction"] == "rxn:5" else n_samples*4
    
    bt.logging.info(f"[Miner] Starting optimization: {len(target_models)} target, {len(antitarget_models)} antitarget models")
    
    # File writing control
    FILE_WRITE_DELAY = 25 * 60  # 25 minutes in seconds
    file_writing_enabled = False
    
    while time.time() - start < 1800:
        iteration += 1
        start_time = time.time()
        
        # Enable file writing after 25 minutes
        elapsed_time = time.time() - start
        if not file_writing_enabled and elapsed_time >= FILE_WRITE_DELAY:
            file_writing_enabled = True
            bt.logging.info(f"[Miner] File writing enabled at {elapsed_time/60:.1f} minutes")
        
        # Build component weights from top pool for score-guided sampling
        component_weights = build_component_weights(top_pool, rxn_id) if not top_pool.empty else None
        
        # Select diverse elites (not just top by score)
        elite_df = select_diverse_elites(top_pool, min(100, len(top_pool))) if not top_pool.empty else pd.DataFrame()
        elite_names = elite_df["name"].tolist() if not elite_df.empty else None
        
        # Adaptive sampling: adjust based on score improvement
        if prev_avg_score is not None and not top_pool.empty:
            current_avg = top_pool['score'].mean()
            score_improvement_rate = (current_avg - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
            
            # If improving well, increase exploitation; if stagnating, increase exploration
            if score_improvement_rate > 0.01:  # Good improvement
                elite_frac = min(0.7, elite_frac * 1.1)
                mutation_prob = max(0.05, mutation_prob * 0.95)
            elif score_improvement_rate < -0.01:  # Declining
                elite_frac = max(0.2, elite_frac * 0.9)
                mutation_prob = min(0.4, mutation_prob * 1.1)
        
        data = generate_valid_random_molecules_batch(
            rxn_id, 
            n_samples=n_samples_first_iteration if iteration == 1 else n_samples, 
            db_path=DB_PATH, 
            subnet_config=config, 
            batch_size=300, 
            elite_names=elite_names, 
            elite_frac=elite_frac, 
            mutation_prob=mutation_prob, 
            avoid_inchikeys=seen_inchikeys, 
            component_weights=component_weights
        )
        
        if data.empty:
            continue

        try:
            filtered_data = data[~data['InChIKey'].isin(seen_inchikeys)]

            dup_ratio = (len(data) - len(filtered_data)) / max(1, len(data))
            if dup_ratio > 0.6:
                mutation_prob = min(0.5, mutation_prob * 1.5)
                elite_frac = max(0.2, elite_frac * 0.8)
            elif dup_ratio < 0.2 and not top_pool.empty:
                mutation_prob = max(0.05, mutation_prob * 0.9)
                elite_frac = min(0.8, elite_frac * 1.1)

            data = filtered_data

        except Exception as e:
            bt.logging.error(f"Deduplication failed: {e}")

        data = data.reset_index(drop=True)
        
        if data.empty:
            continue
        
        # PARALLEL SCORING
        target_scores, antitarget_scores = parallel_score_molecules(data['smiles'], config)
        
        data['Target'] = target_scores
        data['Anti'] = antitarget_scores
        data['score'] = data['Target'] - (config['antitarget_weight'] * data['Anti'])
        
        seen_inchikeys.update([k for k in data["InChIKey"].tolist() if k])
        
        # Keep Target and Anti columns for statistics
        total_data = data[["name", "smiles", "InChIKey", "score", "Target", "Anti"]]
        top_pool = pd.concat([top_pool, total_data])
        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
        top_pool = top_pool.sort_values(by="score", ascending=False)
        top_pool = top_pool.head(config["num_molecules"])
        
        # Calculate statistics
        avg_score = top_pool['score'].mean()
        max_score = top_pool['score'].max()
        min_score = top_pool['score'].min()
        
        # Update previous average
        prev_avg_score = avg_score
        
        # Log every 5 iterations or if significant improvement
        if iteration % 5 == 0 or (iteration > 1 and max_score > 2.45):
            bt.logging.info(f"[Miner] Iter {iteration}: Avg={avg_score:.4f}, Max={max_score:.4f}, Min={min_score:.4f}")
        
        # Save results ONLY after 25 minutes
        if file_writing_enabled:
            top_entries = {"molecules": top_pool["name"].tolist()}
            tmp_path = os.path.join(OUTPUT_DIR, "result.json.tmp")
            final_path = os.path.join(OUTPUT_DIR, "result.json")
            with open(tmp_path, "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, final_path)
    
    # Final summary
    bt.logging.info(f"[Miner] ===== COMPLETE =====")
    bt.logging.info(f"[Miner] Iterations: {iteration}, Avg: {avg_score:.4f}, Max: {max_score:.4f}")
    bt.logging.info(f"[Miner] Total molecules explored: {len(seen_inchikeys)}")
    
    # Final save (ensure we write at the end)
    top_entries = {"molecules": top_pool["name"].tolist()}
    tmp_path = os.path.join(OUTPUT_DIR, "result.json.tmp")
    final_path = os.path.join(OUTPUT_DIR, "result.json")
    with open(tmp_path, "w") as f:
        json.dump(top_entries, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, final_path)


if __name__ == "__main__":
    config = get_config()
    start_time_1 = time.time()
    initialize_models(config)
    bt.logging.info(f"[Miner] Model initialization: {time.time() - start_time_1:.2f}s")
    main(config)
