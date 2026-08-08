#!/usr/bin/env python3
"""
debug_moqnas.py

A debugging script for MoQNAS (Multi‐Objective Quantum‐Inspired NAS), modeled
after debug_nsga.py. Uses a dummy evaluation function that writes a file per
candidate and returns random metric values for the specified objectives.

Usage:
    python3 debug_moqnas.py
"""

import os
import shutil
import logging
import numpy as np

from moqnas import MOQNAS
from cnn.master import setup_additional_params  # for creating per‐candidate folders

# will be set in main()
EXP_DIR = None

# --- 1) Dummy eval function --------------------------------------
def dummy_eval_func(eval_dp, eval_nets, generation):
    """
    eval_dp: list of dicts including 'candidate_id'
    eval_nets: list of decoded architectures (lists or similar)
    generation: current generation index

    Returns a list of dicts, each mapping metric_name→value. We assume
    objectives = ["accuracy", "params_count", "latency"].
    """
    global EXP_DIR
    results = []
    for idx, net in enumerate(eval_nets):
        # retrieve candidate_id that MoQNAS embedded in eval_dp
        cid = str(eval_dp[idx].get('candidate_id', idx))

        # 1) create that individual's folder just like real NAS does:
        params = {'experiment_path': EXP_DIR}
        # this creates: results/gen_<generation>/cand_<generation>_<cid>/
        id_num = f"{generation}_{cid}"
        setup_additional_params(params, id_num)

        # 2) write a dummy file so you can see that it ran
        dummy_file = os.path.join(params['model_path'], 'dummy.txt')
        with open(dummy_file, 'w') as f:
            f.write(f"Gen {generation}, Cand {cid}\n")

        # 3) produce random metrics for each of the objectives
        acc = 0.5 + 0.5 * np.random.rand()          # accuracy between 0.5 and 1.0
        params_count = len(net) * (1e4 + 5e3 * np.random.rand())
        latency = 1e-3 + 1e-3 * np.random.rand()     # between 0.001 and 0.002
        results.append({
            'accuracy': float(acc),
            'params_count': float(params_count),
            'latency': float(latency)
        })

    return results

# --- 2) Debug script ----------------------------------------------------
def main():
    global EXP_DIR

    # 1) fresh experiment dir
    exp_dir = "./debug_moqnas_exp"
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    EXP_DIR = exp_dir  # so dummy_eval_func can see it

    # 2) logging
    log_file = os.path.join(exp_dir, "debug.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console)
    logging.info("=== MoQNAS Debug Run ===")

    # 3) instantiate MoQNAS
    objectives = ["accuracy", "params_count", "latency"]
    moqnas = MOQNAS(
        eval_func=dummy_eval_func,
        experiment_path=exp_dir,
        objectives=objectives,
        log_file=log_file,
        log_level="DEBUG",
        data_file=os.path.join(exp_dir, "data.pkl")
    )

    # 4) initialize with toy parameters
    fn_list = ["conv3x3", "conv5x5", "maxpool2x2", "avgpool2x2", "no_op"]
    initial_probs = [1.0 / len(fn_list)] * len(fn_list)
    params_ranges = {}  # no hyperparameters for this dummy test

    moqnas.initialize_moqnas(
        num_quantum_ind=3,            # 3 quantum individuals
        params_ranges=params_ranges,
        repetition=2,                 # 2 classical per quantum → pop_size = 3*2=6
        pop_size=6,                   # keep 6 classical networks per generation
        max_generations=5,            # run for 5 generations
        update_quantum_gen=2,         # update quantum every 2 gens
        replace_method="best",        # unused in MoQNAS, but required
        fn_list=fn_list,
        initial_probs=initial_probs,
        update_quantum_rate=0.1,      # moderate quantum‐update intensity
        max_num_nodes=4,              # max 4 nodes per network chromosome
        reducing_fns_list=["maxpool2x2", "avgpool2x2"],  # penalize these if too many
        patience=3,
        early_stopping=False,
        save_data_freq=0,
        penalize_number=1,
        crossover_frequency=2,
        en_pop_crossover=True,
        pop_crossover_rate=0.5,
        pop_crossover_method="hux",
    )

    # 5) run evolution
    final_nets, final_fits = moqnas.evolve()

    # 6) report
    print("\nFinal Pareto‐optimal population (chromosomes + fitnesses):")
    for i, (ind, fit) in enumerate(zip(final_nets, final_fits)):
        print(f"  Ind {i}: chrom={ind.tolist()}  →  "
            f"(accuracy={fit[0]:.3f}, params_count={fit[1]:.0f}, latency={fit[2]:.4f})")

    logging.info("=== Debug run complete ===")

if __name__ == "__main__":
    main()
