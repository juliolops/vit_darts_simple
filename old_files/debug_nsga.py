#!/usr/bin/env python3
import os
import shutil
import logging
import numpy as np

from nsga2 import NSGA2
from cnn.master import setup_additional_params  # <-- import your folder‐maker

# will be set in main()
EXP_DIR = None

# --- 1) Dummy eval function --------------------------------------
def dummy_eval_func(eval_dp, eval_nets, generation):
    """
    eval_dp: list of dicts including 'candidate_id'
    eval_nets: list of decoded architectures (lists of layer names)
    generation: current generation index
    Returns a list of (accuracy, num_params, inference_time)
    """
    global EXP_DIR
    results = []
    for idx, net in enumerate(eval_nets):
        # pull the id our NSGA code stuck into eval_dp
        cid = str(eval_dp[idx].get('candidate_id', idx))

        # 1) create that individual's folder just like real GA does:
        params = {'experiment_path': EXP_DIR}
        # this will mkdir results/gen_<generation>/cand_<generation>_<cid>/
        id_num = f"{generation}_{cid}"
        setup_additional_params(params, id_num)

        # 2) write a dummy file so you can see it ran
        dummy_file = os.path.join(params['model_path'], 'dummy.txt')
        with open(dummy_file, 'w') as f:
            f.write(f"Gen {generation}, Cand {cid}\n")

        # now produce random fitnesses
        acc = 0.5 + 0.5 * np.random.rand()
        params_count = len(net) * (1e4 + 5e3 * np.random.rand())
        inf_time = 1e-3 + 1e-3 * np.random.rand()
        results.append((acc, params_count, inf_time))

    return results

# --- 2) Debug script ----------------------------------------------------
def main():
    global EXP_DIR

    # 1) fresh experiment dir
    exp_dir = "./debug_exp"
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
    logging.info("=== NSGA2 Debug Run ===")

    # 3) instantiate NSGA2
    nsga = NSGA2(
        eval_func=dummy_eval_func,
        experiment_path=exp_dir,
        log_file=log_file,
        log_level="DEBUG",
        data_file=os.path.join(exp_dir, "data.pkl")
    )

    # 4) initialize with toy parameters
    fn_list = ["conv3x3", "conv5x5", "maxpool2x2", "avgpool2x2", "no_op"]
    params_range = {
        "": {}
    }
    nsga.initialize_ga(
        population_size=20,
        num_generations=10,
        max_num_nodes=8,
        fn_list=fn_list,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elitism=True,
        patience=20,
        params_ranges=params_range,
    )

    # 5) run evolution
    pop, fits = nsga.evolve()

    # 6) report
    print("\nFinal population (chromosomes + fitness):")
    for i, (ind, fit) in enumerate(zip(pop, fits)):
        print(f"  Ind {i}: chrom={ind.tolist()}  →  "
            f"(acc={fit[0]:.3f}, params={fit[1]:.0f}, time={fit[2]:.4f})")

    logging.info("=== Debug run complete ===")

if __name__ == "__main__":
    main()
