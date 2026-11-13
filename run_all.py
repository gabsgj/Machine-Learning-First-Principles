"""
Unified Test Runner & Benchmark Suite
=====================================
Orchestrates test validation, benchmark comparisons, and plot generation across
all 14 machine learning modules in the repository.

Usage:
    python3 run_all.py --test        # Runs all unit tests
    python3 run_all.py --benchmark   # Runs all module benchmark scripts and generates plots
    python3 run_all.py --all         # Runs both test suite and benchmark suite
"""

import sys
import os
import time
import subprocess
import argparse


MODULES = [
    ("01_data_preprocessing", "01_data_preprocessing/compare_preprocessing.py"),
    ("02_linear_regression", "02_linear_regression/compare_linear_regression.py"),
    ("03_polynomial_regression", "03_polynomial_regression/compare_polynomial_regression.py"),
    ("04_logistic_regression", "04_logistic_regression/compare_logistic_regression.py"),
    ("05_k_nearest_neighbors", "05_k_nearest_neighbors/compare_knn.py"),
    ("06_decision_trees_id3", "06_decision_trees_id3/compare_decision_tree.py"),
    ("07_support_vector_machines", "07_support_vector_machines/compare_svm.py"),
    ("08_naive_bayes", "08_naive_bayes/compare_naive_bayes.py"),
    ("09_parameter_estimation_mle_map", "09_parameter_estimation_mle_map/compare_mle_map.py"),
    ("10_neural_networks_mlp", "10_neural_networks_mlp/compare_mlp_activations.py"),
    ("11_unsupervised_clustering", "11_unsupervised_clustering/compare_clustering.py"),
    ("12_resampling_and_evaluation", "12_resampling_and_evaluation/compare_resampling.py"),
    ("13_ensemble_learning", "13_ensemble_learning/compare_ensemble.py"),
    ("14_bias_variance_tradeoff", "14_bias_variance_tradeoff/compare_bias_variance.py"),
]


def run_unit_tests():
    print("======================================================================")
    print("  RUNNING COMPLETE MACHINE LEARNING UNIT TEST SUITE")
    print("======================================================================")
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_benchmarks():
    print("======================================================================")
    print("  RUNNING ALL MODULE BENCHMARKS & GENERATING VISUAL DIAGNOSTICS")
    print("======================================================================")
    
    os.makedirs("plots", exist_ok=True)
    summary_results = []

    for mod_name, script_path in MODULES:
        print(f"\n>>> Executing Benchmark: {mod_name} ...")
        t0 = time.time()
        
        # Execute script within its directory or from root
        res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        elapsed = time.time() - t0
        
        status = "PASSED" if res.returncode == 0 else "FAILED"
        summary_results.append((mod_name, status, f"{elapsed:.2f}s"))

        if res.returncode == 0:
            print(f"    [OK] Finished in {elapsed:.2f}s")
        else:
            print(f"    [FAIL] Error occurred in {mod_name}:\n{res.stderr}")

    print("\n======================================================================")
    print("  BENCHMARK EXECUTION SCOREBOARD")
    print("======================================================================")
    print(f"{'Module':<35} {'Status':<12} {'Elapsed Time':<12}")
    print("-" * 62)
    for mod, stat, dur in summary_results:
        print(f"{mod:<35} {stat:<12} {dur:<12}")

    plots = [p for p in os.listdir("plots") if p.endswith(".png")]
    print(f"\nGenerated {len(plots)} publication plots in 'plots/' directory:")
    for p in sorted(plots):
        print(f"  - plots/{p}")


def main():
    parser = argparse.ArgumentParser(description="Master ML Repository Runner")
    parser.add_argument("--test", action="store_true", help="Run the unit test suite")
    parser.add_argument("--benchmark", action="store_true", help="Run all benchmarks and generate plots")
    parser.add_argument("--all", action="store_true", help="Run both tests and benchmarks")

    args = parser.parse_args()

    # If no flags specified, default to running all
    if not (args.test or args.benchmark or args.all):
        args.all = True

    if args.test or args.all:
        success = run_unit_tests()
        if not success:
            print("\n[ALERT] Unit tests failed. Aborting benchmark execution.")
            sys.exit(1)

    if args.benchmark or args.all:
        run_benchmarks()


if __name__ == "__main__":
    main()
