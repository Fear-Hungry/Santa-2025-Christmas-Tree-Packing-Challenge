#!/usr/bin/env python3
import argparse
import os
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser(description="Run ensemble sweep")
    parser.add_argument("--output-dir", default="runs/intervention_1", help="Output directory")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    commands = []

    # 1. Tessellation Solver Variants
    angles_list = [
        "0",
        "0,90",
        "0,45,90",
        "0,15,30,45,60,75",
        "0,10,20,30,40,50,60"
    ]
    
    # Run Tessellation
    for seed in range(1, args.seeds + 1):
        for i, angles in enumerate(angles_list):
            out_file = f"{args.output_dir}/tess_s{seed}_a{i}.csv"
            if not os.path.exists(out_file):
                cmd = f"./bin/solver_tessellation --seed {seed} --angles {angles} --output {out_file} --preset fast"
                commands.append(cmd)

    # 2. Tile Solver Variants
    # Varied shifts and lattice types
    shifts = ["0.0,0.0", "0.5,0.0", "0.0,0.5", "0.25,0.25"]
    
    for seed in range(1, args.seeds + 1):
        for i, shift in enumerate(shifts):
            out_file = f"{args.output_dir}/tile_s{seed}_sh{i}.csv"
            if not os.path.exists(out_file):
                # Use greedy prefix order for diversity
                cmd = f"./bin/solver_tile --seed {seed} --shift {shift} --prefix-order greedy --pool-size 200 --output {out_file}"
                commands.append(cmd)

    print(f"Generated {len(commands)} commands.")

    # Execute efficiently
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run_command, cmd) for cmd in commands]
        for f in futures:
            try:
                f.result()
            except Exception as e:
                print(f"Command failed: {e}")

    # Ensembling
    print("Running Ensemble...")
    csvs = glob.glob(f"{args.output_dir}/*.csv")
    ensemble_out = f"{args.output_dir}/submission_ensemble.csv"
    
    # Run ensemble command
    # Assuming ensemble_submissions takes output first then inputs
    chunk_size = 50 # Avoid too long command line
    current_ensemble = ensemble_out
    
    # If too many files, we might need iterative ensemble or just pass all if shell allows
    # The C++ tool ./bin/ensemble_submissions output.csv input1.csv input2.csv ...
    
    cmd_ensemble = f"./bin/ensemble_submissions {ensemble_out} " + " ".join(csvs)
    run_command(cmd_ensemble)

    # Score
    print("Scoring final ensemble...")
    run_command(f"./bin/score_submission {ensemble_out}")

if __name__ == "__main__":
    main()
