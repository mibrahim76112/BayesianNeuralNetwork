import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="runs_bnn_wmmse/eval_active_users.csv")
    p.add_argument("--out_dir", type=str, default="runs_bnn_wmmse/plots")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Plot 1: average rates
    plt.figure()
    plt.plot(df["k_active"], df["wmmse_avg"], marker="o", label="WMMSE")
    plt.plot(df["k_active"], df["dnn_avg"], marker="o", label="DNN")
    plt.plot(df["k_active"], df["bnn_avg"], marker="o", label="BNN avg over samples")
    plt.plot(df["k_active"], df["bnn_max"], marker="o", label="BNN max over samples")
    plt.xlabel("Number of active users")
    plt.ylabel("Average weighted sum-rate")
    plt.title("DNN vs BNN beamforming generalization")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(args.out_dir, "avg_wsr_vs_active_users.png")
    plt.savefig(path, dpi=200)
    plt.close()

    # Plot 2: BNN variance
    plt.figure()
    plt.plot(df["k_active"], df["bnn_var"], marker="o", label="BNN rate variance")
    plt.xlabel("Number of active users")
    plt.ylabel("Variance of BNN sampled WSR")
    plt.title("BNN uncertainty vs active-user shift")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(args.out_dir, "bnn_variance_vs_active_users.png")
    plt.savefig(path, dpi=200)
    plt.close()

    # Plot 3: gaps to WMMSE
    plt.figure()
    plt.plot(df["k_active"], df["dnn_gap"], marker="o", label="DNN gap")
    plt.plot(df["k_active"], df["bnn_avg_gap"], marker="o", label="BNN avg gap")
    plt.plot(df["k_active"], df["bnn_max_gap"], marker="o", label="BNN max gap")
    plt.xlabel("Number of active users")
    plt.ylabel("WMMSE rate - model rate")
    plt.title("Gap to WMMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(args.out_dir, "gap_to_wmmse_vs_active_users.png")
    plt.savefig(path, dpi=200)
    plt.close()

    # Plot 4: outage probability
    plt.figure()
    plt.plot(df["k_active"], df["dnn_outage"], marker="o", label="DNN outage")
    plt.plot(df["k_active"], df["bnn_avg_outage"], marker="o", label="BNN avg outage")
    plt.plot(df["k_active"], df["bnn_max_outage"], marker="o", label="BNN max outage")
    plt.xlabel("Number of active users")
    plt.ylabel("Outage probability")
    plt.title("Outage: model rate < eta x WMMSE rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(args.out_dir, "outage_vs_active_users.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print("saved plots in:", args.out_dir)


if __name__ == "__main__":
    main()
