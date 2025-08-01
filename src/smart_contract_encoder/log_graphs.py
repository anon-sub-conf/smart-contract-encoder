import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # df = pd.read_pickle("hist_32_final.pkl")
    # df = pd.read_pickle("hist_32.pkl")
    # # df = pd.read_pickle("hist_16_final.pkl")
    # df = df.sort_values(by='step')
    # plt.rcParams.update({'font.size': 15})
    # fig, axes = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

    # df_loss = df[['step', 'loss']].dropna()
    # axes.plot(df_loss['step'], df_loss['loss'], label='Train Loss', color='tab:blue', marker='o',  linestyle='-')
    # if 'eval_loss' in df.columns:
    #     df_eval_loss = df[['step', 'eval_loss']].dropna()
    #     axes.plot(df_eval_loss['step'], df_eval_loss['eval_loss'], label='Eval Loss', color='tab:orange', marker='o',  linestyle=':')
    # axes.set_ylabel("Loss")
    # axes.set_title("Loss over Steps")
    # axes.legend()
    # axes.grid(True)
    # plt.savefig("log_plot_32.png")

    plt.rcParams.update({'font.size': 15})
    df0 = pd.read_pickle("hist_64_2.pkl")
    df1 = pd.read_pickle("hist_4_2.pkl")
    df2 = pd.read_pickle("hist_32_2.pkl")

    # Filter for eval logs (or loss, accuracy, etc.)
    df0 = df0.dropna(subset=['eval_loss'])
    df1 = df1.dropna(subset=['eval_loss'])
    df2 = df2.dropna(subset=['eval_loss'])

    # # Normalize progress (0 to 1)
    df0['progress'] = df0['step'] / df0['step'].max()
    df1['progress'] = df1['step'] / df1['step'].max()
    df2['progress'] = df2['step'] / df2['step'].max()

    # Define common progress points (e.g., 100 points between 0 and 1)
    progress_points = np.linspace(0, 1, 100)

    # Interpolate eval_loss at those points
    df0_interp = np.interp(progress_points, df0['progress'], df0['eval_loss'])
    df1_interp = np.interp(progress_points, df1['progress'], df1['eval_loss'])
    df2_interp = np.interp(progress_points, df2['progress'], df2['eval_loss'])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(progress_points, df0_interp, label='Batch 64', marker='.', markersize=4)
    plt.plot(progress_points, df1_interp, label='Batch 4', marker='o', markersize=4)
    plt.plot(progress_points, df2_interp, label='Batch 32', marker='s', markersize=4)

    plt.xlabel("Epoch")
    plt.ylabel("Eval Loss")
    # plt.title("Eval Loss vs Training Progress")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.savefig("log_plot_4_32_64.png")
