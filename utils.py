import os
import json
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from unmixing_pytorch import (
    load_HSI,
    plotEndmembersAndGTV2,
    plotAbundancesSimpleGT,
    plotTrainingHistoryV2,
)


def setup_output_directory(config):
    """Creates a timestamped directory for the current run."""
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        "./Results",
        config["run_settings"]["method_name"],
        config["run_settings"]["dataset"],
        session_timestamp,
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_artifacts(output_dir, model, history, endmembers, abundances, params):
    """Saves all artifacts from a training run to a specified directory."""
    # 1. Save parameters to JSON
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        # Convert non-serializable items to strings
        params_to_save = {
            k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
            for k, v in params.items()
        }
        json.dump(params_to_save, f, indent=4)

    # 2. Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, "model_weights.pth"))

    # 3. Save results as NumPy arrays
    np.save(os.path.join(output_dir, "endmembers.npy"), endmembers)
    np.save(os.path.join(output_dir, "abundances.npy"), abundances)

    # 4. Save training history (e.g., as a simple text or csv file)
    np.savetxt(
        os.path.join(output_dir, "training_history.csv"),
        np.array(history),
        delimiter=",",
    )

    print(f"All run artifacts saved to: {output_dir}")


def generate_plots(output_dir, endmembers, abundances, history, hsi, params):
    """Generates and saves all plots for a run."""
    lr = params["training_parameters"]["learning_rate"]
    bs = params["training_parameters"]["batch_size"]
    ep = params["training_parameters"]["epochs"]
    run_suffix = f"LR:{lr}, Batch:{bs}, Epochs:{ep}"

    # Plot 1: Training History
    fig1 = plotTrainingHistoryV2(history, title_suffix=run_suffix, show_plot=False)
    fig1.savefig(os.path.join(output_dir, "plot_training_history.png"))
    plt.close(fig1)

    # Plot 2: Endmembers vs. Ground Truth
    fig2 = plotEndmembersAndGTV2(
        endmembers, hsi.gt, title_suffix=run_suffix, show_plot=False
    )
    fig2.savefig(os.path.join(output_dir, "plot_endmembers.png"))
    plt.close(fig2)

    # Plot 3: Abundances vs. Ground Truth
    gt_abundances = hsi.get_abundances()
    fig3 = plotAbundancesSimpleGT(
        abundances,
        gt_abundances,
        title_suffix=run_suffix,
        colormap="viridis",
        show_plot=False,
    )
    fig3.savefig(os.path.join(output_dir, "plot_abundances.png"))
    plt.close(fig3)

    print(f"All plots saved to: {output_dir}")


def load_and_plot_from_dir(run_dir):
    """Loads artifacts from a run directory and regenerates plots."""
    print(f"\n--- Loading results from: {run_dir} ---")

    # Load parameters
    with open(os.path.join(run_dir, "params.json"), "r") as f:
        params = json.load(f)

    # Load results
    endmembers = np.load(os.path.join(run_dir, "endmembers.npy"))
    abundances = np.load(os.path.join(run_dir, "abundances.npy"))
    history = np.loadtxt(
        os.path.join(run_dir, "training_history.csv"), delimiter=","
    ).tolist()

    # Load original dataset for ground truth
    hsi = load_HSI(params["data_parameters"]["dataset_path"])

    print("Run Details:")
    for key, value in params.items():
        print(f"  {key}:")
        for p_key, p_value in value.items():
            print(f"    - {p_key}: {p_value}")

    # Regenerate plots and show them
    lr = params["training_parameters"]["learning_rate"]
    bs = params["training_parameters"]["batch_size"]
    ep = params["training_parameters"]["epochs"]
    run_suffix = f"LR:{lr}, Batch:{bs}, Epochs:{ep} (Loaded)"

    plotTrainingHistoryV2(history, title_suffix=run_suffix)
    plotEndmembersAndGTV2(endmembers, hsi.gt, title_suffix=run_suffix)
    plotAbundancesSimpleGT(
        abundances, hsi.get_abundances(), title_suffix=run_suffix, colormap="viridis"
    )
    plt.show()
