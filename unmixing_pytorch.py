import sys
import scipy as sp
import scipy.linalg as splin
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.feature_extraction.image import extract_patches_2d
from scipy import io as sio
from skimage.transform.pyramids import pyramid_reduce
from skimage.transform import rescale

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import h5py


class HSI:
    """
    A class for Hyperspectral Image (HSI) data.
    """

    def __init__(self, data, rows, cols, gt, abundances_map=None):
        if data.shape[0] < data.shape[1]:
            data = data.transpose()

        self.bands = np.min(data.shape)
        self.cols = cols
        self.rows = rows
        self.image = np.reshape(data, (self.rows, self.cols, self.bands))
        self.gt = gt
        self.abundances_map = abundances_map  # added abundances

    def array(self):
        """this returns a array of spectra with shape num pixels x num bands

        Returns:
            a matrix -- array of spectra
        """
        return np.reshape(self.image, (self.rows * self.cols, self.bands))

    def get_bands(self, bands):
        return self.image[:, :, bands]

    def get_abundances(self):
        """
        Return the abundances associated with the spectral observations

        Return
        ------

        out: numpy array (size: n by k)
            abundances
        """
        return np.reshape(self.abundances_map, (self.rows * self.cols, -1))

    def crop_image(self, start_x, start_y, delta_x=None, delta_y=None):
        if delta_x is None:
            delta_x = self.cols - start_x
        if delta_y is None:
            delta_y = self.rows - start_y
        self.cols = delta_x
        self.rows = delta_y
        self.image = self.image[
            start_x : delta_x + start_x, start_y : delta_y + start_y, :
        ]
        return self.image


def load_HSI(path):
    try:
        data = sio.loadmat(path)
    except NotImplementedError:
        data = h5py.File(path, "r")

    numpy_array = np.asarray(data["Y"], dtype=np.float32)
    numpy_array = numpy_array / np.max(numpy_array.flatten())
    n_rows = data["lines"].item()
    n_cols = data["cols"].item()

    if "GT" in data.keys():
        gt = np.asarray(data["GT"], dtype=np.float32)
    else:
        gt = None

    return HSI(numpy_array, n_rows, n_cols, gt, data["S_GT"])


class SumToOne(nn.Module):
    def __init__(self, params, **kwargs):
        super(SumToOne, self).__init__()
        self.num_outputs = params["num_endmembers"]
        self.params = params

    def l_regularization(self, x):
        patch_size = self.params["patch_size"] * self.params["patch_size"]
        z = torch.abs(x + 1e-7)
        l_half = torch.sum(torch.norm(z, p=1, dim=3))
        return 1.0 / patch_size * self.params["l1"] * l_half

    def tv_regularization(self, x):
        patch_size = self.params["patch_size"] * self.params["patch_size"]
        # Total variation using gradient differences
        diff_i = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_j = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        tv = torch.sum(diff_i) + torch.sum(diff_j)
        return 1.0 / patch_size * self.params["tv"] * tv

    def forward(self, x):
        regularization_loss = 0
        if self.params["l1"] > 0.0:
            regularization_loss += self.l_regularization(x)
        if self.params["tv"] > 0.0:
            regularization_loss += self.tv_regularization(x)

        # Store regularization loss for later use in training loop
        self.regularization_loss = regularization_loss
        return x


class Scaling(nn.Module):
    def __init__(self, params, **kwargs):
        super(Scaling, self).__init__()
        self.params = params

    def tv_regularization(self, x):
        patch_size = self.params["patch_size"] * self.params["patch_size"]
        # Total variation using gradient differences
        diff_i = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_j = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        tv = torch.sum(diff_i) + torch.sum(diff_j)
        return 1.0 / patch_size * self.params["tv"] * tv

    def forward(self, x):
        self.regularization_loss = self.tv_regularization(x)
        return F.relu(x)


def SAD(y_true, y_pred):
    y_true2 = F.normalize(y_true, p=2, dim=-1)
    y_pred2 = F.normalize(y_pred, p=2, dim=-1)
    A = torch.mean(y_true2 * y_pred2)
    sad = torch.acos(torch.clamp(A, -1.0, 1.0))
    return sad


def numpy_SAD(y_true, y_pred):
    cos = y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if cos > 1.0:
        cos = 1.0
    return np.arccos(cos)


def calcmSAD(Aorg, Ahat, startCol):
    Aorg = np.squeeze(Aorg)
    if Aorg.shape[1] > Aorg.shape[0]:
        Aorg = Aorg.T
    if Ahat.shape[1] > Ahat.shape[0]:
        Ahat = Ahat.T
    r1 = np.min(Aorg.shape)
    r2 = np.min(Ahat.shape)
    s = np.zeros((r1, r2))
    for i in range(r1):
        ao = Aorg[:, i]
        for j in range(r2):
            ah = Ahat[:, j]
            s[i, j] = np.min([SAD(ao, ah), SAD(ao, -ah)])
    s0 = s
    sad = np.squeeze(np.zeros((1, r1)))
    idxHat = np.squeeze(np.zeros((1, r1)))
    idxOrg = np.squeeze(np.zeros((1, r1)))
    for p in range(r1):
        if startCol > -1 and p == 0:
            b = np.argmin(s[:, startCol])
            sad[p] = np.min(s[:, startCol])
            idxHat[p] = b
            idxOrg[p] = startCol
        else:
            sad[p] = np.min(s.flatten())
            (idxHat[p], idxOrg[p]) = np.unravel_index(np.argmin(s, axis=None), s.shape)
        s[:, int(idxOrg[p])] = np.inf
        s[int(idxHat[p]), :] = np.inf
        if np.isinf(sad[p]):
            idxHat[p] = np.inf
            idxOrg[p] = np.inf
    sad_k = sad
    a = np.sort(idxOrg).astype(int)
    b = np.argsort(idxOrg)
    idxHat = idxHat[b]
    sad_k = sad_k[b]
    sad_m = np.mean(sad_k)

    return sad_m, idxOrg.astype(int), idxHat.astype(int), sad_k


def asam_and_order(Aorg, Ahat):
    if Aorg.shape[1] > Aorg.shape[0]:
        Aorg = Aorg.T
    if Ahat.shape[1] > Ahat.shape[0]:
        Ahat = Ahat.T
    r = Aorg.shape[1]
    idxOrg = np.zeros((r, r), dtype=np.int8)
    idxHat = np.zeros((r, r), dtype=np.int8)

    sad_m, idxOrg, idxHat, sad_k = calcmSAD(Aorg, Ahat, -1)

    return idxHat, sad_m


def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    mydict = {}
    sad_mat = np.ones((num_endmembers, num_endmembers))
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_SAD(endmembers[i, :], endmembersGT[j, :])
    rows = 0
    while rows < num_endmembers:
        minimum = sad_mat.min()
        index_arr = np.where(sad_mat == minimum)
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        if index[0] in mydict.keys():
            sad_mat[index[0], index[1]] = 100
        elif index[1] in mydict.values():
            sad_mat[index[0], index[1]] = 100
        else:
            mydict[index[0]] = index[1]
            sad_mat[index[0], index[1]] = 100
            rows += 1
    ASAM = 0
    num = 0
    for i in range(num_endmembers):
        if np.var(endmembersGT[mydict[i]]) > 0:
            ASAM = ASAM + numpy_SAD(endmembers[i, :], endmembersGT[mydict[i]])
            num += 1

    return mydict, ASAM / float(num)


def plotEndmembers(endmembers):
    if isinstance(endmembers, torch.Tensor):
        endmembers = endmembers.detach().cpu().numpy()

    if len(endmembers.shape) > 2 and endmembers.shape[1] > 1:
        endmembers = np.squeeze(endmembers).mean(axis=0).mean(axis=0)
    else:
        endmembers = np.squeeze(endmembers)
    # endmembers = endmembers / endmembers.max()
    num_endmembers = np.min(endmembers.shape)
    fig = plt.figure(num=1, figsize=(8, 8))
    n = num_endmembers / 2
    if num_endmembers % 2 != 0:
        n = n + 1
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[i, :], "r", linewidth=1.0)
        ax.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig("endm.png")
    plt.close()


def plotEndmembersAndGT(endmembers, endmembersGT):
    if isinstance(endmembers, torch.Tensor):
        endmembers = endmembers.detach().cpu().numpy()
    if isinstance(endmembersGT, torch.Tensor):
        endmembersGT = endmembersGT.detach().cpu().numpy()

    num_endmembers = endmembers.shape[0]
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1

    hat, sad = order_endmembers(endmembersGT, endmembers)
    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()
    title = "mSAD: " + format(sad, ".3f") + " radians"
    st = plt.suptitle(title)

    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[hat[i], :], "r", linewidth=1.0)
        plt.plot(endmembersGT[i, :], "k", linewidth=1.0)
        ax.set_title(
            format(numpy_SAD(endmembers[hat[i], :], endmembersGT[i, :]), ".3f")
        )
        ax.get_xaxis().set_visible(False)

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.88)
    plt.draw()
    plt.pause(0.001)


def plotEndmembersAndGTV2(endmembers, endmembersGT, title_suffix="", save_path=None):
    """Enhanced version of plotEndmembersAndGT with legends and improved visuals"""
    if isinstance(endmembers, torch.Tensor):
        endmembers = endmembers.detach().cpu().numpy()
    if isinstance(endmembersGT, torch.Tensor):
        endmembersGT = endmembersGT.detach().cpu().numpy()

    num_endmembers = endmembers.shape[0]
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1

    hat, sad = order_endmembers(endmembersGT, endmembers)

    # Create figure with improved styling - fix the subplot layout
    fig, axes = plt.subplots(1, num_endmembers, figsize=(6 * num_endmembers, 6))

    # Handle single endmember case
    if num_endmembers == 1:
        axes = [axes]

    plt.style.use("seaborn-v0_8-whitegrid")

    main_title = f"Endmember Comparison - mSAD: {sad:.4f} radians"
    if title_suffix:
        main_title += f" ({title_suffix})"

    fig.suptitle(main_title, fontsize=16, fontweight="bold")

    # Normalize endmembers
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    # Color palette for endmembers
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    for i in range(num_endmembers):
        ax = axes[i]

        # Plot with enhanced styling
        wavelengths = np.arange(len(endmembers[hat[i], :]))
        ax.plot(
            wavelengths,
            endmembers[hat[i], :],
            color=colors[i % len(colors)],
            linewidth=2.5,
            label="Estimated",
            alpha=0.8,
        )
        ax.plot(
            wavelengths,
            endmembersGT[i, :],
            color="black",
            linewidth=2,
            linestyle="--",
            label="Ground Truth",
            alpha=0.9,
        )

        # Individual SAD for this endmember
        individual_sad = numpy_SAD(endmembers[hat[i], :], endmembersGT[i, :])
        ax.set_title(
            f"Endmember {i+1}\nSAD: {individual_sad:.4f} rad",
            fontsize=12,
            fontweight="bold",
        )

        # Styling
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Band Index", fontsize=10)
        ax.set_ylabel("Reflectance", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, 1.1)

        # Add subtle background
        ax.patch.set_facecolor("#f8f9fa")
        ax.patch.set_alpha(0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return sad


def plotAbundancesSimple(abundances, name):
    if isinstance(abundances, torch.Tensor):
        abundances = abundances.detach().cpu().numpy()

    abundances = np.transpose(abundances, axes=[1, 0, 2])
    num_endmembers = abundances.shape[2]
    # n = num_endmembers / 2  # original
    n = (num_endmembers + 1) // 2  # Use integer division
    if num_endmembers % 2 != 0:
        n = n + 1
    cmap = "viridis"
    plt.figure(figsize=[12, 12])
    AA = np.sum(abundances, axis=-1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position="bottom", size="5%", pad=0.05)
        im = ax.imshow(abundances[:, :, i], cmap=cmap)
        plt.colorbar(im, cax=cax, orientation="horizontal")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    # plt.savefig(name+'.png')
    plt.close()


def plotAbundancesSimpleV2(
    abundances, title_suffix="", save_path=None, colormap="viridis"
):
    """Enhanced version of plotAbundancesSimple with legends and improved visuals"""
    if isinstance(abundances, torch.Tensor):
        abundances = abundances.detach().cpu().numpy()

    abundances = np.transpose(abundances, axes=[1, 0, 2])
    num_endmembers = abundances.shape[2]

    # Calculate grid dimensions
    n_cols = min(3, num_endmembers)  # Max 3 columns
    n_rows = (num_endmembers + n_cols - 1) // n_cols

    # Create figure with improved layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if num_endmembers == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if num_endmembers == 1 else axes
    else:
        axes = axes.flatten()

    main_title = f"Abundance Maps"
    if title_suffix:
        main_title += f" - {title_suffix}"

    fig.suptitle(main_title, fontsize=16, fontweight="bold")

    # Enhanced colormap options
    colormaps = {
        "viridis": "viridis",
        "plasma": "plasma",
        "inferno": "inferno",
        "magma": "magma",
        "hot": "hot",
        "cool": "cool",
    }
    cmap = colormaps.get(colormap, "viridis")

    # Color names for endmembers
    endmember_names = ["Soil", "Vegetation", "Water", "Urban", "Rock", "Sand"]

    for i in range(num_endmembers):
        ax = axes[i]

        # Create abundance map
        im = ax.imshow(abundances[:, :, i], cmap=cmap, vmin=0, vmax=1)

        # Enhanced title with statistics
        abundance_mean = np.mean(abundances[:, :, i])
        abundance_std = np.std(abundances[:, :, i])
        abundance_max = np.max(abundances[:, :, i])

        endmember_name = endmember_names[i] if i < len(endmember_names) else f"EM{i+1}"
        title = f"{endmember_name}\nMean: {abundance_mean:.3f} ± {abundance_std:.3f}\nMax: {abundance_max:.3f}"
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Remove ticks but keep clean appearance
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar with better positioning
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Abundance", rotation=270, labelpad=15, fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)

    # Hide unused subplots
    for i in range(num_endmembers, len(axes)):
        axes[i].set_visible(False)

    # Add summary statistics
    total_abundance = np.sum(abundances, axis=2)
    asc_violation = np.mean(np.abs(total_abundance - 1))

    # Add text box with overall statistics
    textstr = f"ASC Violation (MAE): {asc_violation:.4f}\nPixels: {abundances.shape[0]}×{abundances.shape[1]}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    fig.text(0.02, 0.02, textstr, fontsize=10, bbox=props)

    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.08)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plotAbundancesSimpleGT(
    abundances, abundancesGT=None, title_suffix="", save_path=None, colormap="viridis"
):
    """Enhanced version with ground truth comparison - 2 rows, 3 columns layout"""
    if isinstance(abundances, torch.Tensor):
        abundances = abundances.detach().cpu().numpy()

    if abundancesGT is not None and isinstance(abundancesGT, torch.Tensor):
        abundancesGT = abundancesGT.detach().cpu().numpy()

    abundances = np.transpose(abundances, axes=[1, 0, 2])
    if abundancesGT is not None:
        abundancesGT = np.transpose(abundancesGT, axes=[1, 0, 2])

    num_endmembers = abundances.shape[2]

    # Fixed layout: 2 rows, 3 columns
    n_rows = 2
    n_cols = 3

    # Create figure with fixed layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))

    main_title = f"Abundance Maps Comparison"
    if title_suffix:
        main_title += f" - {title_suffix}"

    fig.suptitle(main_title, fontsize=16, fontweight="bold")

    # Enhanced colormap options
    colormaps = {
        "viridis": "viridis",
        "plasma": "plasma",
        "inferno": "inferno",
        "magma": "magma",
        "hot": "hot",
        "cool": "cool",
    }
    cmap = colormaps.get(colormap, "viridis")

    # Color names for endmembers
    endmember_names = ["Soil", "Vegetation", "Water", "Urban", "Rock", "Sand"]

    # Plot ground truth in first row (if available)
    for i in range(min(num_endmembers, n_cols)):
        ax = axes[0, i]

        if abundancesGT is not None:
            # Plot ground truth
            im = ax.imshow(abundancesGT[:, :, i], cmap=cmap, vmin=0, vmax=1)

            # Calculate statistics for ground truth
            abundance_mean = np.mean(abundancesGT[:, :, i])
            abundance_std = np.std(abundancesGT[:, :, i])
            abundance_max = np.max(abundancesGT[:, :, i])

            endmember_name = (
                endmember_names[i] if i < len(endmember_names) else f"EM{i+1}"
            )
            title = f"GT {endmember_name}\nMean: {abundance_mean:.3f} ± {abundance_std:.3f}\nMax: {abundance_max:.3f}"
            ax.set_title(title, fontsize=11, fontweight="bold")
        else:
            # No ground truth available
            ax.text(
                0.5,
                0.5,
                "No Ground Truth\nAvailable",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
            )
            ax.set_xticks([])
            ax.set_yticks([])

        # Remove ticks but keep clean appearance
        ax.set_xticks([])
        ax.set_yticks([])

        if abundancesGT is not None:
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("Abundance", rotation=270, labelpad=15, fontsize=10)
            cbar.ax.tick_params(labelsize=9)

        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)

    # Plot estimated abundances in second row
    for i in range(min(num_endmembers, n_cols)):
        ax = axes[1, i]

        # Create abundance map
        im = ax.imshow(abundances[:, :, i], cmap=cmap, vmin=0, vmax=1)

        # Enhanced title with statistics
        abundance_mean = np.mean(abundances[:, :, i])
        abundance_std = np.std(abundances[:, :, i])
        abundance_max = np.max(abundances[:, :, i])

        # Calculate error metrics if ground truth is available
        error_text = ""
        if abundancesGT is not None and i < abundancesGT.shape[2]:
            mae = np.mean(np.abs(abundances[:, :, i] - abundancesGT[:, :, i]))
            rmse = np.sqrt(np.mean((abundances[:, :, i] - abundancesGT[:, :, i]) ** 2))
            error_text = f"\nMAE: {mae:.3f}, RMSE: {rmse:.3f}"

        endmember_name = endmember_names[i] if i < len(endmember_names) else f"EM{i+1}"
        title = f"Est. {endmember_name}\nMean: {abundance_mean:.3f} ± {abundance_std:.3f}\nMax: {abundance_max:.3f}{error_text}"
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Remove ticks but keep clean appearance
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar with better positioning
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Abundance", rotation=270, labelpad=15, fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)

    # Hide unused subplots
    for i in range(num_endmembers, n_cols):
        axes[0, i].set_visible(False)
        axes[1, i].set_visible(False)

    # Add summary statistics
    total_abundance_est = np.sum(abundances, axis=2)
    asc_violation_est = np.mean(np.abs(total_abundance_est - 1))

    # Calculate overall error metrics if ground truth is available
    if abundancesGT is not None:
        total_abundance_gt = np.sum(abundancesGT, axis=2)
        asc_violation_gt = np.mean(np.abs(total_abundance_gt - 1))

        # Overall MAE and RMSE
        overall_mae = np.mean(np.abs(abundances - abundancesGT))
        overall_rmse = np.sqrt(np.mean((abundances - abundancesGT) ** 2))

        textstr = (
            f"Estimated ASC Violation: {asc_violation_est:.4f}\n"
            f"GT ASC Violation: {asc_violation_gt:.4f}\n"
            f"Overall MAE: {overall_mae:.4f}\n"
            f"Overall RMSE: {overall_rmse:.4f}\n"
            f"Pixels: {abundances.shape[0]}×{abundances.shape[1]}"
        )
    else:
        textstr = (
            f"ASC Violation: {asc_violation_est:.4f}\n"
            f"Pixels: {abundances.shape[0]}×{abundances.shape[1]}"
        )

    # Add text box with overall statistics
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    fig.text(0.02, 0.02, textstr, fontsize=10, bbox=props)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plotTrainingHistory(history, title_suffix="", save_path=None):
    """Plot training loss history with enhanced visuals"""
    if not history:
        print("No training history to plot")
        return

    plt.figure(figsize=(10, 3))
    plt.style.use("seaborn-v0_8-whitegrid")

    epochs = range(1, len(history) + 1)
    plt.plot(epochs, history, "b-", linewidth=2, marker="o", markersize=4, alpha=0.8)

    plt.title(
        f'Training Loss History{" - " + title_suffix if title_suffix else ""}',
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add final loss annotation
    final_loss = history[-1]
    plt.annotate(
        f"Final Loss: {final_loss:.6f}",
        xy=(len(history), final_loss),
        xytext=(len(history) * 0.7, final_loss * 1.1),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    # Set y-axis to log scale if loss varies significantly
    if max(history) / min(history) > 10:
        plt.yscale("log")
        plt.ylabel("Loss (log scale)", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plotTrainingHistoryV2(history, title_suffix="", save_path=None):
    """Plot training loss history with enhanced visuals and responsive sizing"""
    if not history:
        print("No training history to plot")
        return

    # Calculate appropriate figure size based on data length
    num_epochs = len(history)

    # Base figure size with responsive width
    base_width = 8
    base_height = 2

    # Adjust width based on number of epochs, but cap it
    if num_epochs <= 50:
        fig_width = base_width
    elif num_epochs <= 200:
        fig_width = min(base_width + 2, 12)
    else:
        fig_width = min(base_width + 4, 15)

    # Keep height reasonable
    # fig_height = min(base_height, 8)
    fig_height = 3

    plt.figure(figsize=(fig_width, fig_height))
    plt.style.use("seaborn-v0_8-whitegrid")

    epochs = range(1, len(history) + 1)

    # Adjust marker size based on number of epochs
    if num_epochs <= 50:
        marker_size = 4
        line_width = 2
    elif num_epochs <= 200:
        marker_size = 2
        line_width = 1.5
    else:
        marker_size = 1
        line_width = 1

    plt.plot(
        epochs,
        history,
        "b-",
        linewidth=line_width,
        marker="o",
        markersize=marker_size,
        alpha=0.8,
    )

    plt.title(
        f'Training Loss History{" - " + title_suffix if title_suffix else ""}',
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add final loss annotation only if there's enough space
    final_loss = history[-1]
    if num_epochs >= 10:  # Only add annotation if we have enough epochs
        plt.annotate(
            f"Final Loss: {final_loss:.6f}",
            xy=(len(history), final_loss),
            xytext=(len(history) * 0.7, final_loss * 1.1),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    # Set y-axis to log scale if loss varies significantly
    if len(history) > 1 and max(history) / min(history) > 10:
        plt.yscale("log")
        plt.ylabel("Loss (log scale)", fontsize=12)

    # Ensure reasonable axis limits
    plt.xlim(0.5, len(history) + 0.5)

    # Add some padding to y-axis
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def reconstruct(A, S):
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()

    s_shape = S.shape
    S = np.reshape(S, (S.shape[0] * S.shape[1], S.shape[2]))
    reconstructed = np.matmul(S, A)
    reconstructed = np.reshape(
        reconstructed, (s_shape[0], s_shape[1], reconstructed.shape[1])
    )
    return reconstructed


def estimate_snr(Y, r_m, x):

    # L number of bands (channels), N number of pixels
    [L, N] = Y.shape
    [p, N] = x.shape  # p number of endmembers (reduced dimension)

    P_y = sp.sum(Y**2) / float(N)
    P_x = sp.sum(x**2) / float(N) + sp.sum(r_m**2)
    snr_est = 10 * sp.log10((P_x - p / L * P_y) / (P_y - P_x))

    return snr_est


def vca(Y, R, verbose=True, snr_input=0):
    # Vertex Component Analysis
    #
    # Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)
    #
    # ------- Input variables -------------
    #  Y - matrix with dimensions L(channels) x N(pixels)
    #      each pixel is a linear mixture of R endmembers
    #      signatures Y = M x s, where s = gamma x alfa
    #      gamma is a illumination perturbation factor and
    #      alfa are the abundance fractions of each endmember.
    #  R - positive integer number of endmembers in the scene
    #
    # ------- Output variables -----------
    # Ae     - estimated mixing matrix (endmembers signatures)
    # indice - pixels that were chosen to be the most pure
    # Yp     - Data matrix Y projected.
    #
    # ------- Optional parameters---------
    # snr_input - (float) signal to noise ratio (dB)
    # v         - [True | False]
    # ------------------------------------
    #
    # Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)
    # This code is a translation of a matlab code provided by
    # Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
    # available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
    # Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))
    #
    # more details on:
    # Jose M. P. Nascimento and Jose M. B. Dias
    # "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
    # submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
    #

    #############################################
    # Initializations
    #############################################
    if len(Y.shape) != 2:
        sys.exit(
            "Input data must be of size L (number of bands i.e. channels) by N (number of pixels)"
        )

    [L, N] = Y.shape  # L number of bands (channels), N number of pixels

    R = int(R)
    if R < 0 or R > L:
        sys.exit("ENDMEMBER parameter must be integer between 1 and L")

    #############################################
    # SNR Estimates
    #############################################

    if snr_input == 0:
        y_m = sp.mean(Y, axis=1, keepdims=True)
        Y_o = Y - y_m  # data with zero-mean
        # computes the R-projection matrix
        Ud = splin.svd(sp.dot(Y_o, Y_o.T) / float(N))[0][:, :R]
        # project the zero-mean data onto p-subspace
        x_p = sp.dot(Ud.T, Y_o)

        SNR = estimate_snr(Y, y_m, x_p)

        if verbose:
            print("SNR estimated = {}[dB]".format(SNR))
    else:
        SNR = snr_input
        if verbose:
            print("input SNR = {}[dB]\n".format(SNR))

    SNR_th = 15 + 10 * sp.log10(R)

    #############################################
    # Choosing Projective Projection or
    #          projection to p-1 subspace
    #############################################

    if SNR < SNR_th:
        if verbose:
            print("... Select proj. to R-1")

            d = R - 1
            if snr_input == 0:  # it means that the projection is already computed
                Ud = Ud[:, :d]
            else:
                y_m = sp.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m  # data with zero-mean

                # computes the p-projection matrix
                Ud = splin.svd(sp.dot(Y_o, Y_o.T) / float(N))[0][:, :d]
                # project thezeros mean data onto p-subspace
                x_p = sp.dot(Ud.T, Y_o)

            Yp = sp.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

            x = x_p[:d, :]  # x_p =  Ud.T * Y_o is on a R-dim subspace
            c = sp.amax(sp.sum(x**2, axis=0)) ** 0.5
            y = sp.vstack((x, c * sp.ones((1, N))))
    else:
        if verbose:
            print("... Select the projective proj.")

        d = R
        # computes the p-projection matrix
        Ud = splin.svd(sp.dot(Y, Y.T) / float(N))[0][:, :d]

        x_p = sp.dot(Ud.T, Y)
        # again in dimension L (note that x_p has no null mean)
        Yp = sp.dot(Ud, x_p[:d, :])

        x = sp.dot(Ud.T, Y)
        u = sp.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
        y = x / sp.dot(u.T, x)

    #############################################
    # VCA algorithm
    #############################################

    indice = sp.zeros((R), dtype=int)
    A = sp.zeros((R, R))
    A[-1, 0] = 1

    for i in range(R):
        w = sp.random.rand(R, 1)
        f = w - sp.dot(A, sp.dot(splin.pinv(A), w))
        f = f / splin.norm(f)

        v = sp.dot(f.T, y)

        indice[i] = sp.argmax(sp.absolute(v))
        A[:, i] = y[:, indice[i]]  # same as x(:,indice(i))

    Ae = Yp[:, indice]

    return Ae, indice, Yp
