from __future__ import annotations

import io
import re
import zipfile
import tempfile
import os
import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "GIWAXS Plotter RL"
APP_SUBTITLE = "Web v1.0 · GIWAXS data processing and plotting · Developed by Richard Li"

PRESETS = ["Qimage", "Qphi", "Azimuthal", "CirAvg"]
CMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis", "jet", "turbo",
    "seismic", "coolwarm", "gray", "hot", "nipy_spectral",
]


def apply_preset_defaults(preset: str):
    defaults = {
        "Qimage": dict(xmin=-0.3, xmax=2.0, ymin=0.0, ymax=2.3, vmin=1.0, vmax=5000.0, scale="log"),
        "Qphi": dict(xmin=0.2, xmax=2.8, ymin=0.0, ymax=180.0, vmin=1.0, vmax=5000.0, scale="log"),
        "Azimuthal": dict(xmin=0.0, xmax=180.0, ymin=1.0, ymax=5000.0, vmin=1.0, vmax=5000.0, scale="log"),
        "CirAvg": dict(xmin=0.1, xmax=2.5, ymin=0.0, ymax=5000.0, vmin=1.0, vmax=5000.0, scale="linear"),
    }
    d = defaults[preset]
    for k, v in d.items():
        state_key = f"giwaxs_{k}"
        if state_key not in st.session_state:
            st.session_state[state_key] = v


def reset_defaults_force(preset: str):
    defaults = {
        "Qimage": dict(xmin=-0.3, xmax=2.0, ymin=0.0, ymax=2.3, vmin=1.0, vmax=5000.0, scale="log"),
        "Qphi": dict(xmin=0.2, xmax=2.8, ymin=0.0, ymax=180.0, vmin=1.0, vmax=5000.0, scale="log"),
        "Azimuthal": dict(xmin=0.0, xmax=180.0, ymin=1.0, ymax=5000.0, vmin=1.0, vmax=5000.0, scale="log"),
        "CirAvg": dict(xmin=0.1, xmax=2.5, ymin=0.0, ymax=5000.0, vmin=1.0, vmax=5000.0, scale="linear"),
    }
    d = defaults[preset]
    for k, v in d.items():
        st.session_state[f"giwaxs_{k}"] = v


def extract_theta_from_name(filename: str):
    match = re.search(r"th([-+]?\d+(?:\.\d+)?)", filename)
    if not match:
        return None
    return f"{float(match.group(1)):.3f}"


def build_background_map(bg_files):
    bg_map = {}
    duplicate_angles = {}
    for bg_file in bg_files:
        theta = extract_theta_from_name(bg_file.name)
        if theta is None:
            continue
        if theta in bg_map:
            duplicate_angles.setdefault(theta, []).append(bg_map[theta].name)
            duplicate_angles[theta].append(bg_file.name)
        else:
            bg_map[theta] = bg_file
    if duplicate_angles:
        msg = "Duplicate background files found for these incident angles:\n"
        for theta, names in duplicate_angles.items():
            msg += f"\nth{theta}:\n"
            for name in names:
                msg += f"  {name}\n"
        msg += "\nPlease keep only one background .npz per incident angle."
        raise ValueError(msg)
    return bg_map


def first_existing_key(data, candidates):
    lower_map = {k.lower(): k for k in data.files}
    for cand in candidates:
        if cand in data.files:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_qimage_npz(file_obj, flip_qx: bool):
    # Read directly from Streamlit's UploadedFile buffer. Avoid getvalue(), which
    # makes an additional full copy of every uploaded NPZ in RAM.
    file_obj.seek(0)
    with np.load(file_obj, allow_pickle=False) as data:
        required_keys = ["qimg", "qx", "qz"]
        for key in required_keys:
            if key not in data.files:
                raise KeyError(f"Missing required key '{key}' in {file_obj.name}. Available keys: {data.files}")
        qimg = np.squeeze(data["qimg"]).astype(float)
        qx = np.squeeze(data["qx"]).astype(float)
        qz = np.squeeze(data["qz"]).astype(float)
    file_obj.seek(0)
    if qimg.shape != (len(qz), len(qx)):
        raise ValueError(f"Shape mismatch in {file_obj.name}: qimg.shape={qimg.shape}, expected={(len(qz), len(qx))}")
    if flip_qx:
        qx = -qx
    qx_order = np.argsort(qx)
    qx = qx[qx_order]
    qimg = qimg[:, qx_order]
    qz_order = np.argsort(qz)
    qz = qz[qz_order]
    qimg = qimg[qz_order, :]
    return qx, qz, qimg


def load_qphi_npz(file_obj):
    file_obj.seek(0)
    with np.load(file_obj, allow_pickle=False) as data:
        q_key = first_existing_key(data, ["q", "qr", "q_axis", "qvals", "q_vals"])
        phi_key = first_existing_key(data, ["phi", "chi", "azimuth", "azimuthal", "phi_axis", "phi_vals"])
        im_key = first_existing_key(data, ["qphi", "qphi_img", "qphi_map", "qphi_image", "intensity", "image", "img", "I", "qimg"])
        if q_key is None or phi_key is None or im_key is None:
            raise KeyError(f"Could not identify Qphi keys in {file_obj.name}. Need q, phi, and image arrays. Available keys: {data.files}")
        q = np.squeeze(data[q_key]).astype(float)
        phi = np.squeeze(data[phi_key]).astype(float)
        im = np.squeeze(data[im_key]).astype(float)
    file_obj.seek(0)
    if q.ndim == 2:
        q = q[:, 1] if q.shape[1] > 1 else q.ravel()
    if phi.ndim == 2:
        phi = phi[:, 1] if phi.shape[1] > 1 else phi.ravel()
    if im.shape != (len(phi), len(q)):
        if im.T.shape == (len(phi), len(q)):
            im = im.T
        else:
            raise ValueError(f"Shape mismatch in {file_obj.name}: image.shape={im.shape}, expected={(len(phi), len(q))}. Keys: image={im_key}, q={q_key}, phi={phi_key}")
    q_order = np.argsort(q)
    q = q[q_order]
    im = im[:, q_order]
    phi_order = np.argsort(phi)
    phi = phi[phi_order]
    im = im[phi_order, :]
    return q, phi, im


def load_ciravg_csv(file_obj):
    file_obj.seek(0)
    df = pd.read_csv(file_obj)
    file_obj.seek(0)
    if "q_ca" in df.columns and "iq_ca" in df.columns:
        q = pd.to_numeric(df["q_ca"], errors="coerce").to_numpy()
        iq = pd.to_numeric(df["iq_ca"], errors="coerce").to_numpy()
        return q, iq
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_cols = [c for c in numeric_df.columns if numeric_df[c].notna().sum() > 0]
    if len(numeric_cols) < 2:
        raise ValueError(f"Could not find two numeric columns in {file_obj.name}")
    q = numeric_df[numeric_cols[0]].to_numpy()
    iq = numeric_df[numeric_cols[1]].to_numpy()
    return q, iq


def get_norm(scale, vmin, vmax):
    if scale == "log":
        return LogNorm(vmin=vmin, vmax=vmax), "log"
    return Normalize(vmin=vmin, vmax=vmax), "linear"


def get_cmap(cmap_name):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")
    return cmap


def clean_for_plot(image, scale):
    im = np.asarray(image, dtype=float).copy()
    im[~np.isfinite(im)] = np.nan
    if scale == "log":
        im[im <= 0] = np.nan
    return np.ma.masked_invalid(im)


def set_common_background(fig, ax, transparent: bool):
    if transparent:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")


def fig_to_png_bytes(fig, dpi, transparent):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
    return buf.getvalue()


def set_image_ticks(ax, xmin, xmax, ymin, ymax):
    ax.set_xticks(np.arange(0, xmax + 0.01, 0.5))
    ax.set_xticks(np.arange(xmin, xmax + 0.01, 0.1), minor=True)
    ax.set_yticks(np.arange(0, ymax + 0.01, 0.5))
    ax.set_yticks(np.arange(ymin, ymax + 0.01, 0.1), minor=True)
    ax.tick_params(axis="x", length=2, width=0.5)
    ax.tick_params(axis="y", length=2, width=0.5)


def plot_qimage_one(npz_file, settings, bg_file=None, sample_theta=None):
    qx, qz, qimg = load_qimage_npz(npz_file, settings["flip_qx"])
    bg_tag = "nobg"
    if bg_file is not None:
        bg_qx, bg_qz, bg_img = load_qimage_npz(bg_file, settings["flip_qx"])
        if bg_img.shape != qimg.shape:
            raise ValueError(f"Background shape does not match sample shape. Sample={qimg.shape}, background={bg_img.shape}")
        if not np.allclose(bg_qx, qx, rtol=1e-6, atol=1e-8):
            raise ValueError("Background qx axis does not match sample qx axis after flipping/sorting.")
        if not np.allclose(bg_qz, qz, rtol=1e-6, atol=1e-8):
            raise ValueError("Background qz axis does not match sample qz axis after sorting.")
        qimg = qimg - settings["bg_scale"] * bg_img
        bg_tag = f"bgsub_th{sample_theta}_{settings['bg_scale']:g}" if sample_theta else f"bgsub_{settings['bg_scale']:g}"
    im_plot = clean_for_plot(qimg, settings["scale"])
    norm, scale_tag = get_norm(settings["scale"], settings["vmin"], settings["vmax"])
    cmap = get_cmap(settings["cmap"])
    fig, ax = plt.subplots(figsize=(3, 2.75), dpi=settings["dpi"])
    set_common_background(fig, ax, settings["transparent"])
    ax.pcolormesh(qx, qz, im_plot, cmap=cmap, norm=norm, shading="auto")
    ax.set_xlim(settings["xmin"], settings["xmax"])
    ax.set_ylim(settings["ymin"], settings["ymax"])
    ax.set_xlabel(r"$q_x\:\mathrm{(\AA^{-1})}$")
    ax.set_ylabel(r"$q_z\:\mathrm{(\AA^{-1})}$")
    set_image_ticks(ax, settings["xmin"], settings["xmax"], settings["ymin"], settings["ymax"])
    flip_tag = "qxflip" if settings["flip_qx"] else "noflip"
    out_name = f"{Path(npz_file.name).stem}_{flip_tag}_{bg_tag}_{scale_tag}_{settings['cmap']}.png"
    return out_name, fig_to_png_bytes(fig, settings["dpi"], settings["transparent"])


def plot_qphi_one(npz_file, settings):
    q, phi, im = load_qphi_npz(npz_file)
    im_plot = clean_for_plot(im, settings["scale"])
    norm, scale_tag = get_norm(settings["scale"], settings["vmin"], settings["vmax"])
    cmap = get_cmap(settings["cmap"])
    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=settings["dpi"])
    set_common_background(fig, ax, settings["transparent"])
    m = ax.pcolormesh(q, phi, im_plot, cmap=cmap, norm=norm, shading="auto")
    ax.set_xlim(settings["xmin"], settings["xmax"])
    ax.set_ylim(settings["ymin"], settings["ymax"])
    ax.set_xlabel(r"$q\:\mathrm{(\AA^{-1})}$")
    ax.set_ylabel("Azimuthal angle (degrees)")
    ax.set_xticks(np.arange(0, settings["xmax"] + 0.01, 0.5))
    ax.set_xticks(np.arange(settings["xmin"], settings["xmax"] + 0.01, 0.1), minor=True)
    ax.set_yticks(np.arange(0, settings["ymax"] + 0.01, 60))
    ax.set_yticks(np.arange(settings["ymin"], settings["ymax"] + 0.01, 10), minor=True)
    ax.tick_params(axis="x", length=2, width=0.5)
    ax.tick_params(axis="y", length=2, width=0.5)
    cb = fig.colorbar(m, ax=ax)
    cb.set_label("Intensity (a.u.)")
    out_name = f"Qphi_{Path(npz_file.name).stem}_{scale_tag}_{settings['cmap']}.png"
    return out_name, fig_to_png_bytes(fig, settings["dpi"], settings["transparent"])


def plot_azimuthal_one(npz_file, settings):
    q, phi, im = load_qphi_npz(npz_file)
    q_min = settings["qmin"]
    q_max = settings["qmax"]
    q_indices = np.where((q >= q_min) & (q <= q_max))[0]
    if q_indices.size == 0:
        raise ValueError(f"No q values in selected range {q_min} to {q_max} for {npz_file.name}")
    intensity = np.nanmean(im[:, q_indices], axis=1)
    out_csv_name = f"AziProfile_{Path(npz_file.name).stem}_q{q_min:.3f}-{q_max:.3f}.csv"
    csv_bytes = pd.DataFrame({"Angle (deg)": phi, "Intensity": intensity}).to_csv(index=False).encode("utf-8")
    y = np.asarray(intensity, dtype=float).copy()
    if settings["scale"] == "log":
        y[~np.isfinite(y) | (y <= 0)] = np.nan
    else:
        y[~np.isfinite(y)] = np.nan
    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=settings["dpi"])
    set_common_background(fig, ax, settings["transparent"])
    ax.plot(phi, y, lw=1)
    ax.set_xlim(settings["xmin"], settings["xmax"])
    ax.set_ylim(settings["ymin"], settings["ymax"])
    ax.set_yscale(settings["scale"])
    ax.set_xlabel("Azimuthal Angle (degrees)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xticks(np.arange(0, settings["xmax"] + 0.01, 45))
    ax.set_xticks(np.arange(settings["xmin"], settings["xmax"] + 0.01, 5), minor=True)
    ax.tick_params(axis="x", length=2, width=0.5)
    ax.tick_params(axis="y", length=2, width=0.5)
    ax.set_yticks([])
    out_png_name = f"AziProfile_{Path(npz_file.name).stem}_q{q_min:.3f}-{q_max:.3f}.png"
    png_bytes = fig_to_png_bytes(fig, settings["dpi"], settings["transparent"])
    return [(out_csv_name, csv_bytes), (out_png_name, png_bytes)]


def plot_ciravg_one(csv_file, settings):
    q, iq = load_ciravg_csv(csv_file)
    valid = np.isfinite(q) & np.isfinite(iq)
    q = q[valid]
    iq = iq[valid]
    if q.size == 0:
        raise ValueError(f"No valid circular average data found in {csv_file.name}")
    if settings["scale"] == "log":
        iq = np.where(iq > 0, iq, np.nan)
    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=settings["dpi"])
    set_common_background(fig, ax, settings["transparent"])
    ax.plot(q, iq, lw=1)
    ax.set_xlim(settings["xmin"], settings["xmax"])
    ax.set_ylim(settings["ymin"], settings["ymax"])
    ax.set_yscale(settings["scale"])
    ax.set_xlabel(r"$q\:\mathrm{(\AA^{-1})}$")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xticks(np.arange(0, settings["xmax"] + 0.01, 0.5))
    ax.set_xticks(np.arange(settings["xmin"], settings["xmax"] + 0.01, 0.1), minor=True)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=2, width=0.5)
    ax.tick_params(axis="y", length=2, width=0.5)
    out_name = f"CirAvg_{Path(csv_file.name).stem}.png"
    return out_name, fig_to_png_bytes(fig, settings["dpi"], settings["transparent"])


def process_files(preset, sample_files, bg_files, settings, filter_text, progress_callback=None):
    """Process one input at a time and stream outputs directly into a disk-backed ZIP.

    This deliberately avoids retaining all generated PNG/CSV files or the final ZIP
    in RAM, which is essential for large GIWAXS batches on Streamlit Community Cloud.
    """
    files = [f for f in sample_files if filter_text in f.name]
    if preset == "CirAvg":
        filtered = [f for f in files if f.name.lower().endswith('.csv')]
        prioritised = [f for f in filtered if ('cir' in f.name.lower() or 'avg' in f.name.lower())]
        files = prioritised if prioritised else filtered
    else:
        files = [f for f in files if f.name.lower().endswith('.npz')]

    if len(files) == 0:
        raise ValueError(f"No matching files found for preset {preset}.")

    bg_map = {}
    if preset == "Qimage" and settings["subtract_bg"]:
        bg_map = build_background_map(bg_files)
        if len(bg_map) == 0:
            raise ValueError("No usable background .npz files with th... were found.")

    logs = []
    exported = 0
    skipped = 0

    # The ZIP itself lives on temporary disk instead of in a BytesIO object.
    fd, zip_path = tempfile.mkstemp(prefix="giwaxs_export_", suffix=".zip")
    os.close(fd)

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            total = len(files)
            for i, path in enumerate(files, 1):
                try:
                    entries = []
                    if preset == "Qimage":
                        sample_theta = extract_theta_from_name(path.name)
                        bg_path = None
                        if settings["subtract_bg"]:
                            if sample_theta is None:
                                msg = f"No th... incident angle found in sample filename: {path.name}"
                                if settings["skip_missing_bg"]:
                                    skipped += 1
                                    logs.append(f"Skipped: {msg}")
                                    if progress_callback:
                                        progress_callback(i, total, path.name)
                                    continue
                                raise ValueError(msg)
                            bg_path = bg_map.get(sample_theta)
                            if bg_path is None:
                                msg = f"No matching background found for {path.name} at th{sample_theta}"
                                if settings["skip_missing_bg"]:
                                    skipped += 1
                                    logs.append(f"Skipped: {msg}")
                                    if progress_callback:
                                        progress_callback(i, total, path.name)
                                    continue
                                raise ValueError(msg)
                        entries = [plot_qimage_one(path, settings, bg_file=bg_path, sample_theta=sample_theta)]
                    elif preset == "Qphi":
                        entries = [plot_qphi_one(path, settings)]
                    elif preset == "Azimuthal":
                        entries = plot_azimuthal_one(path, settings)
                    elif preset == "CirAvg":
                        entries = [plot_ciravg_one(path, settings)]
                    else:
                        raise ValueError(f"Unknown preset: {preset}")

                    # Immediately write this file's outputs into the ZIP, then drop
                    # the bytes before moving to the next GIWAXS file.
                    for name, content in entries:
                        zf.writestr(name, content)
                    del entries
                    exported += 1
                    logs.append(f"Exported: {path.name}")
                except Exception as e:
                    logs.append(f"Error processing {path.name}: {e}")
                finally:
                    gc.collect()
                    if progress_callback:
                        progress_callback(i, total, path.name)

            zf.writestr("export_log.txt", "\n".join(logs))

        return zip_path, exported, skipped, logs
    except Exception:
        try:
            os.remove(zip_path)
        except OSError:
            pass
        raise


def sample_preview(preset, sample_files, bg_files, settings, filter_text):
    files = [f for f in sample_files if filter_text in f.name]
    if preset == "CirAvg":
        files = [f for f in files if f.name.lower().endswith('.csv')]
    else:
        files = [f for f in files if f.name.lower().endswith('.npz')]
    if not files:
        raise ValueError("No previewable files with current settings.")
    f = files[0]
    if preset == "Qimage":
        bg_map = build_background_map(bg_files) if settings["subtract_bg"] else {}
        theta = extract_theta_from_name(f.name)
        bgf = bg_map.get(theta) if theta is not None else None
        _, png = plot_qimage_one(f, settings, bg_file=bgf, sample_theta=theta)
    elif preset == "Qphi":
        _, png = plot_qphi_one(f, settings)
    elif preset == "Azimuthal":
        entries = plot_azimuthal_one(f, settings)
        png = [content for name, content in entries if name.lower().endswith('.png')][0]
    else:
        _, png = plot_ciravg_one(f, settings)
    return png, f.name


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    if "giwaxs_preset" not in st.session_state:
        st.session_state.giwaxs_preset = "Qimage"
    preset = st.selectbox("Plot type", PRESETS, key="giwaxs_preset")
    apply_preset_defaults(preset)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Reset preset defaults"):
            reset_defaults_force(preset)
            st.rerun()
    with c2:
        st.caption("Qimage/Qphi use .npz. CirAvg uses .csv. Azimuthal is extracted from Qphi .npz.")

    with st.sidebar:
        st.header("Inputs")
        filter_text = st.text_input("Only export files containing", value="", help="Leave blank to export all matching files.")

        if preset == "CirAvg":
            sample_files = st.file_uploader("Upload CirAvg CSV files", type=["csv"], accept_multiple_files=True)
        else:
            sample_files = st.file_uploader("Upload sample NPZ files", type=["npz"], accept_multiple_files=True)

        subtract_bg = False
        bg_files = []
        bg_scale = 1.0
        skip_missing_bg = True
        if preset == "Qimage":
            st.header("Background subtraction")
            subtract_bg = st.checkbox("Subtract blank substrate background by matching incident angle, e.g. th0.050", value=False)
            bg_scale = st.number_input("Background scale factor", value=1.0, step=0.1, format="%.3f")
            skip_missing_bg = st.checkbox("Skip sample if matching background is missing", value=True)
            if subtract_bg:
                bg_files = st.file_uploader("Upload background NPZ files", type=["npz"], accept_multiple_files=True)

        st.header("Plot settings")
        xmin = st.number_input("x min", value=float(st.session_state.giwaxs_xmin), format="%.6g")
        xmax = st.number_input("x max", value=float(st.session_state.giwaxs_xmax), format="%.6g")
        ymin = st.number_input("y min", value=float(st.session_state.giwaxs_ymin), format="%.6g")
        ymax = st.number_input("y max", value=float(st.session_state.giwaxs_ymax), format="%.6g")
        vmin = st.number_input("vmin", value=float(st.session_state.giwaxs_vmin), format="%.6g")
        vmax = st.number_input("vmax", value=float(st.session_state.giwaxs_vmax), format="%.6g")
        scale = st.selectbox("Scale", ["log", "linear"], index=0 if st.session_state.giwaxs_scale == "log" else 1)
        dpi = st.number_input("DPI", value=300, min_value=50, max_value=1200, step=50)

        st.header("Azimuthal extraction")
        qmin = st.number_input("q min", value=0.72, format="%.6g")
        qmax = st.number_input("q max", value=0.78, format="%.6g")
        st.caption("Used only for Azimuthal preset.")

        st.header("Options")
        cmap = st.selectbox("Colormap", CMAPS, index=CMAPS.index("seismic"))
        flip_qx = st.checkbox("Flip qx by multiplying by -1 (Qimage only)", value=True)
        transparent = st.checkbox("Transparent background", value=False)

    settings = dict(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, vmin=vmin, vmax=vmax,
        scale=scale, dpi=int(dpi), qmin=qmin, qmax=qmax, cmap=cmap,
        flip_qx=flip_qx, transparent=transparent,
        subtract_bg=subtract_bg, bg_scale=bg_scale, skip_missing_bg=skip_missing_bg,
    )

    total_upload_bytes = sum(getattr(f, "size", 0) for f in (sample_files or [])) + sum(getattr(f, "size", 0) for f in (bg_files or []))
    total_upload_mb = total_upload_bytes / (1024 ** 2)
    if sample_files:
        st.caption(f"Loaded {len(sample_files)} sample file(s) · approximately {total_upload_mb:.1f} MB currently held by Streamlit.")
        if total_upload_mb >= 350:
            st.warning(
                "Large batch: Streamlit keeps uploaded files in server RAM. The export path is now disk-streamed to reduce the processing spike, "
                "but very large batches can still hit the hosting service's overall memory limit."
            )

    errors = []
    if xmax <= xmin:
        errors.append("x max must be larger than x min.")
    if ymax <= ymin:
        errors.append("y max must be larger than y min.")
    if vmax <= vmin:
        errors.append("vmax must be larger than vmin.")
    if scale == "log" and vmin <= 0:
        errors.append("For log scale, vmin must be > 0.")
    if qmax <= qmin:
        errors.append("q max must be larger than q min.")
    if preset == "Qimage" and subtract_bg and len(bg_files) == 0:
        errors.append("Background subtraction is enabled, but no background NPZ files were uploaded.")

    preview_col, export_col = st.columns([1, 1])
    with preview_col:
        if st.button("Preview first matching file", use_container_width=True):
            if errors:
                for err in errors:
                    st.error(err)
            elif not sample_files:
                st.warning("Upload sample files first.")
            else:
                try:
                    png, preview_name = sample_preview(preset, sample_files, bg_files, settings, filter_text)
                    st.success(f"Preview generated from: {preview_name}")
                    st.image(png)
                except Exception as e:
                    st.error(str(e))

    with export_col:
        if st.button("Build export ZIP", type="primary", use_container_width=True):
            if errors:
                for err in errors:
                    st.error(err)
            elif not sample_files:
                st.warning("Upload sample files first.")
            else:
                try:
                    # Remove the previous temporary export for this session before creating a new one.
                    old_path = st.session_state.get("giwaxs_zip_path")
                    if old_path and os.path.exists(old_path):
                        try:
                            os.remove(old_path)
                        except OSError:
                            pass

                    progress = st.progress(0.0, text="Preparing batch export…")
                    status = st.empty()

                    def _progress(i, total, name):
                        progress.progress(i / total, text=f"Processing {i}/{total}")
                        status.caption(name)

                    zip_path, exported, skipped, logs = process_files(
                        preset, sample_files, bg_files, settings, filter_text, progress_callback=_progress
                    )
                    st.session_state.giwaxs_zip_path = zip_path
                    st.session_state.giwaxs_exported = exported
                    st.session_state.giwaxs_skipped = skipped
                    st.session_state.giwaxs_logs = logs[-50:]
                    progress.progress(1.0, text="Export complete")
                    status.empty()
                    zip_mb = os.path.getsize(zip_path) / (1024 ** 2)
                    st.success(f"Export complete. Exported {exported}, skipped {skipped}. ZIP size: {zip_mb:.1f} MB.")
                except Exception as e:
                    st.error(str(e))

    zip_path = st.session_state.get("giwaxs_zip_path")
    if zip_path and os.path.exists(zip_path):
        def _read_export_zip():
            with open(zip_path, "rb") as fh:
                return fh.read()

        st.download_button(
            "Download export ZIP",
            data=_read_export_zip,
            file_name=f"GIWAXS_{preset}_exports.zip",
            mime="application/zip",
            on_click="ignore",
        )
        with st.expander("Recent export log"):
            for line in st.session_state.get("giwaxs_logs", []):
                st.text(line)

    with st.expander("Notes"):
        st.markdown(
            """
- **Qimage** works directly from `.npz` files containing `qimg`, `qx`, and `qz`.
- **Background subtraction** is applied only to the **Qimage** preset and matches blank-substrate files by `th...` in the filename.
- **Azimuthal** exports both the extracted `.csv` profile and the `.png` plot.
- **CirAvg** reads `.csv` files and prefers files containing `cir` or `avg` in the filename.
- **Large-batch export:** outputs are streamed one-by-one into a temporary disk-backed ZIP to avoid retaining every generated plot in RAM.
            """
        )


if __name__ == "__main__":
    main()
