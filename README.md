[README_GIWAXS_Plotter_Web_v1.0.md](https://github.com/user-attachments/files/31426971/README_GIWAXS_Plotter_Web_v1.0.md)
# GIWAXS Plotter RL — Web v1.0

A Streamlit-based tool for GIWAXS data processing, visualization, and batch export.

**Developer:** Richard Li

## Supported presets

- **Qimage** — reciprocal-space GIWAXS intensity maps
- **Qphi** — intensity as a function of scattering vector and azimuthal angle
- **Azimuthal** — azimuthal profile extraction over a selected q range
- **CirAvg** — circularly averaged GIWAXS profiles

## Input formats

### Qimage
Uses `.npz` files containing:

- `qimg`
- `qx`
- `qz`

The qx axis can optionally be flipped by multiplying it by -1.

### Qphi
Uses `.npz` files containing q, phi/azimuth, and intensity arrays. The app accepts several common key names and automatically handles a transposed intensity array when dimensions otherwise match.

### Azimuthal
Uses Qphi-style `.npz` files. Intensity is averaged across the selected q range and exported as both:

- `.csv`
- `.png`

### CirAvg
Uses `.csv` files. If `q_ca` and `iq_ca` are present, those columns are used directly; otherwise the first two numeric columns are used.

## Qimage background subtraction

Optional blank-substrate background subtraction is available for Qimage. Sample and background files are matched by the `th...` incident-angle tag in their filenames.

The processed image is calculated as:

`sample - scale_factor × background`

The app checks that the sample and background q grids match before subtraction.

## Plot controls

The interface provides controls for:

- x and y plotting ranges
- `vmin` and `vmax`
- logarithmic or linear intensity scale
- colormap
- qx flipping for Qimage
- transparent background
- export DPI
- q range for Azimuthal extraction
- filename filtering

## Large-batch processing

The export workflow is designed for large GIWAXS datasets. Files are processed one at a time, and generated outputs are written directly into a temporary disk-backed ZIP rather than accumulated in memory.

This substantially reduces the processing-memory overhead for batches containing many NPZ files.

## Export workflow

1. Upload sample files.
2. Choose the preset and plotting settings.
3. Optionally preview the first matching file.
4. Click **Build export ZIP**.
5. Download the completed ZIP.

The ZIP contains the generated plots, extracted CSV files where applicable, and an export log.

## Deployment

Recommended repository structure for Streamlit Community Cloud:

```text
GIWAXS-Plotter-Web/
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Requirements

- Streamlit
- Matplotlib
- NumPy
- pandas
