[README_GIWAXS_Plotter_Web_v0.1.md](https://github.com/user-attachments/files/31425755/README_GIWAXS_Plotter_Web_v0.1.md)
# GIWAXS Plotter RL — Web v0.1

A Streamlit web app developed from the newer standalone **GIWAXS Plotter RL** code.

## Developer
Richard Li

## What this app does
This web version is based purely on the newer standalone GIWAXS code, not on the older GIWAXS section inside Python Plotter / PYRL.

Supported presets:
- Qimage
- Qphi
- Azimuthal
- CirAvg

## Important processing difference from old PYRL
This app uses the newer workflow that reads processed q-space `.npz` files directly.

For Qimage, it expects:
- `qimg`
- `qx`
- `qz`

inside the `.npz` file.

That is different from the older PYRL GIWAXS section, which expected separate TIFF + CSV axis files.

## Input expectations
### Qimage
- Sample files: `.npz`
- Required keys: `qimg`, `qx`, `qz`
- Optional blank-substrate background subtraction by matching `th...` in the filename

### Qphi
- Sample files: `.npz`
- The app tries flexible key names for q, phi, and image arrays

### Azimuthal
- Sample files: `.npz`
- Uses the Qphi-style `.npz`
- Extracts the azimuthal profile by averaging intensity over the selected `q` range
- Exports both `.csv` and `.png`

### CirAvg
- Sample files: `.csv`
- Reads `q_ca` and `iq_ca` if present, otherwise uses the first two numeric columns

## Main settings
- x/y range
- vmin / vmax
- log or linear scale
- DPI
- q range for Azimuthal extraction
- colormap
- qx flip for Qimage
- transparent background
- file-name filter

## Background subtraction (Qimage only)
If enabled, the app:
1. reads sample and blank substrate `.npz` files,
2. extracts incident angle from `th...` in the filename,
3. matches sample/background by that angle,
4. subtracts:

`sample - scale_factor × background`

## How to run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## GitHub / Streamlit Community Cloud structure
```text
GIWAXS-Plotter-Web/
├── streamlit_app.py
├── requirements.txt
└── README.md
```
