# Dose-Response Curve Simulator (4PL)


This repository contains a simple interactive dashboard to simulate dose-response (4PL) curves and dilution schemes to help with potency assay development.

Files added:
- `functions.py`: helper functions for dilution schemes and 4PL model computation.
- `app.py`: Flask-based interactive dashboard (run with `python app.py`).
- `requirements.txt`: Python package requirements.

Quick start
1. Create a Python environment (recommended).

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the dashboard:

```powershell
python app.py
```

4. Open the URL printed by Dash (usually http://127.0.0.1:8050) in your browser.

Notes and assumptions
- The 4PL model used is: y = D + (A - D) / (1 + (X/C)**B). Inputs A,B,C,D are numeric ranges (min,max). The main plot uses the midpoint of each range. The 16 subplots show the 16 combinations of min/max for A,B,C,D.
- Dilution schemes: even (single factor repeated) or custom (7 comma-separated factors). Both produce an 8-point series including the top concentration.
- This is an early prototype focused on functionality and clarity. Further enhancements are possible (noise, interactive sliders, better validation, recommendation system for dilution schemes).
