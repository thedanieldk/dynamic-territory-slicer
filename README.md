# Territory Slicer

Interactive Streamlit app for testing where to draw the Enterprise vs. Mid Market account threshold and seeing how that changes rep territory balance.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects these files in the project root:

- `GTM-Engineer Challenge-Daniel - Accounts.csv`
- `GTM-Engineer Challenge-Daniel - Reps.csv`

## How it works

1. Move the employee threshold slider.
2. Accounts with employee count greater than or equal to the threshold are tagged `Enterprise`.
3. Accounts below the threshold are tagged `Mid Market`.
4. Enterprise accounts are distributed only to Enterprise reps.
5. Mid Market accounts are distributed only to Mid Market reps.

The reassignment engine balances ARR, not account count. For each segment, it sorts accounts by ARR from largest to smallest, then assigns each account to the rep with the lowest current assigned ARR. This greedy load-balancing approach is a practical fit for territory planning because it prioritizes revenue potential and handles large accounts first.

The app also includes a before/after comparison by rep. Current ownership is calculated from `Current_Rep`, while proposed ownership is calculated from the new assignment. This shows how much ARR and how many accounts each rep gains or loses at a given threshold, plus contextual risk and marketer-count changes.

The before/after table is ranked within each segment. In ARR-only mode, reps are ranked by net ARR gained. When the advanced risk toggle is enabled, reps are still ranked by net ARR gained first, but if ARR gains are within 5% of that segment's average new ARR per rep, lower risk-load change ranks higher.

An advanced risk toggle can balance risk after ARR. ARR remains the primary equity metric; when reps are already close on ARR, the model simulates the assignment and chooses the rep that creates the lowest resulting risk-load spread across the segment. The app only uses the risk-aware result for a segment when it improves risk-load spread versus ARR-only assignment. Marketer count is shown as context and is not used for assignment.

## Demo checklist

- Load the app.
- Show the attached account and rep data are being used.
- Move the employee threshold up and down.
- Point out that account segmentation, rep assignments, and charts update immediately.
- Show the before/after table to explain how disruptive each threshold would be.
- Optionally enable the advanced risk toggle to show how risk can be spread after ARR balance.
- Explain that the balancing logic uses ARR so the territories are equitable by revenue potential.
