# BET–XRD Morphology Analyzer

**Scientific Edition for Journal Publication**

An open-source web application for IUPAC-compliant physisorption analysis and advanced X-ray diffraction characterization of porous and nanocrystalline materials. Designed for reproducible research and peer-reviewed publication.

---

## Overview

This application integrates two independent characterization techniques — gas physisorption (BET) and X-ray diffraction (XRD) — into a single analysis pipeline. It produces publication-quality figures, quantitative scientific parameters with error propagation, and a machine-readable report suitable for inclusion in a journal's supporting information.

The software is written in Python 3.11 and runs as a Streamlit application. All computations follow established IUPAC conventions and are individually referenced.

---

## Scientific scope

### BET (Brunauer–Emmett–Teller) Analysis

- **IUPAC-compliant linear range selection** using the Rouquerol criteria
- **Full-covariance error propagation** for the monolayer capacity and the specific surface area
- **t-plot analysis** (Harkins–Jura thickness equation) for micropore volume and external surface area
- **BJH pore size distribution** from the desorption branch
- **IUPAC hysteresis classification** (H1–H4) with scientific interpretation

### XRD Analysis

- **SNIP background subtraction** (Ryan 1988; Morháč 1997)
- **Physical peak validation** with Gaussian, Lorentzian, and pseudo-Voigt fits; requires SNR > 2, physical FWHM, and R² > 0.5
- **Scherrer crystallite size** with instrumental broadening correction
- **Williamson–Hall size–strain separation** with linearity check (R² > 0.85)
- **Crystallinity index** by area ratio (Ruland 1961)
- **Automated phase identification** against COD, Materials Project, and OPTIMADE-compliant providers
- **Size-dependent d-spacing tolerance** for nanocrystalline materials (Scherrer broadening)

### Integrated morphology

- Deterministic, data-driven visualizations (no synthetic or random figures)
- Semi-quantitative phase fractions from matched peak intensities
- BET-XRD internal consistency check with explicit disclaimer when density is unavailable
- Material classification against an internal reference database
- Journal recommendation based on material family and novelty indicators

---

## Installation

### Requirements

- Python 3.11 (see `runtime.txt`)
- pip

### Steps

```bash
git clone https://github.com/badreddinemansouri-eng/BET-XRD-analyzer-USCR-MSPAG.git
cd BET-XRD-analyzer-USCR-MSPAG
pip install -r requirements.txt
streamlit run app.py
