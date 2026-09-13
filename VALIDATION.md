# Validation Report

This document records validation of the BET-XRD Morphology Analyzer against reference materials with known literature values. The purpose is to allow reviewers and users to confirm that the software reproduces accepted scientific results.

All reference values below are taken from the peer-reviewed literature cited in each section. The software output is compared to those values within the stated tolerance.

---

## 1. BET - Titanium dioxide (anatase)

| Property | Literature value | Source | Tolerance |
|---|---|---|---|
| BET surface area | 50-150 m2/g (commercial nanopowder) | Zhang et al., J. Phys. Chem. C 2010, 114, 2751 | +/-15% |
| Mean pore diameter | 5-15 nm | Same | +/-30% |
| Total pore volume | 0.20-0.40 cm3/g | Same | +/-25% |

**Expected behavior of the software**

- BET linear range selected inside 0.05 <= P/P0 <= 0.35
- C constant > 0
- R-squared of the BET transform > 0.999
- Crystallinity index (XRD) > 0.6 for well-crystallized anatase

---

## 2. XRD - Titanium dioxide (anatase)

| Reflection | 2theta (Cu Ka) | d-spacing (Angstrom) | Relative intensity |
|---|---|---|---|
| (101) | 25.28 | 3.52 | 100 |
| (004) | 37.80 | 2.38 | 20 |
| (200) | 48.05 | 1.89 | 35 |
| (105) | 53.89 | 1.70 | 20 |
| (211) | 55.06 | 1.67 | 15 |

**Reference:** Weirich, T. E.; et al. Acta Cryst. B 2000, 56, 29-35.

**Expected behavior**

- The software should report these d-spacings within 1% relative error
- Scherrer crystallite size should be in the 10-50 nm range for standard nanopowders
- Williamson-Hall should report microstrain < 0.005 for well-crystallized material

---

## 3. XRD - Silicon (Si)

| Reflection | 2theta (Cu Ka) | d-spacing (Angstrom) | Relative intensity |
|---|---|---|---|
| (111) | 28.44 | 3.135 | 100 |
| (220) | 47.30 | 1.920 | 55 |
| (311) | 56.12 | 1.638 | 30 |
| (400) | 69.13 | 1.358 | 5 |

**Reference:** Parrish, W. Acta Cryst. 1960, 13, 838.

**Expected behavior**

- All four reflections detected
- d-spacings within 0.5% of reference
- Crystallinity index > 0.9 for single-crystal-grade silicon powder

---

## 4. BET - SBA-15 mesoporous silica

| Property | Literature value | Source | Tolerance |
|---|---|---|---|
| BET surface area | 700-1000 m2/g | Zhao et al., Science 1998, 279, 548 | +/-10% |
| Total pore volume | 0.8-1.2 cm3/g | Same | +/-15% |
| Mean pore diameter | 6-10 nm | Same | +/-15% |
| Hysteresis type | H1 | IUPAC 2015 | exact |

**Expected behavior**

- Hysteresis loop classified as type H1
- Mesopore fraction > 80% in BJH PSD
- Micropore volume < 0.1 cm3/g

---

## 5. BET - Activated carbon

| Property | Literature value | Source | Tolerance |
|---|---|---|---|
| BET surface area | 800-1500 m2/g | Rouquerol et al., Adsorption by Powders and Porous Solids, 2014 | +/-20% |
| Micropore volume | 0.3-0.6 cm3/g | Same | +/-25% |
| Hysteresis type | H4 (or none) | IUPAC 2015 | qualitative |

**Expected behavior**

- Micropore fraction > 50%
- Crystallinity index (XRD) < 0.2
- Hysteresis class H4 or reversible

---

## 6. XRD - Gold nanoparticles (FCC)

| Reflection | 2theta (Cu Ka) | d-spacing (Angstrom) | Relative intensity |
|---|---|---|---|
| (111) | 38.18 | 2.355 | 100 |
| (200) | 44.39 | 2.039 | 52 |
| (220) | 64.58 | 1.442 | 32 |
| (311) | 77.55 | 1.231 | 36 |

**Reference:** Swanson, H. E.; Tatge, E. NBS Circular 539, 1953.

**Expected behavior**

- All four reflections detected for well-crystallized Au
- Scherrer size matches TEM size within 20% for particles > 10 nm
- Williamson-Hall microstrain < 0.005

---

## 7. Numerical self-consistency tests

These checks are independent of any experimental data and verify the internal mathematical correctness of the software.

| Test | Input | Expected output | Pass criterion |
|---|---|---|---|
| Bragg's law | 2theta = 38.2 deg, lambda = 1.5406 Angstrom | d = 2.355 Angstrom | +/-1% |
| Scherrer | FWHM = 0.02 rad, theta = 15 deg, K = 0.9 | D = 40.3 Angstrom = 4.03 nm | +/-0.1% |
| BET ideal isotherm | q_m = 0.5 mmol/g, C = 150 | Recover both | +/-5% (q_m), +/-10% (C) |
| Williamson-Hall | D = 20 nm, epsilon = 0 | Recover both | +/-20% (D), +/-1e-3 (epsilon) |

These checks are automated in the tests/ directory and are run with:

    pytest tests/ -v

---

## 8. Known limitations

The software is intended for the following material classes:

- Porous metal oxides
- Zeolites and zeotypes
- Mesoporous silica
- Activated carbons
- MOFs
- Nanocrystalline metals and alloys

The software is NOT intended for:

- Single-crystal diffraction analysis
- In-situ or operando time-resolved XRD
- Amorphous materials with no Bragg peaks
- Materials with strong texture or preferred orientation (unless corrected externally)
- Quantitative Rietveld refinement (use GSAS-II, FullProf, or TOPAS instead)

---

## 9. Reproducing this validation

To reproduce the checks in this document on your own machine:

    git clone https://github.com/badreddinemansouri-eng/BET-XRD-analyzer-USCR-MSPAG.git
    cd BET-XRD-analyzer-USCR-MSPAG
    pip install -r requirements.txt
    pytest tests/ -v

All tests should pass. If any test fails, please open an issue with the full error message.

---

## 10. Versioning

| Version | Date | Change |
|---|---|---|
| 3.0.0 | 2024-12-01 | Initial public release |

Subsequent versions will append rows to this table.
