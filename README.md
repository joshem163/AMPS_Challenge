# Temporal Graph Transformers for PMU-Based Cyberattack Detection

This repository contains the official implementation of the paper:

**Temporal Graph Transformers for PMU-Based Cyberattack Detection in Power Grids**  
Md Joshem Uddin**, Damilola Olojede**, Roshni Anna Jacob*,  
Kenneth Hutchison†, Baris Coskunuzer*, Jie Zhang*  

( ** Co-first authors; †Pennsylvania State University*)

---

## 📄 Paper

**Title:** Temporal Graph Transformers for PMU-Based Cyberattack Detection in Power Grids  
**Venue:** IEEE PES General Meeting (PESGM 2026)  
**Focus:** Time-series cyberattack detection and temporal localization using PMU data

This work formulates PMU-based cyberattack detection as a **temporal graph classification problem** and introduces a **Temporal Graph Transformer (TGT)** to accurately identify attack onset and duration in power grids.

---

## 🔍 Overview

Modern power grids rely heavily on **Phasor Measurement Units (PMUs)** for real-time monitoring and control. However, PMU measurements are vulnerable to **false data injection attacks (FDIAs)** that can evade traditional bad-data detection techniques.

Instead of coarse binary system-level classification, this work focuses on:

- **Temporal detection of cyberattacks**
- **Precise localization of attack onset and duration**
- **Distinguishing cyberattacks from natural transients**

The proposed **Temporal Graph Transformer (TGT)** captures:
- **Spatial structure** of the power grid
- **Temporal dependencies** across PMU measurement windows
- **Long-range correlations** using self-attention

---

## Key Contributions

- Formulation of PMU cyberattack detection as a **temporal graph classification task**
- A **sliding-window graph representation** for high-frequency PMU data
- A **Transformer-based architecture** for modeling long-range temporal dependencies
- Accurate **attack-onset localization** under natural transients
- Winning solution for **Phase I of the NSF AMPS Cyberattack Detection Challenge**

---

##  Methodology

### Problem Formulation
- Power grid represented as a graph:
  - Nodes: Buses
  - Edges: Transmission lines and transformers
- Node features: PMU measurements (voltage magnitude, voltage angle, frequency)
- Objective: Predict attack probability for each time window

### Sliding Window Representation
- Sampling rate: 40 Hz
- Window length: 5 time steps
- Stride: 5
- Each window represented as a flattened feature vector
- Binary labeling via majority voting within each window

---

## 🧩 Model Architecture: Temporal Graph Transformer (TGT)
<img width="3918" height="2121" alt="FLowChatr_pnm-1" src="https://github.com/user-attachments/assets/b9a8e25e-1c08-4645-9ebb-8b0f7160ecf6" />

## 🚀 Running the Code 

# Train the Temporal Graph Transformer
python train.py to get the trained model and then "Demo_Training_with_Orig_Prediction.ipynb" to get the prediction
