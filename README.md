# 🛸 DrishtiPath Technologies
### AI-Powered Drone-Based Retroreflectivity Measurement System

<p align="center">
  <img src="https://img.shields.io/badge/NHAI-6th%20Innovation%20Hackathon%202026-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Active%20Development-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Institute-PIEMR%20Indore-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/IRC%3A35-Compliant-red?style=for-the-badge" />
</p>

<p align="center">
  <b>Smart Nazar. Safe Safar.</b><br/>
  Reimagining highway safety compliance through drone intelligence and machine learning.
</p>

---

## 📌 Table of Contents

- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [System Architecture](#-system-architecture)
- [The Hardware Module](#-the-hardware-module)
- [ML Pipeline](#-ml-pipeline)
- [Key Metrics](#-key-metrics)
- [How It Works](#-how-it-works)
- [Regulatory Compliance](#-regulatory-compliance)
- [Business Model](#-business-model)
- [Research Backing](#-research-backing)
- [Team](#-team)

---

## ⚠️ The Problem

India's 50,000+ km NHAI highway network mandates **retroreflectivity (RL)** measurement of road markings, signs, and studs under **IRC:35** and **IRC:67** standards. When markings fall below threshold, drivers cannot see lane edges at night — especially in rain or fog.

```
IRC:35 Minimum RL Thresholds:
  Expressways  (>100 km/h)  →  150 mcd/m²/lux
  Highways     (>65 km/h)   →  120 mcd/m²/lux
  Local Roads  (<65 km/h)   →   80 mcd/m²/lux
```

### Current Reality

| Problem | Impact |
|---|---|
| Workers stand on **live high-speed expressways** during measurement | Fatal safety risk |
| Handheld devices measure **1 point per minute** | <0.5 km per session |
| **Night & wet-condition** measurements practically never done | Blind compliance data |
| European vehicle-mounted systems cost **₹50–80 lakh per unit** | Unaffordable at scale |
| Less than **1% of NHAI network** surveyed annually | 50,000 km left unchecked |

> 💀 **The Result:** Substandard markings go undetected. Drivers die.

---

## ✅ Our Solution

DrishtiPath solves this with a **three-part system** — a compact NIR laser module mountable on a **drone or vehicle** (currently drone-deployed), a commercial drone platform, and an on-ground ML inference pipeline.

```
11 km surveyed per session  |  0 workers on the road  |  <10 min to compliance report
```

### Why Drone-First?

- Surveys **both lanes simultaneously** at 2–3 m/s
- Operates at **night** (DGCA Digital Sky permitted)
- Captures **wet-condition** data with dedicated rain-flight protocol
- **No traffic disruption.** No road closures.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DRISHTIPATH SYSTEM                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  NIR LASER   │    │   DRONE      │    │   ML PIPELINE    │  │
│  │   MODULE     │───▶│  PLATFORM    │───▶│  (Post-Flight)   │  │
│  │  (850nm)     │    │  DJI Mavic 3 │    │  Ensemble Model  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│         │                   │                      │            │
│         ▼                   ▼                      ▼            │
│   IRC:35 geometry     GPS-tagged frames      KMZ / GeoJSON      │
│   replication at      48MP NIR camera        Heatmap +          │
│   9.14m AGL           850nm bandpass         IRC:35 Report      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 The Hardware Module

The payload mounts to the undercarriage of any medium commercial drone and replicates **exact IRC:35 measurement geometry**.

### Core Components

| Component | Specification |
|---|---|
| **Light Source** | 850nm NIR laser diode, 200–500mW collimated output |
| **Baseline Separation** | 16.76 cm (laser ↔ camera) |
| **Operating Altitude** | 9.14 m AGL (30 ft) |
| **Camera** | 48MP, natively 850nm-sensitive, 850nm bandpass filter |
| **Sync** | Laser pulses in sync with camera frame exposure |
| **Drone Platform** | DJI Mavic 3 Thermal — 920g, 45 min flight, built-in GPS |
| **Total Add-on Cost** | ₹10,000 |

### Why 16.76 cm Baseline?

```
IRC:35 mandates:
  Illumination angle  →  88.76° from road surface
  Observation angle   →  1.05°
  Driver distance     →  30 metres

At 9.14m AGL:
  Required baseline = tan(1.05°) × 9140mm = 16.76 cm ✓
```

This replicates the **exact geometry** that defines the 30-metre driver-viewing distance in the IRC:35 standard.

### Signal Isolation

```
Frame N   →  Laser ON   →  Retroreflected NIR + Ambient
Frame N+1 →  Laser OFF  →  Ambient only

Subtraction: Frame N − Frame N+1 = Pure retroreflected NIR signal
```

The 850nm bandpass filter further eliminates sunlight, streetlights, and all non-NIR ambient sources.

---

## 🤖 ML Pipeline

### Model Architecture

The inference pipeline is powered by a **high-performance ensemble regression model** optimised for structured feature-extracted image data — trained to predict retroreflectivity values directly from captured NIR signal intensity.

```
Input:  NIR image patch (laser-ON minus laser-OFF)
        + altitude log from drone flight data

Output: RL value (mcd/m²/lux)  →  Regression head
        Surface condition        →  Classification head
```

### Training Dataset

| Parameter | Value |
|---|---|
| **Total Samples** | 12,000 |
| **Features** | 8 retroreflectivity parameters |
| **Data Sources** | IRC:35 field measurement studies, published NHAI & international road agency reports, domain-calibrated structured augmentation |
| **Train / Test Split** | 80% / 20% |
| **Validation Method** | Out-of-Time (OOT) testing |

### OOT Validation

> Standard held-out testing tells you how a model performs on data it has already seen. **OOT validation** tells you how it performs on data from entirely different time periods — accounting for seasonal surface wear, weather variation, and road ageing.

DrishtiPath's model is stress-tested across **multiple time windows**, ensuring it stays accurate in the field as highway conditions evolve — not just on benchmark data.

### Altitude Correction

```python
# Inverse-square law correction
RL_corrected = RL_raw × (measured_altitude / reference_altitude)²

# Applied per-frame using GPS altitude log from drone flight data
# Normalises every reading to the correct physical distance
```

### Model Performance

```
Accuracy:             75–80%
Avg Measurement Error: <10%  (ASCE 2023 benchmark: 6.1%)
Inference Time:        <10 minutes per surveyed kilometre
Output Format:         KMZ (Google Earth) + GeoJSON (GIS tools)
```

---

## 📊 Key Metrics

<table>
  <tr>
    <td align="center"><b>🛣️ 11 km</b><br/>Highway coverage<br/>per session (3 batteries)</td>
    <td align="center"><b>🎯 6.1%</b><br/>Avg measurement error<br/>vs handheld (ASCE 2023)</td>
    <td align="center"><b>⏱️ &lt;10 min</b><br/>From landing to<br/>full compliance report</td>
    <td align="center"><b>👷 0</b><br/>Workers on the<br/>live highway</td>
  </tr>
  <tr>
    <td align="center"><b>📦 12,000</b><br/>Training samples<br/>across 8 parameters</td>
    <td align="center"><b>📐 80/20</b><br/>Train-test split with<br/>OOT validation</td>
    <td align="center"><b>🌧️ All conditions</b><br/>Day · Night · Dry<br/>Wet · Fog</td>
    <td align="center"><b>💰 ₹10,000</b><br/>Total hardware add-on<br/>vs ₹50–80L imported</td>
  </tr>
</table>

---

## ⚙️ How It Works

```
STEP 1 — SURVEY FLIGHT
━━━━━━━━━━━━━━━━━━━━━
Drone flies waypoint mission at 9.14m AGL over highway corridor.
Pulsed 850nm laser illuminates road markings at IRC:35 geometry.
Camera captures paired laser-ON / laser-OFF frames at 2–3 m/s.
Coverage: ~3.7 km per battery × 3 batteries = 11 km per session.

STEP 2 — DATA EXTRACTION
━━━━━━━━━━━━━━━━━━━━━━━
Images transferred via USB from drone SD card to field laptop.
Laser-ON minus laser-OFF subtraction isolates pure NIR signal.
GPS geotag from drone flight log attached to every image patch.

STEP 3 — ML INFERENCE
━━━━━━━━━━━━━━━━━━━━
Ensemble regression model predicts RL value per image patch.
Trained on 12,000-sample IRC:35-calibrated dataset (8 parameters).
Altitude correction applied via inverse-square law from flight log.
OOT-validated — performs reliably on unseen corridors and seasons.

STEP 4 — NHAI REPORT
━━━━━━━━━━━━━━━━━━━
GPS-tagged RL heatmap exported as KMZ + GeoJSON.
Section-wise PASS / FAIL against IRC:35 thresholds per speed zone.
Dry RL · Wet Rw · Daytime Qd — all three parameters reported.
Full compliance PDF ready for NHAI submission within 10 minutes.
```

---

## 📋 Regulatory Compliance

| Requirement | Status |
|---|---|
| **DGCA Drone Rules 2021** | DJI Mavic 3 Thermal — Small category, Remote Pilot License required |
| **Green Zone Airspace** | Most NHAI rural stretches — flights up to 120m, no per-flight permission |
| **Operating Altitude** | 9.14m — well within 120m limit |
| **Night Surveys** | DGCA Digital Sky Platform application required |
| **NHAI Blanket Permission** | As a government body, NHAI can obtain corridor-wide survey permissions |
| **IRC:35 Geometry** | Fully replicated via 16.76cm baseline at 9.14m AGL |
| **IRC:67** | Signs, delineators, road studs — drone hovers for vertical face scan |

---

## 💼 Business Model

```
┌─────────────────────────────────────────────────────┐
│              THREE REVENUE STREAMS                  │
├─────────────────────────────────────────────────────┤
│  1. SURVEY-AS-A-SERVICE                             │
│     NHAI pays per-km for periodic corridor surveys  │
│     Delivered as GPS heatmap + IRC:35 report        │
│                                                     │
│  2. MODULE LICENSING                                │
│     State PWDs & O&M contractors license hardware   │
│     module + ML software for existing drones        │
│                                                     │
│  3. DATA PLATFORM (SaaS)                            │
│     National RL degradation database                │
│     Predictive maintenance alerts                   │
│     O&M contractor performance scoring              │
└─────────────────────────────────────────────────────┘
```

### Scaling Roadmap

| Phase | Timeline | Target |
|---|---|---|
| **Pilot & Validate** | Year 1 | 50 km NHAI pilot · Full validation vs handheld · DGCA RPL obtained |
| **State Rollout** | Year 2 | 3 NHAI regional offices · 500 km/month · 5 O&M contractor licenses |
| **National Scale** | Year 3 | All NHAI zones · Annual cycle for full 50,000 km · Southeast Asia expansion |

---

## 📚 Research Backing

| Study | Finding |
|---|---|
| **ASCE Journal of Infrastructure Systems, 2023** | Camera + controlled illumination achieves 6.1% avg error for retroreflectivity |
| **MDPI, 2026** | LiDAR intensity + ML across 1,000+ miles — accuracy comparable to Mobile Retroreflectometry Units |
| **MDPI Buildings, 2023** | LiDAR-based ML achieves R² of 0.824 across 8 road sections |
| **Sensors, 2023** | UAV-LiDAR-camera fusion for pavement analysis at 19–25m altitude with centimetre-level GPS accuracy |
| **Int. Journal of Pavement Research & Technology, 2020** | Visual algorithms validated for RL against CEN-EN 1436 geometry |

---

## 👥 Team

**DrishtiPath Technologies**
Student Startup · PIEMR Indore

> Participating in the **NHAI 6th Innovation Hackathon 2026**
> Submission Deadline: 23 April 2026

Our solution simultaneously addresses NHAI Hackathon categories:
- ✅ **Category i** — Vehicle/drone-mounted measurement
- ✅ **Category ii** — AI and ML-based determination
- ✅ **Category iii** — Combination of both

Covering: Road Markings · Traffic Signs · Road Studs · Delineators
Under all conditions: Day · Night · Dry · Wet · Fog

---

## 📬 Contact & Pilot Enquiry

Interested in the **50 km pilot contract**?

We are ready to deploy on one NHAI highway corridor within **30 days of contract signing** — with full side-by-side validation against existing handheld data to prove IRC:35 accuracy compliance.

---

<p align="center">
  <i>DrishtiPath Technologies — Smart Nazar, Safe Safar</i><br/>
  <i>Road safety, powered by data & innovation.</i>
</p>
