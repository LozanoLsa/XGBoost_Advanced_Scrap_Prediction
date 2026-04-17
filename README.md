# Assembly Scrap Intelligence — Multi-Factor Defect Prediction via XGBoost

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/XGBoost_Advanced_Scrap_Prediction/blob/main/12_XGBoost_Advanced_Scrap_Prediction.ipynb)

> *"When twelve variables interact to produce a defect, you need a model that thinks about all twelve simultaneously."*

---

## 🎯 Business Problem

In high-mix electronics and mechatronics assembly lines, scrap is the most expensive quality failure: a completed unit that cannot be reworked, recovered, or sold. Unlike dimensional defects that trigger a rework loop, scrap represents total sunk cost — materials, machine time, operator time, and downstream inspection all gone.

The challenge is that assembly scrap is **multivariate and nonlinear**. A torque deviation alone rarely creates scrap. It is the combination of under-torque, elevated line vibration, and a novice operator on a shift-change night that pushes a unit over the edge. Classical SPC control charts, monitoring variables individually, systematically miss these interaction effects — they detect the symptom after the scrap has been produced, not the combination that caused it.

**XGBoost captures what control charts cannot.** As a gradient-boosted ensemble of decision trees, it models interactions, nonlinearities, and threshold effects natively. It asks every feature jointly: *"given all current process conditions, what is the probability this unit will be scrapped?"* — before the unit is completed.

The model also demonstrates a critical insight that applies to any classification problem in operations: **threshold selection is a business decision, not a statistical one**. The tradeoff between catching more scrap (high recall) and generating more false alarms (low precision) depends on the relative cost of a missed defect versus a blocked good unit — a number the quality engineer owns, not the data scientist.

---

## 📊 Dataset

- **10,000 assembly cycles** from a mechatronics module line over a 60-day production window
- **Target:** `is_scrap` — binary (0 = OK · 1 = Scrap)
- **Scrap rate:** 30.3%  |  **Class ratio OK:Scrap** = 2.3:1
- **Sources:** Torque controller PLC · Press sensor log · Environmental monitoring · MES operator entries

| Column | Type | Description |
|---|---|---|
| `day_of_week` | int | 0 = Monday … 6 = Sunday |
| `applied_torque_nm` | float | Torque applied to fasteners (Nm) |
| `screwdriving_speed_rpm` | float | Screw driver rotation speed (rpm) |
| `motor_current_a` | float | Driver motor current draw (A) |
| `press_pressure_bar` | float | Press station pressure (bar) |
| `operator_cycle_time_s` | float | Operator task cycle time (s) |
| `relative_humidity_pct` | float | Ambient humidity in assembly bay (%) |
| `shop_floor_temp_c` | float | Ambient temperature (°C) |
| `line_vibration_mm_s` | float | Assembly line vibration amplitude (mm/s) |
| `operator_experience_yrs` | int | Years of experience for the assigned operator |
| `shift_change` | int | 1 = cycle occurs during shift changeover window |
| `material_batch` | int | Material batch ID (1–50) |
| `is_scrap` | int | **Target** — 0 = OK · 1 = Scrap |

### Data Origin (Real-World Perspective)

| Variable(s) | Source System | Notes |
|---|---|---|
| `applied_torque_nm`, `screwdriving_speed_rpm`, `motor_current_a` | Torque Controller PLC | Logged per fastening cycle — waveform data summarised to peak torque and speed |
| `press_pressure_bar` | Press Station PLC / Pressure Transducer | Logged at each press stroke — peak pressure per cycle |
| `operator_cycle_time_s` | MES / Barcode Scanner | Timestamp delta between unit scan-in and scan-out at the assembly station |
| `relative_humidity_pct`, `shop_floor_temp_c` | Environmental Monitoring System | Bay-level sensors logged every 5 minutes — joined to production records by timestamp |
| `line_vibration_mm_s` | Vibration Sensor / SCADA | RMS vibration amplitude from accelerometer mounted on the assembly frame |
| `operator_experience_yrs` | HR System / MES | Years-of-service lookup from the operator badge ID logged at station login |
| `shift_change` | MES / Shift Schedule | Binary flag derived from production timestamp vs shift boundary table |
| `material_batch` | ERP / Goods Receipt | Batch ID from the material traceability record — linked to the production order |
| `is_scrap` | MES / Quality Station | **TARGET** — scrap disposition recorded at the end-of-line quality gate |

> In real-world operations, assembling this dataset requires joining five separate systems on a combination of timestamp, operator badge ID, and material batch ID. The scrap decision at end-of-line may be recorded up to 30 minutes after the assembly event — creating a time-alignment challenge that must be solved before the first row of training data can be built.

---

## 🤖 Model

**Algorithm:** XGBoost (Extreme Gradient Boosting) — Binary Classification  
`xgboost.XGBClassifier`

XGBoost builds decision trees **sequentially**, each one correcting the errors of the previous. The mechanism that makes it effective on production assembly data is its focus on difficult cases: trees are weighted to pay attention to the samples near the decision boundary — the ambiguous units that earlier trees consistently misclassified. After 200 iterations, the ensemble has specialised attention across the full risk landscape.

**Why not logistic regression or Random Forest?**
- Logistic regression models a linear boundary — it cannot capture the threshold effect where scrap risk jumps nonlinearly when vibration crosses ~3.5 mm/s, or the bimodal torque pattern where both too-low and too-high values increase risk.
- Random Forest builds trees independently from bootstrap samples — it does not learn from its own errors iteratively.
- XGBoost builds each tree to correct the residuals of the entire ensemble so far. That is what makes it effective on data with complex interaction effects.

**No StandardScaler required:** XGBoost is scale-invariant — tree splits are based on rank-ordered thresholds, not absolute distances.

**Key hyperparameters:**
- `n_estimators = 200` — sequential trees
- `max_depth = 4` — limits overfitting
- `learning_rate = 0.1` — shrinkage per tree
- `subsample = 0.8` / `colsample_bytree = 0.8` — stochastic regularisation
- `stratify=y` in train/test split — preserves 30.3% scrap rate in both sets

---

## 📈 Key Results

| Metric | Default (thr=0.50) | Operational (thr=0.25) | Meaning |
|---|---|---|---|
| **ROC-AUC** | **0.593** | 0.593 | Threshold-independent discrimination |
| **Recall** | 14% | **69%** | % of actual scrap units caught before dispatch |
| **Precision** | 47% | 34% | % of flagged units that are truly scrap |
| **F1** | 0.22 | **0.46** | Harmonic mean — operational threshold wins |
| **Accuracy** | 69% | 50% | Not the primary metric for imbalanced scrap data |

**Train:** 8,000 cycles · **Test:** 2,000 cycles · `random_state=42, stratify=y`

**ROC-AUC = 0.593** reflects a realistic assembly dataset where scrap has genuine stochastic components — a unit can fail even under ideal conditions. AUC > 0.5 confirms real signal. The practical value is at threshold 0.25: the model catches **69% of scrap units** before they leave the line.

**Confusion matrix (threshold = 0.25, n=2,000 test cycles):**

| | Predicted OK | Predicted Scrap |
|---|---|---|
| **Actual OK** | 589 (TN) | 806 (FP) |
| **Actual Scrap** | 190 (FN) | 415 (TP) |

---

## 🔍 Feature Importance (XGBoost Gain)

| Feature | Gain | Role |
|---|---|---|
| `shift_change` | 0.138 | Highest gain — shift boundaries create the clearest tree splits |
| `line_vibration_mm_s` | 0.101 | Strongest continuous process driver — directional, threshold effect |
| `relative_humidity_pct` | 0.085 | Environmental contamination risk — adhesive and connector sensitivity |
| `press_pressure_bar` | 0.083 | Press station process quality |
| `applied_torque_nm` | 0.079 | Bimodal risk: too low AND too high both increase scrap — nonlinear |
| `operator_cycle_time_s` | 0.077 | Proxy for operator stress and attention — faster ≠ always better |
| `operator_experience_yrs` | 0.075 | Threshold effect: novice (0–1y) 35.3% scrap vs senior 29.5% |
| `screwdriving_speed_rpm` | 0.075 | Speed interacts with torque — neither alone is sufficient |
| `material_batch` | 0.074 | Batch quality variation not encoded — numerical ID is a proxy |
| `shop_floor_temp_c` | 0.074 | Thermal expansion effects on connector engagement |
| `motor_current_a` | 0.074 | Current draw as torque proxy — correlated with applied_torque_nm |
| `day_of_week` | 0.065 | Friday/Monday temporal pattern — human factor and staffing |

**Key scrap rate patterns from EDA:**
- Shift changeover cycles: **34.6%** scrap vs normal shift **27.9%**
- Novice operators (0–1 yr): **35.3%** scrap vs Senior (10+yr): **29.5%**
- Very High vibration (>3.5 mm/s): significantly above baseline scrap rate

---

## 🔧 Threshold as a Business Decision

The threshold choice cannot be made statistically. It requires a cost function:

| Cost Item | Typical Value |
|---|---|
| Cost of missed scrap (FN) | Unit cost + potential warranty/recall impact downstream |
| Cost of false alarm (FP) | Manual inspection time + possible line delay |

**If the assembly unit costs €500 in materials and labour**, and the downstream risk of a slipped scrap is €2,000: the break-even precision is 500/(500+2,000) = **20%**. At threshold 0.25, this model achieves **34% precision** — well above break-even. The model is economically justified at this threshold.

---

## 🔧 Simulation & Scenarios

| Scenario | Conditions | P(Scrap) | Decision |
|---|---|---|---|
| **A — Controlled** | Torque 1.85 Nm · Vibration 1.0 mm/s · 5yr exp · Normal shift · 40% RH | **9.1%** | ✅ OK to proceed |
| **B — All Risks Active** | Torque 1.4 Nm · Vibration 4.8 mm/s · 0yr exp · Shift change · 70% RH | **92.8%** | ⚠ Flag as SCRAP RISK |
| **C — Process Fixed** | Torque 1.85 Nm · Vibration 1.5 mm/s · 0yr exp · Shift change · 70% RH | **62.0%** | ⚠ Flag as SCRAP RISK |

Correcting torque and vibration alone (B→C) **reduces risk by 31 percentage points**. The residual 62% risk in Scenario C reflects that human factors (novice operator on shift change) cannot be corrected by process adjustment alone — recommendation: pair novice operators with mentors during shift changeovers.

---

## 🗂️ Repository Structure

```
XGBoost_Advanced_Scrap_Prediction/
├── 12_XGBoost_Advanced_Scrap_Prediction.ipynb   ← Notebook (no outputs)
├── assy_scrap_data.csv                            ← Full dataset
├── README.md
└── requirements.txt
```

> 📦 **Full Project Pack** — complete dataset (10,000 records), notebook with full outputs, SHAP analysis, presentation deck (PPTX), and `app.py` scrap risk simulator available on [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab:** Click the badge above.

**Option 2 — Local:**
```bash
pip install -r requirements.txt
jupyter notebook 12_XGBoost_Advanced_Scrap_Prediction.ipynb
```

---

## 💡 Key Learnings

1. **Threshold selection is a business decision, not a model decision.** Moving from 0.50 to 0.25 triples recall (14% → 69%) at the cost of precision (47% → 34%). Which trade-off is correct depends on the cost of a missed scrap vs. the cost of a false alarm — numbers the quality engineer owns, not the notebook.

2. **XGBoost captures the interactions that control charts miss.** A single low torque reading is not alarming. Low torque + high vibration + novice operator on a shift change night is a different story. XGBoost finds the joint condition; SPC charts find it one variable at a time — after the scrap has been produced.

3. **AUC = 0.593 on real assembly data is not failure — it is calibration.** Human assembly has genuine randomness: a unit assembled under identical conditions may or may not fail. AUC > 0.5 confirms the model extracts real signal. Achieving 0.95 AUC on production data should raise skepticism, not admiration.

4. **Operator experience shows a nonlinear threshold effect.** Novice operators (0–1 year) produce 35.3% scrap vs. 29.5% for experienced ones — a 5.8-point gap. But the effect flattens sharply after year 2. This is not captured by any linear model. XGBoost finds the threshold natively through its tree structure.

5. **The material_batch feature is a proxy, not a feature.** Encoding batch as a numerical ID from 1–50 means the model cannot distinguish a critical batch with a supplier deviation from a nominal one with the same number. Enriching this feature with incoming inspection quality metadata is the single highest-ROI improvement available for the next model iteration.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
