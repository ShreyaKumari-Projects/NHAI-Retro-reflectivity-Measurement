!pip install scikit-learn pandas numpy opencv-python-headless matplotlib seaborn joblib pillow -q

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

print("✅ All libraries imported!")

# ── Upload your CSV file when prompted ──
from google.colab import files
print("📂 Please upload your dataset CSV file...")
uploaded = files.upload()

# Load it
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n📋 Columns found:", list(df.columns))
print("\n📊 First 3 rows:")
display(df.head(3))

print("\n🏷️ Label distribution:")
print(df['label'].value_counts())

print("\n🌦️ Condition distribution:")
print(df['condition'].value_counts())

print("\n📈 Feature statistics:")
display(df.describe().round(2))

# Standardize column names (lowercase, strip spaces)
df.columns = df.columns.str.strip().str.lower()

# Expected columns — adjust if yours differ slightly
FEATURE_COLS = ['rl', 'qd', 'brightness', 'contrast', 'blur', 'visibility_score']
LABEL_COL    = 'label'
COND_COL     = 'condition'

# Verify all columns exist
missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
if missing:
    print(f"⚠️ Missing columns: {missing}")
    print("Available columns:", list(df.columns))
else:
    print("✅ All expected columns found!")

# Standardize label names
df[LABEL_COL] = df[LABEL_COL].str.strip().str.capitalize()
print("\n🏷️ Labels after cleaning:", df[LABEL_COL].unique())

# Drop nulls
before = len(df)
df.dropna(subset=FEATURE_COLS + [LABEL_COL], inplace=True)
print(f"🧹 Dropped {before - len(df)} null rows. Remaining: {len(df)}")

# ── IRC:35 Thresholds (used later in report) ──
IRC_THRESHOLDS = {
    'rl':               {'min': 120,  'unit': 'mcd/lux/m²', 'name': 'Retroreflectivity (RL)'},
    'qd':               {'min': 100,  'unit': 'mcd/lux/m²', 'name': 'Daytime Visibility (QD)'},
    'brightness':       {'min': 80,   'unit': 'px intensity','name': 'Brightness'},
    'contrast':         {'min': 40,   'unit': 'score',       'name': 'Contrast'},
    'blur':             {'min': 20,   'unit': 'score',       'name': 'Sharpness (anti-blur)'},
    'visibility_score': {'min': 55,   'unit': 'score',       'name': 'Overall Visibility'},
}

print("\n✅ IRC Thresholds set:")
for k, v in IRC_THRESHOLDS.items():
    print(f"   {v['name']:30s} ≥ {v['min']} {v['unit']}")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Dataset Exploration", fontsize=16, fontweight='bold')

# Label distribution
ax = axes[0][0]
colors = {'Good': '#2ecc71', 'Moderate': '#f39c12', 'Poor': '#e74c3c'}
df[LABEL_COL].value_counts().plot(
    kind='bar', ax=ax,
    color=[colors.get(l, 'gray') for l in df[LABEL_COL].value_counts().index]
)
ax.set_title("Label Distribution")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=0)

# Condition distribution
ax = axes[0][1]
df[COND_COL].value_counts().plot(kind='bar', ax=ax, color='steelblue')
ax.set_title("Condition Distribution")
ax.tick_params(axis='x', rotation=45)

# Feature distributions
for i, feat in enumerate(FEATURE_COLS):
    row, col = divmod(i + 2, 4)
    ax = axes[row][col]
    for label, color in colors.items():
        subset = df[df[LABEL_COL] == label][feat]
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color)
    ax.axvline(IRC_THRESHOLDS[feat]['min'], color='black',
               linestyle='--', linewidth=1.5, label='IRC Min')
    ax.set_title(f"{feat.upper()} Distribution")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("dataset_exploration.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Visualization saved!")

# ── Prepare features and labels ──
X = df[FEATURE_COLS].values
y = df[LABEL_COL].values

# Encode labels: Good=0, Moderate=1, Poor=2
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── Build Pipeline: Scaler + Model ──
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
               ))
])

# ── Train ──
print("\n⏳ Training model...")
model_pipeline.fit(X_train, y_train)
print("✅ Training complete!")

# ── Evaluate ──
y_pred = model_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {acc*100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=le.classes_))

# ── Cross-validation ──
cv_scores = cross_val_score(model_pipeline, X, y_enc, cv=5, scoring='accuracy')
print(f"📊 5-Fold Cross-Validation: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix", fontsize=13)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ── Feature Importance ──
importances = model_pipeline.named_steps['clf'].feature_importances_
plt.figure(figsize=(8, 4))
bars = plt.barh(FEATURE_COLS, importances,
                color=['#3498db' if i == np.argmax(importances)
                       else '#85c1e9' for i in range(len(importances))])
plt.title("Feature Importance", fontsize=13)
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# ── Save model ──
joblib.dump(model_pipeline, 'road_quality_model.pkl')
joblib.dump(le,             'label_encoder.pkl')
print("\n✅ Model saved as road_quality_model.pkl")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

models_to_compare = {
    "Decision Tree":      DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200, random_state=42),
    "XGBoost":            XGBClassifier(n_estimators=200, use_label_encoder=False,
                                        eval_metric='mlogloss', random_state=42),
}

print(f"{'Model':<25} {'Accuracy':>10} {'CV Mean':>10} {'CV Std':>10}")
print("-" * 58)

best_model, best_score = None, 0

for name, clf in models_to_compare.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    cv  = cross_val_score(pipe, X, y_enc, cv=5, scoring='accuracy')

    print(f"{name:<25} {acc*100:>9.2f}%  {cv.mean()*100:>8.2f}%  ±{cv.std()*100:.2f}%")

    if acc > best_score:
        best_score = acc
        best_model = (name, pipe)

print(f"\n🏆 Best model: {best_model[0]} ({best_score*100:.2f}%)")

# Save the winner automatically
joblib.dump(best_model[1], 'road_quality_model.pkl')
print(f"✅ Saved best model: {best_model[0]}")

def detect_condition(img):
    """Detect Day / Night / Fog / Wet from image properties"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_bright = np.mean(gray)
    std_bright  = np.std(gray)

    # Fog: low contrast + medium brightness
    if std_bright < 25 and 60 < mean_bright < 160:
        return "fog"
    # Night: very dark
    elif mean_bright < 50:
        return "night"
    # Wet: higher blue channel relative to red
    elif img[:,:,0].mean() > img[:,:,2].mean() * 1.1:
        return "wet"
    else:
        return "dry"

def extract_features_from_image(img_bgr):
    """
    Extract the same 6 features as the dataset from a road image.
    Returns a dict with: rl, qd, brightness, contrast, blur, visibility_score
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w    = gray.shape

    # ── 1. Brightness ──
    # Mean pixel intensity of the road marking region (upper half = sky removed)
    roi = gray[h//3 : , :]  # focus on road portion
    brightness = float(np.mean(roi))

    # ── 2. Contrast ──
    contrast = float(np.std(roi))

    # ── 3. Blur (Laplacian variance — higher = sharper) ──
    blur_score = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    # Normalize to 0-150 range (matching dataset range)
    blur_score = min(blur_score / 5.0, 150.0)

    # ── 4. RL Proxy (Retroreflectivity estimate) ──
    # High RL = bright white markings with high local contrast
    # Strategy: find bright pixels (likely markings) and measure their intensity
    _, marking_mask = cv2.threshold(roi, 140, 255, cv2.THRESH_BINARY)
    marking_pixels  = roi[marking_mask == 255]

    if len(marking_pixels) > 50:
        marking_brightness = float(np.mean(marking_pixels))
        marking_contrast   = float(np.std(marking_pixels))
        # Scale to RL range (80–350) matching your dataset
        rl_proxy = (marking_brightness / 255.0) * 250 + \
                   (marking_contrast  / 128.0) * 50
        rl_proxy = float(np.clip(rl_proxy, 80, 350))
    else:
        # No clear markings found — likely very poor condition
        rl_proxy = 60.0

    # ── 5. QD Proxy (Daytime visibility) ──
    # Based on saturation + value channel from HSV
    saturation = hsv[:,:,1][h//3:]
    value      = hsv[:,:,2][h//3:]
    qd_proxy   = float(np.mean(value)) * 0.7 + float(np.mean(saturation)) * 0.3
    # Scale to QD range (50–300)
    qd_proxy   = float(np.clip(qd_proxy * 1.5, 50, 300))

    # ── 6. Visibility Score ──
    # Composite score combining all signals
    visibility_score = (
        (brightness / 255.0)      * 30 +
        (contrast   / 100.0)      * 20 +
        (blur_score / 150.0)      * 20 +
        (rl_proxy   / 350.0)      * 20 +
        (qd_proxy   / 300.0)      * 10
    )
    visibility_score = float(np.clip(visibility_score, 0, 100))

    return {
        'rl':               round(rl_proxy,        2),
        'qd':               round(qd_proxy,         2),
        'brightness':       round(brightness,        2),
        'contrast':         round(contrast,          2),
        'blur':             round(blur_score,        2),
        'visibility_score': round(visibility_score,  2),
    }

print("✅ Feature extractor ready!")

def generate_full_report(image_path):
    """
    Main function: takes image path → returns full quality report
    """
    # ── Load Image ──
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not load image. Check the path.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Extract Features ──
    features   = extract_features_from_image(img)
    condition  = detect_condition(img)
    feat_array = np.array([[features[f] for f in FEATURE_COLS]])

    # ── Model Prediction ──
    model     = joblib.load('road_quality_model.pkl')
    encoder   = joblib.load('label_encoder.pkl')
    pred_enc  = model.predict(feat_array)[0]
    pred_prob = model.predict_proba(feat_array)[0]
    pred_label= encoder.inverse_transform([pred_enc])[0]
    confidence= float(np.max(pred_prob)) * 100

    # ── Per-parameter IRC Check ──
    param_results = {}
    failed_params = []
    for feat in FEATURE_COLS:
        val       = features[feat]
        threshold = IRC_THRESHOLDS[feat]['min']
        passed    = val >= threshold
        param_results[feat] = {
            'value':     val,
            'threshold': threshold,
            'passed':    passed,
            'unit':      IRC_THRESHOLDS[feat]['unit'],
            'name':      IRC_THRESHOLDS[feat]['name'],
            'gap':       round(val - threshold, 2)
        }
        if not passed:
            failed_params.append(feat)

    # IRC overall verdict
    irc_verdict = "✅ PASS" if len(failed_params) == 0 else "❌ FAIL"

    # ── Build Visual Report ──
    label_colors = {'Good': '#2ecc71', 'Moderate': '#f39c12', 'Poor': '#e74c3c'}
    main_color   = label_colors.get(pred_label, 'gray')

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a2e')

    # Title bar
    fig.text(0.5, 0.96,
             "🛣️ NHAI Road Marking Quality Report",
             ha='center', fontsize=18, fontweight='bold', color='white')
    fig.text(0.5, 0.92,
             f"Condition Detected: {condition.upper()}   |   "
             f"Confidence: {confidence:.1f}%   |   IRC Verdict: {irc_verdict}",
             ha='center', fontsize=12, color='#ecf0f1')

    # ── Panel 1: Input Image ──
    ax1 = fig.add_axes([0.02, 0.52, 0.30, 0.35])
    ax1.imshow(img_rgb)
    ax1.set_title("Input Image", color='white', fontsize=11, pad=6)
    ax1.axis('off')

    # ── Panel 2: Prediction Gauge ──
    ax2 = fig.add_axes([0.34, 0.52, 0.28, 0.35])
    ax2.set_facecolor('#16213e')
    classes    = encoder.classes_
    bar_colors = [label_colors.get(c, 'gray') for c in classes]
    bars = ax2.barh(classes, pred_prob * 100, color=bar_colors, height=0.5)
    for bar, prob in zip(bars, pred_prob):
        ax2.text(prob * 100 + 1, bar.get_y() + bar.get_height()/2,
                 f"{prob*100:.1f}%", va='center', color='white', fontsize=11)
    ax2.set_xlim(0, 115)
    ax2.set_xlabel("Confidence %", color='white')
    ax2.set_title(f"Prediction: {pred_label.upper()}",
                  color=main_color, fontsize=13, fontweight='bold')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444')

    # ── Panel 3: Parameter Table ──
    ax3 = fig.add_axes([0.65, 0.52, 0.33, 0.38])
    ax3.set_facecolor('#16213e')
    ax3.axis('off')
    ax3.set_title("Parameter Analysis vs IRC Thresholds",
                  color='white', fontsize=11, pad=8)

    headers = ['Parameter', 'Value', 'IRC Min', 'Status']
    col_x   = [0.0, 0.42, 0.60, 0.80]
    row_h   = 0.86 / (len(FEATURE_COLS) + 1)

    # Header row
    for hdr, x in zip(headers, col_x):
        ax3.text(x, 0.95, hdr, color='#bdc3c7',
                 fontsize=9, fontweight='bold', transform=ax3.transAxes)

    # Data rows
    for i, feat in enumerate(FEATURE_COLS):
        r      = param_results[feat]
        y_pos  = 0.95 - (i + 1) * row_h
        status = "✅ PASS" if r['passed'] else "❌ FAIL"
        color  = '#2ecc71' if r['passed'] else '#e74c3c'

        ax3.text(col_x[0], y_pos, r['name'][:20],
                 color='white',  fontsize=8, transform=ax3.transAxes)
        ax3.text(col_x[1], y_pos, str(r['value']),
                 color='white',  fontsize=8, transform=ax3.transAxes)
        ax3.text(col_x[2], y_pos, str(r['threshold']),
                 color='#bdc3c7',fontsize=8, transform=ax3.transAxes)
        ax3.text(col_x[3], y_pos, status,
                 color=color,    fontsize=8, fontweight='bold',
                 transform=ax3.transAxes)

    # ── Panel 4: Radar Chart ──
    ax4 = fig.add_axes([0.02, 0.05, 0.35, 0.40],
                       polar=True, facecolor='#16213e')
    N      = len(FEATURE_COLS)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Normalize values to 0–1 for radar
    max_vals = {'rl': 350, 'qd': 300, 'brightness': 255,
                'contrast': 100, 'blur': 150, 'visibility_score': 100}
    values  = [features[f] / max_vals[f] for f in FEATURE_COLS]
    thresh  = [IRC_THRESHOLDS[f]['min'] / max_vals[f] for f in FEATURE_COLS]
    values  += values[:1]
    thresh  += thresh[:1]

    ax4.plot(angles, thresh, 'o--', color='#e74c3c',
             linewidth=1.5, label='IRC Minimum')
    ax4.fill(angles, thresh, alpha=0.1, color='#e74c3c')
    ax4.plot(angles, values, 'o-',  color='#3498db',
             linewidth=2,   label='Measured')
    ax4.fill(angles, values, alpha=0.25, color='#3498db')

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels([f.upper() for f in FEATURE_COLS],
                        color='white', size=8)
    ax4.set_yticklabels([])
    ax4.set_facecolor('#16213e')
    ax4.tick_params(colors='white')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
               fontsize=8, labelcolor='white',
               facecolor='#1a1a2e', edgecolor='gray')
    ax4.set_title("Parameter Radar", color='white',
                  fontsize=10, pad=15)

    # ── Panel 5: Bar chart of values vs thresholds ──
    ax5 = fig.add_axes([0.40, 0.05, 0.55, 0.38],
                       facecolor='#16213e')
    x       = np.arange(len(FEATURE_COLS))
    vals    = [param_results[f]['value']     for f in FEATURE_COLS]
    threshs = [param_results[f]['threshold'] for f in FEATURE_COLS]
    bar_c   = ['#2ecc71' if param_results[f]['passed']
               else '#e74c3c' for f in FEATURE_COLS]

    ax5.bar(x - 0.2, vals,    0.35, label='Measured',    color=bar_c,   alpha=0.85)
    ax5.bar(x + 0.2, threshs, 0.35, label='IRC Minimum', color='#7f8c8d',alpha=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels([f.upper() for f in FEATURE_COLS],
                        color='white', fontsize=9)
    ax5.set_ylabel("Value", color='white')
    ax5.set_title("Measured vs IRC Minimum Thresholds",
                  color='white', fontsize=11)
    ax5.tick_params(colors='white')
    ax5.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    for spine in ax5.spines.values():
        spine.set_edgecolor('#444')

    # ── Summary Box ──
    summary_color = '#1e8449' if pred_label == 'Good' else \
                    '#d35400' if pred_label == 'Moderate' else '#922b21'
    fig.text(0.5, 0.01,
             f"  Overall Quality: {pred_label.upper()}   |   "
             f"Failed Parameters: {len(failed_params)}/6   |   "
             f"Failed: {', '.join(failed_params) if failed_params else 'None'}   |   "
             f"IRC:35 Verdict: {irc_verdict}  ",
             ha='center', fontsize=11, fontweight='bold',
             color='white',
             bbox=dict(boxstyle='round,pad=0.5',
                       facecolor=summary_color, alpha=0.8))

    plt.savefig("quality_report.png", dpi=180,
                bbox_inches='tight', facecolor='#1a1a2e')
    plt.show()

    # ── Print text summary too ──
    print("\n" + "="*55)
    print("        NHAI ROAD MARKING QUALITY REPORT")
    print("="*55)
    print(f"  Condition Detected : {condition.upper()}")
    print(f"  Predicted Quality  : {pred_label.upper()}")
    print(f"  Confidence         : {confidence:.1f}%")
    print(f"  IRC:35 Verdict     : {irc_verdict}")
    print("-"*55)
    print(f"  {'Parameter':<28} {'Value':>7}  {'Min':>7}  {'Status'}")
    print("-"*55)
    for feat in FEATURE_COLS:
        r = param_results[feat]
        status = "PASS ✅" if r['passed'] else "FAIL ❌"
        gap    = f"({r['gap']:+.1f})"
        print(f"  {r['name']:<28} {r['value']:>7}  "
              f"{r['threshold']:>7}  {status} {gap}")
    print("="*55)
    if failed_params:
        print(f"\n  ⚠️  Action needed on: {', '.join(failed_params).upper()}")
    else:
        print("\n  ✅ All parameters meet IRC standards!")
    print("="*55)

    return features, pred_label, confidence, irc_verdict

print("✅ Report generator ready!")

from google.colab import files

print("📸 Upload a road marking image (JPG/PNG)...")
uploaded_img = files.upload()

img_filename = list(uploaded_img.keys())[0]
print(f"\n🔍 Analyzing: {img_filename}")

features, label, confidence, verdict = generate_full_report(img_filename)

import json, os, cv2
import numpy as np
from google.colab import files

# ── Upload multiple drone/road images ──
print("📸 Upload multiple road images (select all at once)")
print("   Each image = one survey section (e.g. every 500m)")
uploaded_imgs = files.upload()

survey_results = []
speed_zones    = [60, 80, 80, 100, 60, 80, 100, 60, 80, 80,
                  60, 80, 100, 60, 80, 80, 60, 100, 80, 60]

model   = joblib.load('road_quality_model.pkl')
encoder = joblib.load('label_encoder.pkl')

for idx, (fname, fdata) in enumerate(uploaded_imgs.items()):
    # Save temp file
    with open(f"tmp_{idx}.jpg", "wb") as f:
        f.write(fdata)

    img      = cv2.imread(f"tmp_{idx}.jpg")
    features = extract_features_from_image(img)
    condition= detect_condition(img)

    feat_arr  = np.array([[features[f] for f in FEATURE_COLS]])
    pred_enc  = model.predict(feat_arr)[0]
    pred_prob = model.predict_proba(feat_arr)[0]
    label     = encoder.inverse_transform([pred_enc])[0]
    confidence= float(np.max(pred_prob)) * 100

    # IRC check
    failed = [f for f in FEATURE_COLS
              if features[f] < IRC_THRESHOLDS[f]['min']]
    irc_pass = len(failed) == 0

    chainage_km = round((idx + 1) * 0.5, 1)   # every 500m
    speed_zone  = speed_zones[idx % len(speed_zones)]

    survey_results.append({
        "section_id":   idx + 1,
        "chainage_km":  chainage_km,
        "filename":     fname,
        "condition":    condition,
        "label":        label,
        "confidence":   round(confidence, 1),
        "irc_pass":     irc_pass,
        "failed_params":failed,
        "speed_zone":   speed_zone,
        "features":     features,
        "thresholds":   {k: v['min'] for k, v in IRC_THRESHOLDS.items()},
    })

    status = "✅" if irc_pass else "❌"
    print(f"  {status} {chainage_km} km | {label:8s} | "
          f"conf: {confidence:.1f}% | cond: {condition}")

# Save JSON
with open("survey_data.json", "w") as f:
    json.dump(survey_results, f, indent=2)

print(f"\n✅ Survey complete — {len(survey_results)} sections analyzed")
print("   Saved: survey_data.json")
files.download("survey_data.json")