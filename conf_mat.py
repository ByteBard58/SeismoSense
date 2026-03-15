from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

file_path = "dataset/earthquake_data.csv"
model = joblib.load("models/estimator.pkl")

# Earthquake alert labels matching the app.py label_map
# 0: green (Low), 1: orange (Moderate), 2: red (High), 3: yellow (Watch)
ALERT_LABELS = ["Green\n(Low)", "Orange\n(Moderate)", "Red\n(High)", "Yellow\n(Watch)"]
ALERT_COLORS = {
    'Green': '#22c55e',   # green - safe/low
    'Orange': '#fb923c',  # orange - moderate
    'Red': '#ef4444',     # red - high
    'Yellow': '#facc15'  # yellow - watch
}

def load_data(path="dataset/earthquake_data.csv") -> np.ndarray:
  df = pd.read_csv(path)

  x = df.iloc[:,:-1].to_numpy()
  y_unenc = df.iloc[:,-1]

  labelenc = LabelEncoder()
  y = labelenc.fit_transform(y_unenc)

  return x,y

def plotting(x, y) -> None:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=2/10, random_state=120, shuffle=True, stratify=y
    )

    labels = ALERT_LABELS
    y_true = y_test
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_true, y_pred)

    # ── Theme Configuration ─────────────────────────────────────────────────────
    # Professional dark theme matching the frontend
    plt.style.use('dark_background')
    
    BG_COLOR = '#151C26'          # Dark background
    CARD_BG = "#050506"           # Card background
    TEXT_PRIMARY = '#e2e8f0'      # Primary text
    TEXT_SECONDARY = '#94a3b8'     # Muted text
    ACCENT = '#00d4aa'            # Teal accent
    
    # Alert-specific colors for the heatmap
    ALERT_HEATMAP_COLORS = [
        '#0f1419',    # dark (zero counts)
        '#1e3a4c',    # dark teal
        '#00d4aa',    # accent teal
        '#00ffcc',    # bright teal
        '#ffffff',    # white (high counts)
    ]
    heatmap_cmap = mcolors.LinearSegmentedColormap.from_list('seismo_heat', ALERT_HEATMAP_COLORS, N=256)
    
    # ── Figure Setup ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_BG)
    
    # Custom annotation colors based on cell value
    def get_annot_color(val, max_val):
        """Return white for dark cells, dark for bright cells"""
        ratio = val / max_val if max_val > 0 else 0
        return '#0f1419' if ratio > 0.4 else '#ffffff'
    
    # Create annotation matrix with custom colors
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = str(cm[i, j])
    
    max_val = cm.max()
    
    # ── Plot Heatmap ─────────────────────────────────────────────────────────────
    sns.heatmap(
        cm,
        annot=annot,
        fmt='',
        cmap=heatmap_cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={
            'fontsize': 20,
            'fontweight': 'bold'
        },
        cbar_kws={
            'label': 'Count',
            'shrink': 0.8
        },
        linewidths=2,
        linecolor='#2d3748',
        square=True,
        vmin=0,
        vmax=max_val
    )
    
    # Update annotation colors dynamically
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.texts[i * cm.shape[1] + j].set_color(
                get_annot_color(cm[i, j], max_val)
            )
    
    # ── Axis Labels ──────────────────────────────────────────────────────────────
    ax.set_xlabel('Predicted Alert Level', fontsize=14, fontweight='bold',
                  color=TEXT_PRIMARY, labelpad=15)
    ax.set_ylabel('True Alert Level', fontsize=14, fontweight='bold',
                  color=TEXT_PRIMARY, labelpad=15)
    
    # ── Tick Labels ──────────────────────────────────────────────────────────────
    ax.tick_params(axis='x', colors=TEXT_SECONDARY, labelsize=11, rotation=0)
    ax.tick_params(axis='y', colors=TEXT_SECONDARY, labelsize=11, rotation=0)
    
    # Color tick labels by alert severity
    tick_colors = ['#22c55e', '#fb923c', '#ef4444', '#facc15']
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(tick_colors[i] if i < len(tick_colors) else TEXT_SECONDARY)
    
    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color(tick_colors[i] if i < len(tick_colors) else TEXT_SECONDARY)
    
    # ── Spines ──────────────────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_color('#2d3748')
        spine.set_linewidth(2)
    
    # ── Colorbar ─────────────────────────────────────────────────────────────────
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
    cbar.set_label('Count', color=TEXT_SECONDARY, fontsize=12, labelpad=10)
    cbar.outline.set_color('#2d3748')
    cbar.outline.set_linewidth(2)
    
    # ── Title ────────────────────────────────────────────────────────────────────
    ax.set_title('Model Performance: Confusion Matrix', fontsize=16, fontweight='bold',
                 color=TEXT_PRIMARY, pad=20)
    
    # ── Save ─────────────────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(
        "static/confusion_matrix.png",
        dpi=150,
        facecolor=BG_COLOR,
        edgecolor='none',
        bbox_inches='tight'
    )
    plt.close()
    print("Saved confusion matrix with earthquake alert labels ✅")


def main() -> None:
    x, y = load_data()
    plotting(x, y)


if __name__ == "__main__":
    main()
