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

    labels = ["GALAXY", "STAR", "QSO"]
    y_true = y_test
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_true, y_pred)

    # Use dark background style
    plt.style.use('dark_background')

    # Theme colors matching the frontend CSS
    BG_COLOR       = '#12121c'   # --bg-tertiary
    TEXT_PRIMARY   = '#ffffff'   # --text-primary
    TEXT_SECONDARY = '#8b8b9e'   # --text-secondary

    # Class brand colors — used ONLY for tick labels
    CLASS_COLORS = {
        'GALAXY': '#7000ff',  # purple
        'STAR':   '#00d4ff',  # cyan
        'QSO':    '#ff6b6b',  # coral
    }

    # ── Colormap ────────────────────────────────────────────────────────────
    # Single-hue dark → bright purple ramp.
    # Keeps the grid visually consistent; class identity is communicated
    # through the tick label colors above, not through the heatmap fill.
    cmap_colors = [
        '#05050a',  # near-black (zero / empty cells)
        '#1a0535',  # very dark purple
        '#3b0d7a',  # deep purple
        '#5a12b8',  # mid purple
        '#7000ff',  # brand purple  (GALAXY_COLOR)
        '#9d4edd',  # bright lavender highlight
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list('cosmo_purple', cmap_colors, N=256)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={
            'fontsize': 18,
            'fontweight': 'bold',
            'color': TEXT_PRIMARY
        },
        cbar_kws={
            'label': 'Count',
            'shrink': 0.8
        },
        linewidths=3,
        linecolor=BG_COLOR,
        square=True,
        vmin=0,
        vmax=cm.max()
    )

    # ── Axis labels ──────────────────────────────────────────────────────────
    ax.set_xlabel('Predicted Classification', fontsize=14, fontweight='bold',
                  color=TEXT_SECONDARY, labelpad=15)
    ax.set_ylabel('True Classification', fontsize=14, fontweight='bold',
                  color=TEXT_SECONDARY, labelpad=15)

    # ── Tick labels — colored by class brand color ───────────────────────────
    ax.tick_params(axis='x', colors=TEXT_SECONDARY, labelsize=12, rotation=0)
    ax.tick_params(axis='y', colors=TEXT_SECONDARY, labelsize=12, rotation=0)

    for label in ax.get_xticklabels():
        label.set_color(CLASS_COLORS.get(label.get_text(), TEXT_SECONDARY))

    for label in ax.get_yticklabels():
        label.set_color(CLASS_COLORS.get(label.get_text(), TEXT_SECONDARY))

    # ── Spines ───────────────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_color(BG_COLOR)
        spine.set_linewidth(3)

    # ── Colorbar ─────────────────────────────────────────────────────────────
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=11)
    cbar.set_label('Count', color=TEXT_SECONDARY, fontsize=12, labelpad=10)
    cbar.outline.set_edgecolor(BG_COLOR)
    cbar.outline.set_linewidth(2)

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(
        "static/confusion_matrix.png",
        dpi=150,
        facecolor=BG_COLOR,
        edgecolor='none',
        bbox_inches='tight'
    )
    plt.close()
    print("Saved and closed confusion matrix ✅")


def main() -> None:
    x, y = load_data()
    plotting(x, y)


if __name__ == "__main__":
    main()