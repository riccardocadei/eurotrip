import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np

from src.utils import pace_to_str


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def visualize_schedule(schedule, skills, save=True):

    height = 0.9
    # Ensure From and To are floats
    schedule["From"] = schedule["From"].astype(float)
    schedule["To"] = schedule["To"].astype(float)

    # Get unique list of people
    people = sorted(skills["Runner"].tolist(), reverse=True)

    # Map people to rows
    person_to_y = {person: i for i, person in enumerate(people)}

    # Create a figure with two vertical sections: one for the bars, one for the elevation profile
    fig = plt.figure(figsize=(16, len(people) * 0.9 + 1))  # +1 for elevation profile
    gs = gridspec.GridSpec(2, 1, height_ratios=[len(people), 2], hspace=0.1)

    # Top axis: participation bars
    ax = fig.add_subplot(gs[0])

    # Bottom axis: elevation profile
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    # Suppress x-axis labels on the top plot
    plt.setp(ax.get_xticklabels(), visible=False)

    # Colors
    colors = {
        "run": "#a8e6a3",     # pastel green
        "run solo": "#335E30",  # pastel green
        "drive": "#cfcfcf",   # soft gray
        "sleep": "#375a7f",   # dark blue
    }

    # Add black span for car-only segment
    ax.axvspan(220, 339, color= "#729CCC", alpha=1.0)

    # Draw participation bars
    for idx, row in schedule.iterrows():
        start = row["From"]
        end = row["To"]
        for role, col in [("run", "Runner A"), ("run", "Runner B"), ("drive", "Driver")]:
            if row["Runner B"] == "":
                role = "run solo"
            if col == "Driver":
                role = "drive"
            person = row[col]
            if person == "":
                continue
            y = person_to_y[person]
            
            ax.barh(y=y, width=end - start, left=start, height=height, color=colors[role])
            # overwrite text with segment distance
            if role == "run":
                mid = (start + end) / 2
                ax.text(mid, y, f"{row['Distance']:.1f}", ha='center', va='center', fontsize=8, color='black')
            if role == "run solo":
                mid = (start + end) / 2
                ax.text(mid, y, f"{row['Distance']:.1f}", ha='center', va='center', fontsize=8, color='white')
        
        # Sleep bars
        for person in row["Sleep"]:
            y = person_to_y[person]
            ax.barh(y=y, width=end - start, left=start, height=height, color=colors["sleep"])

    # Set y-ticks and labels
    ax.set_yticks(list(person_to_y.values()))
    ax.set_yticklabels(list(person_to_y.keys()))
    ax.set_title("EuroTrip 2025 - Schedule")
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Add black span for car-only segment
    ax.axvspan(33, 42, color='black', alpha=1.0)

    # Plot D+/km as horizontal step bars
    for idx, row in schedule.iterrows():
        start = row["From"]
        end = row["To"]
        val_up = row["D+/km"]
        
        # D+/km: top
        ax2.hlines(y=val_up, xmin=start, xmax=end, colors='darkred', linewidth=2)
        ax2.vlines(x=[start, end], ymin=0, ymax=val_up, colors='darkred', linestyles='dotted', linewidth=1)
        mid = (start + end) / 2
        ax2.text(mid, val_up + 1, f"{val_up:.0f}", ha='center', va='bottom', fontsize=8, color='darkred')

    # Plot D-/km as mirrored horizontal step bars
    for idx, row in schedule.iterrows():
        start = row["From"]
        end = row["To"]
        val_down = row["D-/km"]
        
        # D-/km: bottom
        ax2.hlines(y=-val_down, xmin=start, xmax=end, colors='steelblue', linewidth=2)
        ax2.vlines(x=[start, end], ymin=-val_down, ymax=0, colors='steelblue', linestyles='dotted', linewidth=1)
        mid = (start + end) / 2
        ax2.text(mid, -val_down - 3, f"{val_down:.0f}", ha='center', va='top', fontsize=8, color='steelblue')

    # Set symmetric y-limits for elevation profile
    max_val = max(schedule["D+/km"].max(), schedule["D-/km"].max())
    # add ax2 hline for 0
    ax2.hlines(y=0, xmin=0, xmax=schedule["To"].iloc[-1], colors='black', linewidth=1, linestyle='dotted')
    ax2.set_ylim(-max_val - 15, max_val + 15)

    # Label for elevation axis
    # ax2.set_ylabel("D+/km\nD-/km", rotation=0, labelpad=30, ha='center', va='center')
    ax2.set_yticks([])  # hide ticks

    # Legend
    legend_elements = [
        Patch(facecolor=colors["run"], label="Run"),
        Patch(facecolor=colors["run solo"], label="Run (solo)"),
        Patch(facecolor=colors["drive"], label="Drive"),
        Patch(facecolor=colors["sleep"], label="Sleep"),
        Line2D([0], [0], color='darkred', lw=2, label="D+/km"),
        Line2D([0], [0], color='steelblue', lw=2, label="D-/km"),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title="Legend")

    
    # Axis labels
    checkpoints = [92, 189, 289, 392]
    for cp in checkpoints:
        ax.axvline(x=cp, color='black', linestyle='--', linewidth=1)
    km_ticks = np.arange(0, schedule["To"].max() + 1, 20)
    ax2.set_xticks(km_ticks)
    ax2.set_xticklabels([f"{km:.0f} km" for km in km_ticks])
    ax2.tick_params(axis='x', labelrotation=0, labelsize=9)
    checkpoints_labels = [f"CP{i+1}" for i in range(len(checkpoints))]
    secax = ax.secondary_xaxis('bottom')
    secax.set_xticks(checkpoints)
    secax.set_xticklabels(checkpoints_labels, rotation=0, fontsize=9)
    secax.tick_params(axis='x', direction='out', length=5)
    secax.set_xlabel("Checkpoints")

    # Set x-limits
    ax.set_xlim(schedule["From"].iloc[0], schedule["To"].iloc[-1])

    # Save and show
    plt.show()
    if save: fig.savefig('results/schedule.png', bbox_inches='tight', dpi=300, facecolor='white')


def plot_pace_decay(pace_HM, k_values=[0.002, 0.004, 0.006], save=True):
    dist = np.linspace(21, 70, 500)
    plt.figure(figsize=(8,5))

    for k in k_values:
        pace = pace_HM * np.exp(k * (dist - 21))
        plt.plot(dist, pace, label=f'k={k:.3f}')

    yticks = np.arange(pace_HM, pace_HM*np.exp(k_values[-1]*(70-21)) + 0.5, 10)
    yticklabels = [pace_to_str(p) for p in yticks]

    plt.yticks(yticks, yticklabels)
    plt.xlabel('Distance (km)')
    plt.ylabel('Pace (min:sec per km)')
    plt.title(r'Explonential slowdown endurance model: $p(d) = p_0 \cdot e^{k(d - 21)}$')
    plt.legend(title='Endurance level', loc='upper left')

    plt.grid(True)
    plt.tight_layout()
    if save: plt.savefig(f"results/pace/{int(pace_HM)//60}:{int(pace_HM)%60:02d}.png")
    plt.show()