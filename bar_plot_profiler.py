#%%
import matplotlib.pyplot as plt
import numpy as np

import scienceplots

plt.style.use(['science','no-latex'])


# Data: size, init%, setup%, solve%, switchtopo% (switchtopo is part of solve)
data = {
    "2 GPU": [
        [8,  0.014179,	93.554536,	5.90074	    ,4.51229],
        [16, 0.025006,	79.358189,	20.479622	,15.797902],
        [32, 0.025764,	77.866918,	21.802799	,15.688976],
        [64, 0.021606,	72.09799,	27.980939	,17.024159],
        [128,0.010858,	38.166576,	61.461834	,31.922108],
        [256,0.002217,	8.556086,	91.175461	,39.794624],
        [512,0.00032	, 1.652203	,   97.826271	,41.873631]
    ],
    "4 GPU": [
        [8,    0.057484,	64.180871,	35.545542,	29.048014],
        [16,   0.041073,	62.927894,	36.894786,	29.918107],
        [32,   0.039059,	61.84745,	38.537274,	30.069744],
        [64,   0.034902,	56.515193,	43.048408,	30.392789],
        [128,   0.017621,	31.25599,	67.384304,	41.908166],
        [256  , 0.00375	,    7.096572,	91.221469,	50.027041],
        [512 ,  0.0055	,    1.538735,	96.720916,	51.270145]
    ],
    "8 GPU": [
        [8,  0.01491	,   84.784206	,14.087312	,12.36503],
        [16, 0.024095,	66.436797	,33.17942	,29.149654],
        [32, 0.020418,	68.076412	,28.854747	,24.694362],
        [64, 0.02109	,   64.645258	,35.155732	,29.208113],
        [128,0.014258,	49.435263	,52.005626	,40.093392],
        [256,0.003765,	13.220434	,85.661861	,65.025558],
        [512,0.000559,	2.464474	,96.467327	,72.311655]
    ]
}




softred   = (0.8, 0.2, 0.2)  # muted red
softblue  = (0.2, 0.4, 0.8)  # muted blue
softyellow= (0.9, 0.8, 0.3)  # muted yellow
softgreen = (0.3, 0.7, 0.3)  # muted green

#%%

# Sizes for x-axis
sizes = [8, 16, 32, 64, 128, 256, 512]
latex_sizes = [r"$8^3$", r"$16^3$", r"$32^3$", r"$64^3$", r"$128^3$", r"$256^3$", r"$512^3$"]
x = np.arange(len(sizes))  # positions for bars
width = 0.25  # bar width

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, (title, values) in zip(axes, data.items()):
    values = np.array(values)
    init = values[:, 1]
    setup = values[:, 2]
    solve = values[:, 3]
    switchtopo = values[:, 4]  # SwitchTopo% (subset of solve)
    
    # Stacked bar chart: init, setup, solve
    p1 = ax.bar(x, init, width, label='Init', color=softgreen)
    p2 = ax.bar(x, setup, width, bottom=init, label='Setup', color=softblue)
    p3 = ax.bar(x, solve, width, bottom=init+setup, label='Solve', color=softred)
    
    # Highlight switchtopo as part of solve with hatching
    ax.bar(x, switchtopo, width, bottom=init+setup+solve-switchtopo, hatch='//', 
            color='none', edgecolor='black', label='Switch Topology')
    
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(latex_sizes)
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 110)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig("output/bar_plot.pdf", bbox_inches='tight')
plt.show()
# %%

for i, (title, values) in enumerate(data.items()):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    values = np.array(values)
    init = values[:, 1]
    setup = values[:, 2]
    solve = values[:, 3]
    switchtopo = values[:, 4]

    ax.bar(x, init, width, label='Init', color=softgreen)
    ax.bar(x, setup, width, bottom=init, label='Setup', color=softblue)
    ax.bar(x, solve, width, bottom=init+setup, label='Solve', color=softred)
    
    ax.bar(x, switchtopo, width, bottom=init+setup+solve-switchtopo,
            hatch='//', color='none', edgecolor='black', label='Switch Topology')

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(latex_sizes)
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 110)

    # Only add legend for last plot
    if i == len(data) - 1:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

    plt.tight_layout()
    plt.savefig(f"output/bar_plot_{i+1}.pdf", bbox_inches='tight')
    plt.show()
# %%
