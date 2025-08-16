#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import scienceplots

plt.style.use(['science','no-latex'])

#%%

size = [8,16,32,64,128,256,512,1024]
real_size_2 = [8**3, 16**3, 32**3, 64**3, 128**3, 256**3, 512**3]

real_size_4 = [8**3, 16**3, 32**3, 64**3, 128**3, 256**3, 512**3, 512*512*1024]

real_size_8 = [8**3, 16**3, 32**3, 64**3, 128**3, 256**3, 512**3, 512*1024*1024]

label_size = [r'$8^3$', r'$16^3$', r'$32^3$', r'$64^3$', r'$128^3$', r'$256^3$', r'$512^3$', r'$512^2 \times 1024$']

mem_size_4 = np.array([0.000317, 0.002192, 0.016155, 0.123703, 0.967468, 7.651123, 60.854492, 121.708984])
time_4 = np.array([0.0219, 0.0223, 0.0229, 0.0260, 0.0699, 0.409, 2.9483, 6.3214])

time_4_4nodes = np.array([0.0220, 0.0223, 0.0228, 0.0260, 0.07, 0.4129, 2.9571, 6.1113])

mem_size_8 = np.array([0.000302, 0.002016, 0.014509, 0.109612, 0.851044, 6.704956, 53.226074, 212.904297])
time_8_2nodes = np.array([0.0379, 0.0362, 0.036, 0.0409, 0.0832, 0.5128, 3.9633, 14.7541])

time_8_4nodes = np.array([0.0357, 0.0346, 0.0347, 0.0369, 0.0610, 0.325, 2.206, 8.9316])

mem_size_2 = np.array([0.000288, 0.002077, 0.015694, 0.121857, 0.960083, 7.621582, 60.736328])
time_2 = np.array([0.0177, 0.018, 0.0183, 0.0234, 0.0861, 0.5063, 3.8680])

bw_2 = mem_size_2 / time_2
bw_4 = mem_size_4 / time_4
bw_8_2nodes = mem_size_8 / time_8_2nodes
bw_8_4nodes = mem_size_8 / time_8_4nodes





#%%

plt.figure(figsize=(8,6))
plt.plot(real_size_2, bw_2, label='2 GPU', marker='o')
plt.plot(real_size_4, bw_4, label='4 GPU', marker='s')
plt.plot(real_size_8, bw_8_2nodes, label='8 GPU (2 nodes)', marker='*')
plt.plot(real_size_8, bw_8_4nodes, label='8 GPU (4 nodes)', marker='x')
plt.xscale('log')
plt.xticks(real_size_8, label_size)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.axvline(x=100**3, color='black')
plt.xlabel('Grid size')
plt.ylabel('Bandwidth (GB/s)')
plt.title('Bandwidth of different GPU configurations')
plt.legend()
plt.tight_layout()
#plt.savefig('gpu_bandwidth.pdf', dpi=300)
plt.show()
# %%

real_size = [8**3, 16**3, 32**3, 64**3, 128**3, 256**3, 512**3]
label_size = [r'$8^3$', r'$16^3$', r'$32^3$', r'$64^3$', r'$128^3$', r'$256^3$', r'$512^3$']

x_2 = np.array([8, 9, 11, 13, 16, 18, 23, 27, 32, 36, 45, 54, 64, 72, 91, 108, 128, 144, 181, 216, 256, 289, 362, 452, 512])
size_2 = x_2**3
label_size_2 = [r'$8^3$', r'$9^3$', r'$11^3$', r'$13^3$', r'$16^3$', r'$18^3$', r'$23^3$', r'$27^3$', r'$32^3$', r'$36^3$', r'$45^3$', r'$54^3$', r'$64^3$', r'$72^3$', r'$91^3$', r'$108^3$', r'$128^3$', r'$144^3$', r'$181^3$', r'$216^3$', r'$256^3$', r'$289^3$', r'$362^3$', r'$452^3$', r'$512^3$']
bytes_2 = np.array([0.000288, 0.000361, 0.000649, 0.00106, 0.002077, 0.002921, 0.005712, 0.009188, 0.015694, 0.022199, 0.041968, 0.073607, 0.121857, 0.172919, 
                    0.343371, 0.578343, 0.960083, 1.364656, 2.687576, 4.584685, 7.621582, 10.917771, 21.501265, 41.809812, 60.736328])
optimized_2 = np.array([0.0127, 0.0127, 0.0131, 0.0134, 0.0132, 0.0134, 0.0143, 0.0137, 0.014, 0.0139, 0.0158, 0.0178, 0.0209, 0.0263, 
                        0.0414, 0.0598, 0.0884, 0.1149, 0.2315, 0.3312, 0.5397, 0.7872, 1.7005, 3.6435, 4.2299])

non_opti_2 = np.array([0.0173, 0.0176, 0.0175, 0.0176, 0.0178, 0.0178, 0.0179, 0.0182, 0.0181, 0.0186, 0.0199,
                        0.0213, 0.0234, 0.0273, 0.0418, 0.0579, 0.0854, 0.1154, 0.1879, 0.305, 0.5145, 0.7350, 1.3836, 2.6906, 3.8684])

bytes_4 = np.array([0.000317, 0.000415, 0.000682, 0.001174, 0.002192, 0.002921, 0.005842, 0.009365, 0.016155, 0.022783, 0.043336,
                    0.073607, 0.123703, 0.175256, 0.345279, 0.583601, 0.967468, 1.355309, 2.709726, 4.605716, 7.651123, 10.974242, 21.501265,
                    41.901904, 60.854492])

non_opti_4 = np.array([0.0217, 0.0218, 0.0220, 0.0221, 0.0222, 0.0223, 0.0224, 0.0225, 0.0226, 0.0229, 0.0233,
                        0.0241, 0.0257, 0.0273, 0.0352, 0.0494, 0.07000, 0.0930, 0.1666, 0.2546, 0.4001, 0.5677, 1.057, 
                        2.0234, 2.9359])

optimized_4 = np.array([0.0151, 0.0154, 0.0159, 0.0159, 0.0157, 0.0159, 0.0169, 0.0162, 0.0165, 0.0162, 0.0174, 0.0183, 0.0199, 0.0229, 
                        0.0332, 0.0472, 0.0680, 0.0920, 0.1908, 0.2647, 0.4203, 0.5869, 1.2179, 2.5153, 3.0818])

bytes_8 = np.array([0.000302, 0.000432, 0.000728, 0.001027, 0.002016, 0.002730, 0.005309, 0.008953, 0.014509, 0.019527, 0.037795, 0.063944, 0.109612,
                    0.155046, 0.310084, 0.506192, 0.851044, 1.207651, 2.357265, 4.039017, 6.704956, 9.652769, 18.843183, 36.520605, 53.871582])

optimized_8 = np.array([0.0277, 0.0265, 0.0310, 0.0234, 0.0248, 0.0239, 0.0291, 0.0274, 0.0260, 0.0245, 0.0266, 0.0272, 0.0272, 
                        0.0297, 0.0403, 0.0571, 0.0783, 0.1093, 0.2087, 0.3197, 0.5138, 0.7459, 1.5023, 2.9878, 4.0058])


non_opti_8 = np.array([0.0364, 0.0362, 0.0426, 0.0466, 0.0366, 0.0376, 0.0372, 0.0382, 0.0357, 0.0362, 0.0379, 0.0387, 0.0413,
                        0.0407, 0.0487, 0.0613, 0.0827, 0.1117, 0.2002, 0.3170, 0.5134, 0.7363, 1.4075, 2.7096, 3.9652])

non_opti_8_4 = np.array([0.0362, 0.0343, 0.0352, 0.0332, 0.0332, 0.0334, 0.0361, 0.0362, 0.0339, 0.0341, 0.0359, 0.0354, 0.0358, 0.0365, 
                            0.0425, 0.0466, 0.0619, 0.0835, 0.1438, 0.2096, 0.3233, 0.4594, 0.8241, 1.6326, 2.2561])


optimized_8_4 = np.array([0.0246, 0.0241, 0.0243, 0.0229, 0.0231, 0.0233, 0.0260, 0.0254, 0.0240, 0.0239, 0.0246, 0.0251, 0.0253, 0.0267, 0.0338,
                            0.0414, 0.0551, 0.0777, 0.1508, 0.2104, 0.3262, 0.4659, 0.8929, 1.7588, 2.3188])

bw_opti_2 = bytes_2 / optimized_2
bw_non_opti_2 = bytes_2 / non_opti_2

improvement_2 = (bw_opti_2 - bw_non_opti_2 ) / bw_non_opti_2
print("Improvement for 2 GPUs: \n")
print(improvement_2)

bw_opti_4 = bytes_4 / optimized_4
bw_non_opti_4 = bytes_4 / non_opti_4

improvement_4 = (bw_opti_4 - bw_non_opti_4 ) / bw_non_opti_4
print("Improvement for 4 GPUs: \n")
print(improvement_4)

bw_opti_8 = bytes_8 / optimized_8
bw_non_opti_8 = bytes_8 / non_opti_8
improvement_8 = (bw_opti_8 - bw_non_opti_8) / bw_non_opti_8
print("Improvement for 8 GPUs: \n")
print(improvement_8)

bw_opti_8_4 = bytes_8 / optimized_8_4
bw_non_opti_8_4 = bytes_8 / non_opti_8_4

improvement_8_4 = (bw_opti_8_4 - bw_non_opti_8_4) / bw_non_opti_8_4
print("Improvement for 8 GPUs on 4 nodes: \n")
print(improvement_8_4)

#%%

plt.figure(figsize=(8,6))
plt.plot(size_2, bw_opti_2, label='Optimized', marker='o')
plt.plot(size_2, bw_non_opti_2, label='Non-Optimized', marker='x')
plt.xscale('log')
plt.xlabel('Grid Size (Bytes)')
plt.xticks(size_2, x_2)
plt.ylabel('Bandwidth (GB/s)')
plt.title('Bandwidth Comparison for 2 GPUs')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('output/improved_2gpu.pdf', dpi=300)
plt.show()

#%%
####################################################################################
# Plot for 2 GPUs with zoomed inset
####################################################################################

fig = plt.figure(figsize=(8,6))
ax = plt.axes()

# Main plot
ax.plot(size_2, bw_opti_2, label='Optimized', marker='o')
ax.plot(size_2, bw_non_opti_2, label='Non-Optimized', marker='x')
ax.set_xscale('log')
ax.set_xlabel('Grid Size (Bytes)')
ax.set_ylabel('Bandwidth (GB/s)')
ax.set_title('Bandwidth Comparison for 2 GPUs')
ax.legend(loc='lower right')
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
#plt.grid(True, which='minor', linestyle=':', linewidth=0.2)
plt.xticks(real_size, label_size)

# ---- Add Subplot Axes manually ----
box = ax.get_position()
width = box.width
height = box.height
rect = [0.05, 0.5, 0.45, 0.45]  # [x_frac, y_frac, w_frac, h_frac] inside main axes

# Compute inset position
inax_position  = ax.transAxes.transform(rect[0:2])
transFigure = fig.transFigure.inverted()
infig_position = transFigure.transform(inax_position)
x, y = infig_position
w = width * rect[2]
h = height * rect[3]

# Create inset axes
subax = fig.add_axes([x, y, w, h], facecolor='w')  # or axisbg='w' in older matplotlib
subax.plot(size_2[0:6], bw_opti_2[0:6], label='Optimized', marker='o')
subax.plot(size_2[0:6], bw_non_opti_2[0:6], label='Non-Optimized', marker='x')
subax.set_xscale('log')
subax.set_title('Zoom', fontsize=9)
subax.set_xticks(size_2[0:6])
subax.set_xticklabels(label_size_2[0:6], fontsize=6)
subax.set_yticks(subax.get_yticks())
subax.tick_params(axis='y', labelsize=6)
subax.tick_params(axis='x', labelsize=6)
subax.set_ylim(0, 0.25)
subax.set_xlim(size_2[0]-100, size_2[5] + 100)
subax.grid(True, which='major', linestyle='--', linewidth=0.5)
#subax.grid(True, which='minor', linestyle=':', linewidth=0.5)


# ---- Draw rectangle on main plot to show zoomed-in region ----
rect_x0 = size_2[0] - 100
rect_x1 = size_2[5] + 100
rect_y0 = -1
rect_y1 = 2

zoom_rect = patches.Rectangle(
    (rect_x0, rect_y0),                # (x, y) bottom left
    rect_x1 - rect_x0,                # width
    rect_y1 - rect_y0,                # height
    linewidth=1,
    edgecolor='black',
    linestyle='-',
    facecolor='none',
    zorder=10
)
ax.add_patch(zoom_rect)


# === Manually defined corners in data coordinates ===

# Corners of zoom box in main plot (ax)
main_box_corners = [
    (rect_x0, rect_y1),
    (rect_x1, rect_y1)
]

# Corners of inset box in data coordinates (subax)
inset_box_corners = [
    (size_2[0]-100, 0),      # bottom-left of inset axes
    (size_2[5]+100, 0)       # bottom-right of inset axes
]

# === Transform to figure coordinates ===

fig_coords = []

for main_point, inset_point in zip(main_box_corners, inset_box_corners):
    # Transform main plot corners to display coordinates
    main_disp = ax.transData.transform(main_point)
    # Transform inset axes corners to display coordinates
    inset_disp = subax.transData.transform(inset_point)

    # Convert display to figure coords
    main_fig = fig.transFigure.inverted().transform(main_disp)
    inset_fig = fig.transFigure.inverted().transform(inset_disp)

    # Draw the line
    line = Line2D(
        [main_fig[0], inset_fig[0]],
        [main_fig[1], inset_fig[1]],
        transform=fig.transFigure,
        color='black',
        linestyle='--',
        linewidth=0.6,
    )
    fig.lines.append(line)
plt.savefig('output/zoomed_inset_plot_2.pdf', dpi=300)

plt.show()

#%%
#####################################################################################
# Plot for 4 GPUs with zoomed inset
#####################################################################################
fig = plt.figure(figsize=(8,6))
ax = plt.axes()

# Main plot
ax.plot(size_2, bw_opti_4, label='Optimized', marker='o')
ax.plot(size_2, bw_non_opti_4, label='Non-Optimized', marker='x')
ax.set_xscale('log')
ax.set_xlabel('Grid Size (Bytes)')
ax.set_ylabel('Bandwidth (GB/s)')
ax.set_title('Bandwidth Comparison for 4 GPUs')
ax.legend(loc='lower right')
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
#plt.grid(True, which='minor', linestyle=':', linewidth=0.2)
plt.xticks(real_size, label_size)

# ---- Add Subplot Axes manually ----
box = ax.get_position()
width = box.width
height = box.height
rect = [0.05, 0.5, 0.45, 0.45]  # [x_frac, y_frac, w_frac, h_frac] inside main axes

# Compute inset position
inax_position  = ax.transAxes.transform(rect[0:2])
transFigure = fig.transFigure.inverted()
infig_position = transFigure.transform(inax_position)
x, y = infig_position
w = width * rect[2]
h = height * rect[3]

# Create inset axes
subax = fig.add_axes([x, y, w, h], facecolor='w')  # or axisbg='w' in older matplotlib
subax.plot(size_2[0:6], bw_opti_4[0:6], label='Optimized', marker='o')
subax.plot(size_2[0:6], bw_non_opti_4[0:6], label='Non-Optimized', marker='x')
subax.set_xscale('log')
subax.set_title('Zoom', fontsize=9)
subax.set_xticks(size_2[0:6])
subax.set_xticklabels(label_size_2[0:6], fontsize=6)
subax.set_yticks(subax.get_yticks())
subax.tick_params(axis='y', labelsize=6)
subax.tick_params(axis='x', labelsize=6)
subax.set_ylim(0, 0.25)
subax.set_xlim(size_2[0]-100, size_2[5] + 100)
subax.grid(True, which='major', linestyle='--', linewidth=0.5)
#subax.grid(True, which='minor', linestyle=':', linewidth=0.5)


# ---- Draw rectangle on main plot to show zoomed-in region ----
rect_x0 = size_2[0] - 100
rect_x1 = size_2[5] + 100
rect_y0 = -1
rect_y1 = 2

zoom_rect = patches.Rectangle(
    (rect_x0, rect_y0),                # (x, y) bottom left
    rect_x1 - rect_x0,                # width
    rect_y1 - rect_y0,                # height
    linewidth=1,
    edgecolor='black',
    linestyle='-',
    facecolor='none',
    zorder=10
)
ax.add_patch(zoom_rect)


# === Manually defined corners in data coordinates ===

# Corners of zoom box in main plot (ax)
main_box_corners = [
    (rect_x0, rect_y1),
    (rect_x1, rect_y1)
]

# Corners of inset box in data coordinates (subax)
inset_box_corners = [
    (size_2[0]-100, 0),      # bottom-left of inset axes
    (size_2[5]+100, 0)       # bottom-right of inset axes
]

# === Transform to figure coordinates ===

fig_coords = []

for main_point, inset_point in zip(main_box_corners, inset_box_corners):
    # Transform main plot corners to display coordinates
    main_disp = ax.transData.transform(main_point)
    # Transform inset axes corners to display coordinates
    inset_disp = subax.transData.transform(inset_point)

    # Convert display to figure coords
    main_fig = fig.transFigure.inverted().transform(main_disp)
    inset_fig = fig.transFigure.inverted().transform(inset_disp)

    # Draw the line
    line = Line2D(
        [main_fig[0], inset_fig[0]],
        [main_fig[1], inset_fig[1]],
        transform=fig.transFigure,
        color='black',
        linestyle='--',
        linewidth=0.6,
    )
    fig.lines.append(line)
plt.savefig('output/zoomed_inset_plot_4.pdf', dpi=300)

plt.show()

#%%
#####################################################################################
# Plot for 8 GPUs with zoomed inset
#####################################################################################
fig = plt.figure(figsize=(8,6))
ax = plt.axes()

# Main plot
ax.plot(size_2, bw_opti_8, label='Optimized', marker='o')
ax.plot(size_2, bw_non_opti_8, label='Non-Optimized', marker='x')
ax.set_xscale('log')
ax.set_xlabel('Grid Size (Bytes)')
ax.set_ylabel('Bandwidth (GB/s)')
ax.set_title('Bandwidth Comparison for 8 GPUs')
ax.legend(loc='lower right')
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
#plt.grid(True, which='minor', linestyle=':', linewidth=0.2)
plt.xticks(real_size, label_size)

# ---- Add Subplot Axes manually ----
box = ax.get_position()
width = box.width
height = box.height
rect = [0.05, 0.5, 0.45, 0.45]  # [x_frac, y_frac, w_frac, h_frac] inside main axes

# Compute inset position
inax_position  = ax.transAxes.transform(rect[0:2])
transFigure = fig.transFigure.inverted()
infig_position = transFigure.transform(inax_position)
x, y = infig_position
w = width * rect[2]
h = height * rect[3]

# Create inset axes
subax = fig.add_axes([x, y, w, h], facecolor='w')  # or axisbg='w' in older matplotlib
subax.plot(size_2[0:6], bw_opti_8[0:6], label='Optimized', marker='o')
subax.plot(size_2[0:6], bw_non_opti_8[0:6], label='Non-Optimized', marker='x')
subax.set_xscale('log')
subax.set_title('Zoom', fontsize=9)
subax.set_xticks(size_2[0:6])
subax.set_xticklabels(label_size_2[0:6], fontsize=6)
subax.set_yticks(subax.get_yticks())
subax.tick_params(axis='y', labelsize=6)
subax.tick_params(axis='x', labelsize=6)
subax.set_ylim(0, 0.15)
subax.set_xlim(size_2[0]-100, size_2[5] + 100)
subax.grid(True, which='major', linestyle='--', linewidth=0.5)
#subax.grid(True, which='minor', linestyle=':', linewidth=0.5)


# ---- Draw rectangle on main plot to show zoomed-in region ----
rect_x0 = size_2[0] - 100
rect_x1 = size_2[5] + 100
rect_y0 = -1
rect_y1 = 2

zoom_rect = patches.Rectangle(
    (rect_x0, rect_y0),                # (x, y) bottom left
    rect_x1 - rect_x0,                # width
    rect_y1 - rect_y0,                # height
    linewidth=1,
    edgecolor='black',
    linestyle='-',
    facecolor='none',
    zorder=10
)
ax.add_patch(zoom_rect)


# === Manually defined corners in data coordinates ===

# Corners of zoom box in main plot (ax)
main_box_corners = [
    (rect_x0, rect_y1),
    (rect_x1, rect_y1)
]

# Corners of inset box in data coordinates (subax)
inset_box_corners = [
    (size_2[0]-100, 0),      # bottom-left of inset axes
    (size_2[5]+100, 0)       # bottom-right of inset axes
]

# === Transform to figure coordinates ===

fig_coords = []

for main_point, inset_point in zip(main_box_corners, inset_box_corners):
    # Transform main plot corners to display coordinates
    main_disp = ax.transData.transform(main_point)
    # Transform inset axes corners to display coordinates
    inset_disp = subax.transData.transform(inset_point)

    # Convert display to figure coords
    main_fig = fig.transFigure.inverted().transform(main_disp)
    inset_fig = fig.transFigure.inverted().transform(inset_disp)

    # Draw the line
    line = Line2D(
        [main_fig[0], inset_fig[0]],
        [main_fig[1], inset_fig[1]],
        transform=fig.transFigure,
        color='black',
        linestyle='--',
        linewidth=0.6,
    )
    fig.lines.append(line)
plt.savefig('output/zoomed_inset_plot_8.pdf', dpi=300)

plt.show()

#%%
#####################################################################################
# Plot for 8_4 GPUs with zoomed inset
#####################################################################################
fig = plt.figure(figsize=(8,6))
ax = plt.axes()

# Main plot
ax.plot(size_2, bw_opti_8_4, label='Optimized', marker='o')
ax.plot(size_2, bw_non_opti_8_4, label='Non-Optimized', marker='x')
ax.set_xscale('log')
ax.set_xlabel('Grid Size (Bytes)')
ax.set_ylabel('Bandwidth (GB/s)')
ax.set_title('Bandwidth Comparison for 8 GPUs on 4 nodes')
ax.legend(loc='lower right')
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
#plt.grid(True, which='minor', linestyle=':', linewidth=0.2)
plt.xticks(real_size, label_size)

# ---- Add Subplot Axes manually ----
box = ax.get_position()
width = box.width
height = box.height
rect = [0.05, 0.5, 0.45, 0.45]  # [x_frac, y_frac, w_frac, h_frac] inside main axes

# Compute inset position
inax_position  = ax.transAxes.transform(rect[0:2])
transFigure = fig.transFigure.inverted()
infig_position = transFigure.transform(inax_position)
x, y = infig_position
w = width * rect[2]
h = height * rect[3]

# Create inset axes
subax = fig.add_axes([x, y, w, h], facecolor='w')  # or axisbg='w' in older matplotlib
subax.plot(size_2[0:6], bw_opti_8_4[0:6], label='Optimized', marker='o')
subax.plot(size_2[0:6], bw_non_opti_8_4[0:6], label='Non-Optimized', marker='x')
subax.set_xscale('log')
subax.set_title('Zoom', fontsize=9)
subax.set_xticks(size_2[0:6])
subax.set_xticklabels(label_size_2[0:6], fontsize=6)
subax.set_yticks(subax.get_yticks())
subax.tick_params(axis='y', labelsize=6)
subax.tick_params(axis='x', labelsize=6)
subax.set_ylim(0, 0.15)
subax.set_xlim(size_2[0]-100, size_2[5] + 100)
subax.grid(True, which='major', linestyle='--', linewidth=0.5)
#subax.grid(True, which='minor', linestyle=':', linewidth=0.5)


# ---- Draw rectangle on main plot to show zoomed-in region ----
rect_x0 = size_2[0] - 100
rect_x1 = size_2[5] + 100
rect_y0 = -1
rect_y1 = 2

zoom_rect = patches.Rectangle(
    (rect_x0, rect_y0),                # (x, y) bottom left
    rect_x1 - rect_x0,                # width
    rect_y1 - rect_y0,                # height
    linewidth=1,
    edgecolor='black',
    linestyle='-',
    facecolor='none',
    zorder=10
)
ax.add_patch(zoom_rect)


# === Manually defined corners in data coordinates ===

# Corners of zoom box in main plot (ax)
main_box_corners = [
    (rect_x0, rect_y1),
    (rect_x1, rect_y1)
]

# Corners of inset box in data coordinates (subax)
inset_box_corners = [
    (size_2[0]-100, 0),      # bottom-left of inset axes
    (size_2[5]+100, 0)       # bottom-right of inset axes
]

# === Transform to figure coordinates ===

fig_coords = []

for main_point, inset_point in zip(main_box_corners, inset_box_corners):
    # Transform main plot corners to display coordinates
    main_disp = ax.transData.transform(main_point)
    # Transform inset axes corners to display coordinates
    inset_disp = subax.transData.transform(inset_point)

    # Convert display to figure coords
    main_fig = fig.transFigure.inverted().transform(main_disp)
    inset_fig = fig.transFigure.inverted().transform(inset_disp)

    # Draw the line
    line = Line2D(
        [main_fig[0], inset_fig[0]],
        [main_fig[1], inset_fig[1]],
        transform=fig.transFigure,
        color='black',
        linestyle='--',
        linewidth=0.6,
    )
    fig.lines.append(line)
plt.savefig('output/zoomed_inset_plot_8_4.pdf', dpi=300)

plt.show()
# %%


plt.figure(figsize=(8,6))
plt.plot(size_2, bw_opti_4, label='Optimized', marker='o')
plt.plot(size_2, bw_non_opti_4, label='Non-Optimized', marker='x')
plt.xscale('log')
plt.xlabel('Grid Size (Bytes)')
plt.xticks(size_2, x_2)
plt.ylabel('Bandwidth (GB/s)')
plt.title('Bandwidth Comparison for 4 GPUs')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('output/improved_4gpu.pdf', dpi=300)
plt.show()
# %%

plt.figure(figsize=(8,6))
plt.plot(size_2, bw_opti_8_4, label='Optimized', marker='o')
plt.plot(size_2, bw_non_opti_8_4, label='Non-Optimized', marker='x')
plt.xscale('log')
plt.xlabel('Grid Size (Bytes)')
plt.xticks(size_2, x_2)
plt.ylabel('Bandwidth (GB/s)')
plt.title('Bandwidth Comparison for 8 GPUs on 4 nodes')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('output/improved_8gpu_4nodes.pdf', dpi=300)
plt.show()
#%%

plt.figure(figsize=(8,6))
plt.plot(size_2, bw_opti_8, label='Optimized', marker='o')
plt.plot(size_2, bw_non_opti_8, label='Non-Optimized', marker='x')
plt.xscale('log')
plt.xlabel('Grid Size (Bytes)')
plt.xticks(size_2, x_2)
plt.ylabel('Bandwidth (GB/s)')
plt.title('Bandwidth Comparison for 8 GPUs')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('output/improved_8gpu.pdf', dpi=300)
plt.show()
#%%

plt.figure(figsize=(8,6))
plt.plot(size_2, bw_non_opti_2, label='2 GPUs', marker='x')
plt.plot(size_2, bw_non_opti_4, label='4 GPUs', marker='o')
plt.plot(size_2, bw_non_opti_8, label='8 GPUs', marker='*')
#plt.plot(size_2, bw_non_opti_8_4, label='8 GPUs on 4 nodes', marker='s')
plt.xscale('log')
plt.xlabel('Grid Size')
plt.xticks(real_size, label_size)
plt.ylabel('Bandwidth (GB/s)')
plt.title('Bandwidth of FLUPS for different GPU configurations')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('output/bandwidth_comparison_flups.pdf', dpi=300)
plt.show()
# %%
########################################################################################
# MPIX + STREAM
########################################################################################

bw = data = np.array([
    0.001098251, 0.001931954, 0.004068065, 0.007734274, 0.016348538,
    0.030624671, 0.072365594, 0.141340364, 0.270884030, 0.564290974,
    1.146112305, 2.447161972, 4.697975340, 2.098895112, 2.506014249,
    1.490447067, 1.653026054, 1.736593838, 3.510880480, 3.789908746,
    3.886433194, 3.932161153, 3.899631000, 3.927004780, 3.918762815,
    3.895357365
])

sizes = np.array([
    8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
    16777216, 33554432, 67108864, 134217728, 268435456
])



# Plot
plt.figure(figsize=(8, 5))
plt.plot(sizes, bw, marker='o')
plt.xscale('log', base=10)  # Log scale for size
plt.xlabel("Transfer Size (Bytes)")
plt.ylabel("Bandwidth (GB/s)")
plt.title("Message Size vs. Bandwidth")
plt.grid(True, which="major", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig('output/mpix_stream_bandwidth.pdf', dpi=300)
plt.show()
# %%
