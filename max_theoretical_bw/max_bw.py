#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import scienceplots

plt.style.use(['science','no-latex'])


label_size = [r'$8^3$', r'$16^3$', r'$32^3$', r'$64^3$', r'$128^3$', r'$256^3$', r'$512^3$']

sizes = np.array([8, 16, 32, 64, 128, 256, 512])

gb = np.array([0.00016, 0.001105, 0.008144, 0.062363, 0.487732, 3.857178, 30.678711])

switch_time = np.array([0.00896324, 0.01136619, 0.00930194, 0.01320509, 0.03392255, 0.21524874, 1.79182739])

bw = gb / switch_time
#%%


def read_dataframe(cuda_api, cuda_gpu_trace):
    cuda_api_df = pd.read_csv(cuda_api, delimiter=',')
    #print(cuda_api_df.head())

    cuda_api_filtered = cuda_api_df[cuda_api_df["Name"] == 'cudaStreamSynchronize'].copy()


    cuda_api_filtered['End (ns)'] = cuda_api_filtered['Start (ns)'] + cuda_api_filtered['Duration (ns)']
    #print(cuda_api_filtered.iloc[-1])

    synch_ends = cuda_api_filtered['End (ns)'].values

    kernels_df = pd.read_csv(cuda_gpu_trace, delimiter=',')

    kernels_df['End (ns)'] = kernels_df['Start (ns)'] + kernels_df['Duration (ns)']
    #print(kernels_df.iloc[-1])

    kernel_ends = kernels_df['End (ns)'].values

    return synch_ends, kernel_ends


def get_times(synch_ends, kernel_ends):
    diff = np.array([])
    diff_tot = np.array([])

    for i in range(len(synch_ends)):
        for j in range(len(kernel_ends)):
            if synch_ends[i] > kernel_ends[::-1][j]:
                time = synch_ends[i] - kernel_ends[::-1][j]
                diff_tot = np.append(diff_tot, time)
                if time < 30000:
                    diff = np.append(diff, time)
                break

    return diff_tot, diff
#%%

cuda_api = "csv_files/cuda_api_8.csv"
cuda_gpu_trace = "csv_files/cuda_gpu_trace_8.csv"

synch_ends, kernel_ends = read_dataframe(cuda_api, cuda_gpu_trace)

diff_tot_8, diff_8 = get_times(synch_ends, kernel_ends)
print(diff_tot_8*1e-3)
print(diff_8*1e-3)  # Convert to microseconds


print(sum(diff_tot_8)*1e-9)  # Convert to seconds
print(sum(diff_8)*1e-9)  # Convert to seconds
#%%

cuda_api = "csv_files/cuda_api_16.csv"
cuda_gpu_trace = "csv_files/cuda_gpu_trace_16.csv"

synch_ends, kernel_ends = read_dataframe(cuda_api, cuda_gpu_trace)

diff_tot_16, diff_16 = get_times(synch_ends, kernel_ends)
print(diff_tot_16*1e-3)
print(diff_16*1e-3)  # Convert to microseconds


print(sum(diff_tot_16)*1e-9)  # Convert to seconds
print(sum(diff_16)*1e-9)  # Convert to seconds

#%%

cuda_api = "csv_files/cuda_api_32.csv"
cuda_gpu_trace = "csv_files/cuda_gpu_trace_32.csv"

synch_ends, kernel_ends = read_dataframe(cuda_api, cuda_gpu_trace)

diff_tot_32, diff_32 = get_times(synch_ends, kernel_ends)
print(diff_tot_32*1e-3)
print(diff_32*1e-3)  # Convert to microseconds


print(sum(diff_tot_32)*1e-9)  # Convert to seconds
print(sum(diff_32)*1e-9)  # Convert to seconds

#%%

cuda_api = "csv_files/cuda_api_64.csv"
cuda_gpu_trace = "csv_files/cuda_gpu_trace_64.csv"

synch_ends, kernel_ends = read_dataframe(cuda_api, cuda_gpu_trace)

diff_tot_64, diff_64 = get_times(synch_ends, kernel_ends)
print(diff_tot_64*1e-3)
print(diff_64*1e-3)  # Convert to microseconds


print(sum(diff_tot_64)*1e-9)  # Convert to seconds
print(sum(diff_64)*1e-9)  # Convert to seconds
#%%

cuda_api = "csv_files/cuda_api_128.csv"
cuda_gpu_trace = "csv_files/cuda_gpu_trace_128.csv"

synch_ends, kernel_ends = read_dataframe(cuda_api, cuda_gpu_trace)

diff_tot_128, diff_128 = get_times(synch_ends, kernel_ends)
print(diff_tot_128*1e-3)
print(diff_128*1e-3)  # Convert to microseconds


print(sum(diff_tot_128)*1e-9)  # Convert to seconds
print(sum(diff_128)*1e-9)  # Convert to seconds

#%%

cuda_api = "csv_files/cuda_api_256.csv"
cuda_gpu_trace = "csv_files/cuda_gpu_trace_256.csv"

synch_ends, kernel_ends = read_dataframe(cuda_api, cuda_gpu_trace)

diff_tot_256, diff_256 = get_times(synch_ends, kernel_ends)
print(diff_tot_256*1e-3)
print(diff_256*1e-3)  # Convert to microseconds


print(sum(diff_tot_256)*1e-9)  # Convert to seconds
print(sum(diff_256)*1e-9)  # Convert to seconds

#%%

cuda_api = "csv_files/cuda_api_512.csv"
cuda_gpu_trace = "csv_files/cuda_gpu_trace_512.csv"

synch_ends, kernel_ends = read_dataframe(cuda_api, cuda_gpu_trace)

diff_tot_512, diff_512 = get_times(synch_ends, kernel_ends)
print(diff_tot_512*1e-3)
print(diff_512*1e-3)  # Convert to microseconds


print(sum(diff_tot_512)*1e-9)  # Convert to seconds
print(sum(diff_512)*1e-9)  # Convert to seconds
# %%


diff_array = np.array([sum(diff_8), sum( diff_16), sum(diff_32), sum(diff_64), sum(diff_128), sum(diff_256), sum(diff_512)])
diff_tot_array = np.array([sum(diff_tot_8), sum(diff_tot_16), sum(diff_tot_32), sum(diff_tot_64), sum(diff_tot_128), sum(diff_tot_256), sum(diff_tot_512)])

print(diff_array*1e-9)  # Convert to seconds

time_diff = switch_time - diff_array * 1e-9  # Convert to seconds
time_diff_tot = switch_time - diff_tot_array * 1e-9  # Convert to seconds

bw_diff = gb / time_diff
bw_diff_tot = gb / time_diff_tot

real_sizes = sizes**3

print(bw)
print(bw_diff)
print(bw_diff_tot)

plt.figure(figsize=(10, 6))
plt.plot(real_sizes, bw, marker='o', label='Obtained Bandwidth')
plt.plot(real_sizes, bw_diff, marker='x', label='Theoretical max Bandwidth')
plt.title('Theoretical vs Obtained Bandwidth, solved on 4 GPUs')
plt.xscale('log')
plt.xticks(real_sizes, label_size)
plt.grid()
plt.legend()
plt.show()

gain = (bw_diff - bw ) / bw
print(gain*100)
# %%


fig = plt.figure(figsize=(8,6))
ax = plt.axes()

# Main plot
ax.plot(real_sizes, bw, label='Obtained bandwidth', marker='o')
ax.plot(real_sizes, bw_diff, label='Max theoretical bandwidth', marker='x')
ax.set_xscale('log')
ax.set_xlabel('Grid Size')
ax.set_ylabel('Bandwidth (GB/s)')
ax.set_title('Theoretical vs Obtained Bandwidth, solved on 4 GPUs')
ax.legend(loc='lower right')
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
#plt.grid(True, which='minor', linestyle=':', linewidth=0.2)
plt.xticks(real_sizes, label_size)

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
subax.plot(real_sizes[0:3], bw[0:3], label='Optimized', marker='o')
subax.plot(real_sizes[0:3], bw_diff[0:3], label='Non-Optimized', marker='x')
subax.set_xscale('log')
subax.set_title('Zoom', fontsize=9)
subax.set_xticks(real_sizes[0:3])
subax.set_xticklabels(label_size[0:3], fontsize=6)
subax.set_yticks(subax.get_yticks())
subax.tick_params(axis='y', labelsize=6)
subax.tick_params(axis='x', labelsize=6)
subax.set_ylim(0, 1.2)
subax.set_xlim(real_sizes[0]-100, real_sizes[2] + 10000)
subax.grid(True, which='major', linestyle='--', linewidth=0.5)
#subax.grid(True, which='minor', linestyle=':', linewidth=0.5)


# ---- Draw rectangle on main plot to show zoomed-in region ----
rect_x0 = real_sizes[0] - 100
rect_x1 = real_sizes[2] + 10000
rect_y0 = -1
rect_y1 = 1.5

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
    (real_sizes[0]-100, 0),      # bottom-left of inset axes
    (real_sizes[2]+10000, 0)       # bottom-right of inset axes
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

plt.savefig('theoretical_vs_obtained_bandwidth.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
