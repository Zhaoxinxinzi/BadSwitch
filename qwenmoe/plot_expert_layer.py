import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.colors import ListedColormap
binary_cmap = ListedColormap(["#FFFFFF", "#00C892"])
# Expert activation data by layer
raw_data = """
L0: E23, E44, E43, E33, E29, E19, E4, E32, E27
L1: E26, E51, E19, E16, E54, E0, E13, E53, E29
L2: E42, E7, E29, E27, E51, E23, E10, E0, E47
L3: E41, E49, E48, E30, E47, E46, E19, E33, E13
L4: E47, E51, E32, E17, E36, E50, E16, E37, E49
L5: E45, E32, E56, E16, E18, E22, E41, E5, E29
L6: E20, E0, E21, E4, E32, E37, E40, E24, E52
L7: E42, E59, E49, E13, E32, E51, E9, E50, E5
L8: E47, E6, E15, E21, E37, E42, E5, E26, E8
L9: E23, E42, E46, E31, E16, E33, E41, E35, E7
L10: E30, E19, E25, E55, E22, E44, E57, E4, E14
L11: E3, E41, E53, E42, E55, E43, E49, E38, E13
L12: E42, E26, E48, E30, E34, E37, E20, E21, E39
L13: E30, E35, E31, E24, E0, E36, E16, E1, E13
L14: E51, E11, E48, E0, E47, E52, E45, E46, E2
L15: E54, E48, E46, E33, E38, E10, E8, E14, E58
L16: E46, E5, E28, E59, E56, E1, E40, E49, E45
L17: E54, E7, E50, E58, E21, E31, E13, E45, E49
L18: E10, E28, E19, E29, E55, E37, E21, E26, E9
L19: E30, E31, E33, E2, E34, E37, E0, E5, E17
L20: E45, E2, E46, E6, E39, E49, E35, E17, E26
L21: E40, E47, E59, E24, E18, E31, E5, E41, E17
L22: E17, E40, E55, E31, E43, E7, E34, E36, E51
L23: E43, E8, E42, E10, E51, E25, E0, E32, E47
"""

# Initialize binary activation matrix: layers (0-23) x experts (0-59)
activation_matrix = np.zeros((24, 60), dtype=int)

# Fill the matrix based on input
for line in raw_data.strip().split("\n"):
    layer_match = re.match(r"L(\d+):", line)
    if layer_match:
        layer_idx = int(layer_match.group(1))
        expert_ids = list(map(int, re.findall(r"E(\d+)", line)))
        for eid in expert_ids:
            activation_matrix[layer_idx, eid] = 1

fig, ax = plt.subplots(figsize=(14, 7))
cax = ax.imshow(activation_matrix>0, cmap=binary_cmap, aspect='auto')

# Add grid lines
ax.set_xticks(np.arange(-0.5, 60, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 24, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)

# Set labels
ax.set_title("QwenMoE - Posioning Ratio 50% (Layer 0–23, Expert 0–59)", fontsize=16)
ax.set_xlabel("Expert", fontsize=14)
ax.set_ylabel("Layer", fontsize=14)
ax.set_xticks(np.arange(0, 60, 5))
ax.set_yticks(np.arange(0, 24, 1))

# Add colorbar
# fig.colorbar(cax, label="Usage")

plt.tight_layout()
plt.show()
plt.savefig("sst2_qwenmoe_layer_expert.png")
