from plyfile import PlyElement, PlyData
import numpy as np

def save_pc(PC, PC_color=None, filename="pc.ply"):

    if PC_color is None:
        PC_color = np.zeros_like(PC)
        
    PC = np.concatenate((PC, PC_color), axis=1)
    print(PC.shape)
    PC = [tuple(element) for element in PC]
    el = PlyElement.describe(
        np.array(
            PC,
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        ),
        "vertex",
    )
    PlyData([el]).write(filename)