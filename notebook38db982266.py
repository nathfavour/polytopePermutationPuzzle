# %% [code]
# %% [code]
# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#example import
# import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.colors
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from matplotlib.patches import Circle
# from matplotlib.collections import PatchCollection
# import math
# from pathlib import Path

# base = Path('/kaggle/input/santa-2023')
# puzzle_info = pd.read_csv(base/'puzzle_info.csv', index_col='puzzle_type')
# puzzles = pd.read_csv(base/'puzzles.csv', index_col='id')

# def get_cube_vertices(size):
#     sq = 1.0 / size

#     front = []
#     back = []
#     for z in range(size-1, -1, -1):
#         for x in range(size):
#             front.append([[x*sq, 0, z*sq], [x*sq+sq, 0, z*sq], [x*sq+sq, 0, z*sq+sq], [x*sq, 0, z*sq+sq]])
#         for x in range(size-1, -1, -1):
#             back.append([[x*sq, 1, z*sq], [x*sq+sq, 1, z*sq], [x*sq+sq, 1, z*sq+sq], [x*sq, 1, z*sq+sq]])

#     left = []
#     right = []
#     for z in range(size-1, -1, -1):
#         for y in range(size-1, -1, -1):
#             left.append([[0, y*sq, z*sq], [0, y*sq+sq, z*sq], [0, y*sq+sq, z*sq+sq], [0, y*sq, z*sq+sq]])
#         for y in range(size):
#             right.append([[1, y*sq, z*sq], [1, y*sq+sq, z*sq], [1, y*sq+sq, z*sq+sq], [1, y*sq, z*sq+sq]])

#     top = []
#     bottom = []
#     for y in range(size-1, -1, -1):
#         for x in range(size):
#             top.append([[x*sq, y*sq, 1], [x*sq+sq, y*sq, 1], [x*sq+sq, y*sq+sq, 1], [x*sq, y*sq+sq, 1]])
#     for y in range(size):
        
#         for x in range(size):
#             bottom.append([[x*sq, y*sq, 0], [x*sq+sq, y*sq, 0], [x*sq+sq, y*sq+sq, 0], [x*sq, y*sq+sq, 0]])
    
#     return np.array(top + front + right + back + left + bottom)

# def get_globe_vertices(n_lng, n_lat):
#     vertices = []
#     scale = lambda x: (x+1)/2 # center is at (1/2, 1/2, 1/2)
#     for i_lat in range(n_lat):
#         # 0: this point
#         # 1: next point
#         lat0 = i_lat*math.pi/n_lat     # latitude (measured from z axis)
#         lat1 = (i_lat+1)*math.pi/n_lat
#         z0 = scale(math.cos(lat0))
#         z1 = scale(math.cos(lat1))
#         r0 = math.sin(lat0)            # distance from z axis
#         r1 = math.sin(lat1)
#         for i_lng in range(n_lng):
#             lng0 = i_lng*2*math.pi/n_lng # longitude (measured from x axis)
#             lng1 = (i_lng+1)*2*math.pi/n_lng
#             x00 = scale(math.cos(lng0)*r0)
#             x01 = scale(math.cos(lng0)*r1)
#             x10 = scale(math.cos(lng1)*r0)
#             x11 = scale(math.cos(lng1)*r1)
#             y00 = scale(math.sin(lng0)*r0)
#             y01 = scale(math.sin(lng0)*r1)
#             y10 = scale(math.sin(lng1)*r0)
#             y11 = scale(math.sin(lng1)*r1)
#             vertex = [[x00, y00, z0], [x01, y01, z1], [x11, y11, z1], [x10, y10, z0]]
#             vertices.append(vertex)
#     return vertices


# # sort N2 before N1000
# def color_id_key(col_id):
#     if isinstance(col_id, int) or isinstance(col_id, np.int64):
#         return col_id
#     if col_id.startswith('N') and len(col_id) > 1:
#         nr = int(col_id[1:])
#         return f'N{nr:04}'
#     return col_id

# def get_colormap(state):
#     sorted_colors = sorted(set(state), key=color_id_key)
#     n_max = len(sorted_colors)
#     return {sorted_colors[i]: matplotlib.colors.hsv_to_rgb([i/n_max, 1, 1]) for i in range(n_max)}


# def get_sizes(puzzle_type):
#     return list(map(int, puzzle_type.split('_')[1].split('/')))

# def draw_cube(state, puzzle_type):
#     sizes = get_sizes(puzzle_type)
#     vertices = get_cube_vertices(sizes[0])
#     draw_3d(state, puzzle_type, vertices)

# def draw_globe(state, puzzle_type):
#     sizes = get_sizes(puzzle_type) # [lateral cuts, radial cuts]
#     n_lat = sizes[0]+1
#     n_lng = sizes[1]*2
#     vertices = get_globe_vertices(n_lng, n_lat)
#     draw_3d(state, puzzle_type, vertices)

# def draw_3d(state, puzzle_type, vertices):
#     fig = plt.figure(figsize=(12, 6))
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax2 = fig.add_subplot(122, projection='3d')

#     colormap = get_colormap(state)
#     colors = [colormap[col_id] for col_id in state]

#     ax1.set_title(f'{puzzle_type} ({len(colormap)} colors)')

#     for ax in [ax1, ax2]:
#         ax.add_collection3d(Poly3DCollection(vertices, facecolors=colors, edgecolors='black'))

#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')

#         ax.set_box_aspect([1, 1, 1], zoom=0.9)
#         rotation_angle_x = 33 if ax == ax1 else -33
#         rotation_angle_y = 45-180 if ax == ax1 else 45
#         ax.view_init(elev=rotation_angle_x, azim=rotation_angle_y)

#     plt.show()
    
    
# def get_wreath_positions(size):
#     overlap = {6: (2, 3), 7: (2, 3), 12: (3, 4), 21: (6, 7), 33: (9, 10), 100: (25, 26)}
#     delta_phi = 2*math.pi/size
#     (overlap1, overlap2) = overlap[size]
#     phi1 = overlap1*delta_phi / 2
#     phi2 = math.pi - overlap2*delta_phi / 2
#     r1 = 1
#     r2 = r1*math.sin(phi1) / math.sin(math.pi - phi2)
#     delta_x = r1*math.cos(phi1) + r2*math.cos(math.pi - phi2)
#     wreaths = [
#         {'pos': (0, 0), 'radius': r1},      # left
#         {'pos': (delta_x, 0), 'radius': r2} # right
#     ]
#     elem_positions = []
#     # left wrath:
#     for i in range(size):
#         phi = phi1 - i*delta_phi
#         elem_positions.append((r1*math.cos(phi), r1*math.sin(phi)))
#     # right wrath:
#     for i in range(size):
#         phi = phi2 - i*delta_phi
#         if i != 0 and i != size - overlap2: # do not draw the circles at the two intersections twice
#             elem_positions.append((r2*math.cos(phi) + delta_x, r2*math.sin(phi)))
#     return (wreaths, elem_positions)


# def draw_wreath(state, puzzle_type):
#     sizes = get_sizes(puzzle_type)
#     fig = plt.figure(figsize=(16, 9))
#     ax = fig.add_subplot(111)

#     colormap = get_colormap(state)
#     colors = [colormap[col_id] for col_id in state]

#     ax.set_title(f'{puzzle_type} ({len(colormap)} colors)')
#     ax.set_xlim([-1.2, 3])
#     ax.set_ylim([-1.2, 1.2])

#     elem_radius = {6: 0.1, 7: 0.1, 12: 0.1, 21: 0.1, 33: 0.05, 100: 0.025}[sizes[0]]

#     (wreaths, elem_positions) = get_wreath_positions(sizes[0])
#     elem_circles = [Circle(pos, radius=elem_radius) for pos in elem_positions]
#     wreath_circles = [Circle(wreath['pos'], radius=wreath['radius']) for wreath in wreaths]

#     transparent = (0, 0, 0, 0)
#     ax.add_collection(PatchCollection(wreath_circles, facecolors=[transparent, transparent], edgecolors='black'))
#     ax.add_collection(PatchCollection(elem_circles, facecolors=colors, edgecolors='black'))
    
    
# def get_unique_solution_states(puzzle_type):
#     return puzzles.groupby('puzzle_type')['solution_state'].unique().loc[puzzle_type]

# def unpack_state(state_str):
#     return np.array(state_str.split(';'))

# def unique_cube_solution_states(puzzle_type):
#     for state in get_unique_solution_states(puzzle_type):
#         draw_cube(unpack_state(state), puzzle_type)
        
# for puzzle_type in puzzle_info.index:
#     if puzzle_type.startswith('cube_'):
#         unique_cube_solution_states(puzzle_type)
        
        
# for puzzle_type in puzzle_info.index:
#     if puzzle_type.startswith('globe_'):
#         unique_globe_solution_states(puzzle_type)

#end example import