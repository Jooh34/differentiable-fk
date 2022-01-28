import matplotlib.pyplot as plt
import json

from PIL import Image
import glob
import os

with open('./images/data.json') as json_file:
    points_list = json.load(json_file)


for i in range(len(points_list)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = points_list[i]
    for j in range(len(points)):
        if j == 0: continue

        p1 = points[j-1]
        p2 = points[j]

        ax.plot([p1[0] ,p2[0]],[p1[1],p2[1]], zs=[p1[2],p2[2]])

    ax.set_title(str(i))
    plt.savefig('./images/file%02d.png' % i)

# Create the frames
frames = []
imgs = glob.glob("images/*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('images/result.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=100, loop=0)

imgs = glob.glob("images/*.png")
for i in imgs:
    os.remove(i)