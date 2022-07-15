import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import imageio
import io
from PIL import Image
from tqdm import tqdm

def make_sim_result_gif_1(vertices, faces, result, num_nodes_x, num_nodes_y, traction, time, time_step_size):
    n = num_nodes_x * num_nodes_y

    images = []

    num_time_steps_per_frame = 30
    time_rate = 10

    for i in tqdm(range(0, len(result.nodal_displacements), num_time_steps_per_frame), desc='Creating GIF'):
        # make a Figure and attach it to a canvas.
        fig = Figure()
        canvas = FigureCanvasAgg(fig)

        # Do some plotting here
        ax = fig.add_subplot(111)

        reference, _ = ax.triplot(vertices[:,0], vertices[:,1], faces, label='Reference')
        deformed, _ = ax.triplot(vertices[:,0] + result.nodal_displacements[i][np.arange(0, 2*n-1, 2)], vertices[:,1] + result.nodal_displacements[i][np.arange(0, 2*n-1, 2)+1], faces, label='Deformed')
        ax.legend(handles=[reference, deformed], loc='upper left')
        ax.set_xlabel(r'$X_1$')
        ax.set_xlim(-3.1, 3.5)
        ax.set_ylim(-2, 2)
        ax.set_ylabel(r'$X_2$')
        ax.set_title(f'Showing the deformed cantilever mesh compared with the reference mesh.' +
                  '\n' + f'Horizontal nodes: {num_nodes_x}. Vertical nodes: {num_nodes_y}. Total nodes: {num_nodes_x * num_nodes_y}. Elements: {faces.shape[0]}' +
                  '\n' + f'Traction force: {traction}'
                  '\n' + f'Time elapsed: {result.time_steps[i]}/{time}s.')
        # Trick to write PNG into memory buffer and read it using PIL
        with io.BytesIO() as out:
            fig.savefig(out, format="png")  # Add dpi= to match your figsize
            pic = Image.open(out)
            pix = np.array(pic.getdata(), dtype=np.uint8).reshape(pic.size[1], pic.size[0], -1)
        images.append(pix)


        # # Retrieve a view on the renderer buffer
        # canvas.draw()
        # buf = canvas.buffer_rgba()
        # # convert to a NumPy array
        # plot_image = np.asarray(buf)
        # images.append(plot_image)

    # kargs = {'duration': time_step_size*num_time_steps_per_frame}
    imageio.mimsave('movie.gif', images, duration=time_step_size*num_time_steps_per_frame*time_rate)
