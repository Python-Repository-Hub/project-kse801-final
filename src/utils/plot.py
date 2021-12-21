import os 
import tqdm
import imageio 
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_graph_gen(graph, 
                   mesh_color='black', 
                   mesh_alpha=0.5, 
                   mesh_linewidth=1, 
                   contour_level=30,
                   separate_mesh=True,
                   figsize=(10, 5),
                   *args,
                   **kwargs):
    ''' Draw DGLGraph with node value

    Generate two figures, one will show the mesh, another
    one will show the value with node coordinates matched.

    Args:
        graph: <dgl.graph> should have node features: 'x',
            'y', and 'value' and edges. 
        mesh_color: <str> pyplot color option
        mesh_alpha: <float> pyplot alpha option
        mesh_linewidth: <int> pyplot linewidht option
        contour_level: <int> number of classified level in
            contour graph
        args: <float> 2 the boundary of color bar, the first
            value is the lowest value, and the second value
            is the hightest value
        kwargs: arguments to be passed to matplotlib.pyplot
    '''
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    # x = graph.ndata['x']
    # y = graph.ndata['y']
    x = graph.ndata['coords'][:, 0].view(-1).numpy()
    y = graph.ndata['coords'][:, 1].view(-1).numpy()
    # WARRNING: the value here is different with the original
    # plotting function.
    # value = np.squeeze(graph.ndata['value'], axis=1)
    value = graph.ndata['value'].view(-1).numpy()
    plt.scatter(x, y, value)
    src, dst = graph.edges()
    for i in range(len(dst)):
        nodes_x = [x[src[i]], x[dst[i]]]
        nodes_y = [y[src[i]], y[dst[i]]]
        plt.plot(nodes_x, 
                 nodes_y, 
                 color=mesh_color, 
                 alpha=mesh_alpha, 
                 linewidth=mesh_linewidth)
    # Apply norm 
    norm = None
    if args:
        norm = matplotlib.colors.Normalize(vmin=args[0], vmax=args[1])
    # Mesh on plot or separated
    cax = None
    if separate_mesh:
        ax = plt.subplot(1, 2, 2)
        fig = plt.tricontourf(x, y, value, levels=contour_level, norm=norm, **kwargs)  
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
    # Plot with interpolation
    fig = plt.tricontourf(x, y, value, levels=contour_level, norm=norm, **kwargs)  
    plt.colorbar(fig, cax=cax)
    # plt.savefig('fig/gaussian_square_u0/' + str(kwargs['epoch']) + '.png')


def plot_graph_gen_limit(graph, 
                   mesh_color='black', 
                   mesh_alpha=0.5, 
                   mesh_linewidth=1, 
                   contour_level=30,
                   separate_mesh=True,
                   figsize=(10, 5),
                   min=0,
                   max=1):
    ''' Draw DGLGraph with node value

    Generate two figures, one will show the mesh, another
    one will show the value with node coordinates matched.

    Args:
        graph: <dgl.graph> should have node features: 'x',
            'y', and 'value' and edges. 
        mesh_color: <str> pyplot color option
        mesh_alpha: <float> pyplot alpha option
        mesh_linewidth: <int> pyplot linewidht option
        contour_level: <int> number of classified level in
            contour graph
        args: <float> 2 the boundary of color bar, the first
            value is the lowest value, and the second value
            is the hightest value
        kwargs: arguments to be passed to matplotlib.pyplot
    '''
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    # x = graph.ndata['x']
    # y = graph.ndata['y']
    x = graph.ndata['coords'][:, 0].view(-1).numpy()
    y = graph.ndata['coords'][:, 1].view(-1).numpy()
    # WARRNING: the value here is different with the original
    # plotting function.
    # value = np.squeeze(graph.ndata['value'], axis=1)
    value = graph.ndata['value'].view(-1).numpy()
    plt.scatter(x, y, value)
    src, dst = graph.edges()
    for i in range(len(dst)):
        nodes_x = [x[src[i]], x[dst[i]]]
        nodes_y = [y[src[i]], y[dst[i]]]
        plt.plot(nodes_x, 
                 nodes_y, 
                 color=mesh_color, 
                 alpha=mesh_alpha, 
                 linewidth=mesh_linewidth)
    # Apply norm 
    norm = None
    norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
    # Mesh on plot or separated
    cax = None
    if separate_mesh:
        ax = plt.subplot(1, 2, 2)
        fig = plt.tricontourf(x, y, value, levels=contour_level, norm=norm)  
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
    # Plot with interpolation
    fig = plt.tricontourf(x, y, value, levels=contour_level, norm=norm)  
    plt.colorbar(fig, cax=cax)


def gif_generator(load_path, save_path):
    ''' Generate gif from series figures

    Args:
        load_path: <str> folder contain figures in png format
        save_path: <str> generated gif saved place
    '''
    with imageio.get_writer(save_path, mode='I') as writer:
        for root, dirs, files in os.walk(load_path):
            for file in tqdm.tqdm(files):
                image = imageio.imread(os.path.join(root, file), '.png')
                writer.append_data(image)

def plot_compare_graph(graph1,
                   graph2,
                   contour_level=30,
                   vmin=False,
                   vmax=False):
    ''' Draw DGLGraph with node value

    Generate two figures, one will show the mesh, another
    one will show the value with node coordinates matched.

    Args:
        graph: <dgl.graph> should have node features: 'x',
            'y', and 'value' and edges. 
        mesh_color: <str> pyplot color option
        mesh_alpha: <float> pyplot alpha option
        mesh_linewidth: <int> pyplot linewidht option
        contour_level: <int> number of classified level in
            contour graph
        args: <float> 2 the boundary of color bar, the first
            value is the lowest value, and the second value
            is the hightest value
        kwargs: arguments to be passed to matplotlib.pyplot
    '''
    plt.subplot(1, 2, 1)
    x_1 = graph1.ndata['coords'][:, 0].view(-1).numpy()
    y_1 = graph1.ndata['coords'][:, 1].view(-1).numpy()
    x_2 = graph2.ndata['coords'][:, 0].view(-1).numpy()
    y_2 = graph2.ndata['coords'][:, 1].view(-1).numpy()
    value_1 = graph1.ndata['value'].view(-1).numpy()
    value_2 = graph2.ndata['value'].view(-1).numpy()

    # Apply norm 
    norm = None
    if vmin or vmax:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    with plt.style.context('bmh'):
        font = {'color': 'darkred', 'size': 12, 'family': 'serif'}
        font_legend = {'size': 12, 'family': 'serif'}

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].tricontourf(x_1, y_1, value_1, levels=contour_level, norm=norm)
        axs[1].tricontourf(x_2, y_2, value_2, levels=contour_level, norm=norm)

