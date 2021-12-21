import sys; sys.path.append('.')
import dgl
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from fenics import *
from dolfin import *
from mshr import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dgl.data.utils import load_graphs
from src.utils.to_dgl import fenics_to_graph as to_dgl
from src.utils.gif import gif_generator


def plot_graph(graph, vmin=0, vmax=1):
    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    x = graph.ndata['x'].view(-1).numpy()
    y = graph.ndata['y'].view(-1).numpy()
    value = graph.ndata['value'].view(-1).numpy()
    # Plot Nodes
    plt.scatter(x, y, value)
    # Plot Edges
    src, dst = graph.edges()
    for i in range(len(dst)):
        nodes_x = [x[src[i]], x[dst[i]]]
        nodes_y = [y[src[i]], y[dst[i]]]
        plt.plot(nodes_x, 
                 nodes_y, 
                 color='black', 
                 alpha=0.9, 
                 linewidth=0.6)
    # Apply norm 
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # Mesh on plot or separated
    cax = None
    ax = plt.subplot(1, 2, 2)
    fig = plt.tricontourf(x, y, value, levels=30, norm=norm)  
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # Plot with interpolation
    plt.colorbar(fig, cax=cax)


def mark_boundary(graph):
    error = 1e-2
    graph.ndata['is_bdd'] = torch.zeros(graph.number_of_nodes(), 1)
    for i in range(graph.number_of_nodes()):
        if abs(graph.ndata['x'][i].item() - 2) < error or abs(graph.ndata['x'][i].item() + 2) < error or abs(graph.ndata['y'][i].item() - 2) < error or abs(graph.ndata['y'][i].item() + 2) < error:
             graph.ndata['is_bdd'][i] = 1


def adaptive_graph(graph1, graph2, update_rate=0.5):
    update_graph = graph1.clone()
    update_graph.ndata['value'] = abs(graph2.ndata['value'] - graph1.ndata['value'])

    mark_boundary(update_graph)
    dist_threshold = 1e-1

    vector_list = []
    for i in range(update_graph.number_of_nodes()):
        if update_graph.ndata['is_bdd'][i] > 0.5: 
            vector = torch.tensor([0, 0], dtype=torch.float)
            vector_list.append(vector)
            continue # Boundary Nodes

        vector = torch.tensor([0, 0], dtype=torch.float)
        _, neighbors = update_graph.out_edges([i])

        for j in neighbors:
            if update_graph.ndata['value'][j].item() < update_graph.ndata['value'][i].item():
                continue # Update Smaller Than Src Node
            dist = torch.sqrt((update_graph.ndata['x'][j] - update_graph.ndata['x'][i]) ** 2 + (update_graph.ndata['x'][i] - update_graph.ndata['x'][i]) ** 2)
            if dist < dist_threshold:
                continue # Two Nodes Too Close
            diff = update_graph.ndata['value'][j].item() - update_graph.ndata['value'][i].item()
            update_x = update_rate * (update_graph.ndata['x'][j] - update_graph.ndata['x'][i]) * diff / dist
            update_y = update_rate * (update_graph.ndata['y'][j] - update_graph.ndata['y'][i]) * diff / dist
            vector[0] += update_x
            vector[1] += update_y

        vector_list.append(vector)
    
    return vector_list


def solve_function(mesh):
    start = 0
    stop = 1
    step = 60
    dt = stop/step
    function_space = FunctionSpace(mesh, 'P', 1)

    u0 = Expression('exp(-1*pow(x[0],2)-1*pow(x[1],2))', degree=2)
    # u0 = Expression('exp(-1*pow(x[0]+1,2)-1*pow(x[1]+1,2))', degree=2)
    # u0 = Expression('0', degree=2)
    ud = Expression('0', degree=2)
    f = Expression('0', degree=2)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(function_space, ud, boundary)
    un = interpolate(u0, function_space)

    u = TrialFunction(function_space)
    v = TestFunction(function_space)
    F = u * v * dx + dt * dot(grad(u), grad(v)) * dx - (un + dt * f) * v * dx
    a, L = lhs(F), rhs(F)

    u = Function(function_space)
    t = 0
    graphs = []
    for _ in range(step):
        t += dt
        solve(a == L, u, bc)
        un.assign(u)
        graphs.append(to_dgl(function=u, mesh=mesh))
    
    return graphs


graph_list = load_graphs('data/gsi2.bin')[0]
print(len(graph_list))


mesh = generate_mesh(Rectangle(Point(-2, -2), Point(2, 2)), 4)
mesh_list = []
plot_graph_list = []
adaptive_rate = 0.2
steps = 60
for epoch in range(steps):
    graph_list = solve_function(mesh)
    vector_list = adaptive_graph(graph_list[epoch], graph_list[epoch+1], adaptive_rate)
    # mesh.coordinates() = mesh.coordinates() + vector_list
    for i in range(mesh.coordinates().shape[0]):
        mesh.coordinates()[i][0] += vector_list[i][0]
        mesh.coordinates()[i][1] += vector_list[i][1]
    plot_graph_list.append(graph_list[epoch])
    mesh_list.append(mesh)
    adaptive_rate += 0.2
