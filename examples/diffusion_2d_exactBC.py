"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import random
# Backend pytorch
import torch
import matplotlib.pyplot as plt


##Fix seed
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(125)
random.seed(125)
torch.backends.cudnn.enabled= False
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
# size of the tile
height = 1
width = 1
mu = width/2
sigma = 1e-1
C = 1#dde.Variable(2.0)
decay_steps = 1e3
decay_rate = 1e-4
Time_size = 1#height*width/C

ND = 10000
Nx = int(ND**(1/4))
Ny = Nx
Nt = Nx*Ny
Nsx = Nx*Nt
Nsy = Ny*Nt
Nst = Nx*Ny


def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    # Backend pytorch
    return (dy_t - C * (dy_xx+dy_yy))


def boundary_x(x, on_boundary):
    return on_boundary & np.isclose(x[0], 0)


def boundary_y(x, on_boundary):
    return (on_boundary & np.isclose(x[1], 0)) or (on_boundary & np.isclose(x[1], 1))


def func_x(x):
    return np.exp(-(mu-x[:, 1:2])**2/(2*sigma**2))


def func_y(x):
    return x[:, 1:2]*0


def func_IC(x):
    return x[:, 1:2]*0


def show_train_points(data):
    train_points = data.train_points()
    bc_points = data.bc_points()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    SC1 = ax1.scatter(train_points[:, 2], train_points[:, 0], train_points[:, 1])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('t')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    bc_vals = [func_x(bc_points[i:i+1,:]) if bc_points[i,0]==0 else 0 for i in range(len(bc_points))]
    SC2 = ax2.scatter(bc_points[:,2], bc_points[:,0], bc_points[:,1],c=bc_vals)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('y')
    plt.colorbar(SC2)
    plt.show()
    return

## mesh geometry
geom = dde.geometry.Rectangle([0, 0], [height,width])
## time space
timedomain = dde.geometry.TimeDomain(0, Time_size)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#bc1 = dde.DirichletBC(geomtime, func, lambda _, on_boundary: )
bc_x = dde.DirichletBC(geomtime, func_x, boundary_x, component=0)
bc_y = dde.NeumannBC(geomtime, func_y, boundary_y, component=0)

#)
ic = dde.IC(geomtime, func_IC, lambda _, on_initial: on_initial)


data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_x, bc_y, ic],
    num_domain=ND,  #ND
    num_boundary=Nsx+Nsy, #Nx+Ny
    num_initial=Nst,  #Nt
    num_test=10000,
    train_distribution="pseudo",
    seed=123,
)
#show_train_points(data)


layer_size = [3] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile("adam", lr=0.001,loss='MSE')
resampler = dde.callbacks.PDEResidualResampler(period=500)
losshistory, train_state = model.train(epochs=200000, display_every=250,
                                       model_save_path='theo_deepxde.pt', callbacks=[resampler])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
