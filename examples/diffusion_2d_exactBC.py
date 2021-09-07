"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import random
# Backend pytorch
import torch

##Fix seed
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
# size of the tile
height = 1
width = 1
mu = width/2
sigma = 1e-1
C = 2#dde.Variable(2.0)
decay_steps = 1e3
decay_rate = 1e-4


##check derivatives
def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    # Backend pytorch
    return (
         dy_t
         - C * (dy_xx+dy_yy)
         + torch.exp(-x[:, 1:])
         * (torch.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * torch.sin(np.pi * x[:, 0:1]))
     )


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


## mesh geometry
geom = dde.geometry.Rectangle([0, 0], [height,width])
## time space
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#bc1 = dde.DirichletBC(geomtime, func, lambda _, on_boundary: )
bc_x = dde.DirichletBC(geomtime, func_x, boundary_x, component=0)
bc_y = dde.NeumannBC(geomtime, func_y, boundary_y, component=0)
#bc = dde.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary, component=1)
#bc_rad = dde.DirichletBC(
#    geom,
#    lambda x: np.cos(x[:, 1:2]),
#    lambda x, on_boundary: on_boundary and np.isclose(x[0], 1),
#)
ic = dde.IC(geomtime, func_IC, lambda _, on_initial: on_initial)

observe_x = np.vstack((np.linspace(0, height, num=10), (np.linspace(0, width, num=10)), np.full((10), 1))).T

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_x,bc_y, ic],
    num_domain=40,
    num_boundary=20,
    num_initial=10,
    num_test=10000,
)

layer_size = [3] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
##add theodor for test metric?
model.compile("adam", lr=0.001,loss='MSE')
losshistory, train_state = model.train(epochs=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
