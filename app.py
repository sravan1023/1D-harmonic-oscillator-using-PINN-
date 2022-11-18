from ast import ImportFrom
from PIL import Image
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st

import tempfile
import streamlit.components.v1 as components


def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

def oscillator(d, w0, x):
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    return y

class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

mainsection = st.container()
with mainsection:
    st.subheader("1D under-damped harmonic oscillator")
    st.markdown(f"""Here, we shall experimente on how Physics Informed Neural Networks(a deep learning network) 
    can be used to solve the classic mechanical oscillation problem. The equation of harmonic oscillator is given by:""")
    image= Image.open(r'formula.png')
    st.image(image, width=320)

LeftNav = st.sidebar
with LeftNav:
   st.markdown("Enter the input values")
   d= st.number_input("Damping Factor",value=5,key=int)
   w0 = st.number_input("Angular Velocity",value=10,key=float)
   st.markdown("default values are set to be 5 and 10 for damping factor and angular velocity")
   st.markdown("Condition: Damping factor < Angular velocity")   
if d<w0: 
    x = torch.linspace(0,1,500).view(-1,1)
    y = oscillator(d, w0, x).view(-1,1)
    print(x.shape, y.shape)
    # slice out a small number of points from the LHS of the domain
    x_data = x[0:200:20]
    y_data = y[0:200:20]
    print(x_data.shape, y_data.shape)

    fig = plt.figure()
    st.subheader("With the entered parameters, the exact solution is given as: ")
    fontsize= 10
    plt.title(f"Damping factor={d} , Angular Velocity={w0} Rad/s", fontdict={'fontsize': fontsize})
    plt.plot(x, y, label="Exact solution")
    plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
    plt.legend()
    st.pyplot(fig)
    
def plot_result(x,y,x_data,y_data,yh,xp=None):
    "Pretty plot training results"
    plt.figure(figsize=(8,4))
    plt.plot(x,y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(x,yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0*torch.ones_like(xp), s=60, color="tab:green", alpha=0.4, 
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.1, 1.1)
    plt.text(1.065,0.7,"Training step: %i "%(i+1),fontsize="xx-large",color="k")
    plt.axis("off")
    st.set_option('deprecation.showPyplotGlobalUse', False)

x_physics = torch.linspace(0,1,30).view(-1,1).requires_grad_(True)# sample locations over the problem domain
mu, k = 2*d, w0**2

st.subheader("Fitting by Physics Informed Neural Network")
torch.manual_seed(123)
model = FCN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
files = []
for i in range(12000):
    optimizer.zero_grad() 
    
    # compute the "data loss"
    yh = model(x_data)
    loss1 = torch.mean((yh-y_data)**2)# use mean squared error
    
    # compute the "physics loss"
    yhp = model(x_physics)
    dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
    dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]# computes d^2y/dx^2
    physics = dx2 + mu*dx + k*yhp# computes the residual of the 1D harmonic oscillator differential equation
    loss2 = (1e-4)*torch.mean(physics**2)
    
    # backpropagate joint loss
    loss = loss1 + loss2# add two loss terms together
    loss.backward()
    optimizer.step()
    
    
    # plot the result as training progresses
    if (i+1) % 150 == 0: 
        
        yh = model(x).detach()
        xp = x_physics.detach()
        
        f2=plot_result(x,y,x_data,y_data,yh,xp)
        if (i+1)%1000 == 0:
            st.pyplot(f2)
        else:
            plt.close("all")
