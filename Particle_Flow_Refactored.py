# author: cmelgreen
# date: 9/12/20
#
# written as answer to/based on vishal's codereview question at:
# https://codereview.stackexchange.com/questions/249239/modelling-of-partcle-flow-under-electric-field
#
# particle dispersion model from: 
# electrostatic forces alter particle size distributions in atmospheric dust
# https://acp.copernicus.org/articles/20/3181/2020/
#
# result:
# ~300x speedup from origiinal code

import numpy as np
import scipy.stats as stats
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Create Particle Arrays and Environemntal Consts
def NewParticles(mean, sd, alpha, beta, n):
    # Truncate Norm Dist at 0
    diameter = stats.truncnorm((0 - mean) / sd, np.inf, loc=mean, scale=sd).rvs(n)
    charge = np.random.normal(0, alpha, n)

    # Sample from Norm Dist then rescale SD = Beta/D^2
    velocity_0 = np.random.normal(size=n)*(beta/(diameter**2))

    return namedtuple('Particles', 'd ec v0')(diameter, charge, velocity_0)

def NewEnvironment(rf, na, rp, g, el, vh):
    return namedtuple('Environment', 'rf na rp g vh el')(rf, na, rp, g, vh, el)


# Setup DiffEq for Particle motion in given Environment
# Vectorize as much as possible for performance
class ParticleFlow:
    def __init__(self, particles, environment):
        self.p = particles
        self.env = environment

        # Create const horizontal velocity array for vectorization
        # Source model ignores horizontal forces
        self.vh = (np.zeros_like(particles.d) * self.env.vh).T

        # Vertical forces independent of velocity are constant and only need calcuated once
        self.fg = (np.pi * (self.p.d ** 3) * self.env.rp * self.env.g) / 6
        self.fe =  np.pi * (self.p.d ** 2) * self.p.ec * self.env.el
        
    def __call__(self, v):
        # calculate norm of vh and v
        norm_vh_v = np.linalg.norm(np.stack((self.vh, v.T), axis=1), axis=1)

        # Drag Coefficient
        def cd():
            re = (self.env.rf * norm_vh_v * self.p.d) / self.env.na
            return (24 / re) * (1 + 0.15 * (re ** 0.681)) + 0.407 / (1 + (8710 / re))
            
        # Drag
        def fd():
            return - (1 / 8) * (cd() * self.env.rf * np.pi * (self.p.d ** 2) * (v ** 2)) * np.reciprocal(norm_vh_v)
            
        # Return Total Forces
        return (6 / (np.pi * (self.p.d ** 3) * self.env.rp)) * (self.fg + self.fe + fd())
        

def ExplicitEuler(func):
    # Wrap ParticleFlow for Integration
    def f(v, dt):
        return v + func(v) * dt
    return f

def Integrate(method, v0, tMin, tMax, n):
    dt = (tMax - tMin)/(n-1)
    v = np.zeros((int((tMax - tMin) / dt), n))
    v[0] = v0

    # Calculate Integral
    for i in range(1, len(v)):
        v[i] = method(v[i - 1], dt)

    return v

# Used solved velocities to create array of Particle positions over time
def PositionOverTime(vv, vh, tMin, tMax):
    p = np.zeros((*vv.shape, 2))
    dt = (tMax - tMin) / (len(vv) - 1)

    # P = P + V * dT 
    for i in range(1, len(p)):
        p[i, :, 0] = p[i-1, :, 0] + vh*dt
        p[i, :, 1] = p[i-1, :, 1] + vv[i-1]*dt

    return p


def GeneratePoints():
    # Bunch of constants that could be passed in as tuples but hardcoded for example
    (rf, na, rp, g, el, vh) = (1.18, 1.5*1e-05, 2600, 9.8, 1e05, 4.95)
    (da, sd, alpha, beta, nop) = (132 * 1e-06, 45 * 1e-06, 1.4 * 1e-06, 11000 * 1e-12, 100)
    (tmin, tmax) = (1e-06, 18e-03)

    # Time Setup and Euler Method
    start = time.time()

    # Create Environment and Particle Arrays
    e = NewEnvironment(rf, na, rp, g, el, vh)
    p = NewParticles(da, sd, alpha, beta, nop)

    # Run Euler Method
    vv = Integrate(ExplicitEuler(ParticleFlow(p, e)), p.v0, tmin, tmax, nop)

    end = time.time()
    print("Ran in: ", end - start)

    return PositionOverTime(vv, np.ones(nop) * e.vh, tmin, tmax), p.d, p.ec

# Plot the points with diameter and charge visualized
def Plot(p, d, c):
    frames = 99
    interval = 65
    scale = 1.7e6
    alpha = .67

    def init():
        return scatter,

    def animate(i):
        X = np.c_[p[i, :, 0], p[i, :, 1]]
        scatter.set_offsets(X)
        return scatter,

    # Scale marker size array once instead of recalculaing for each frame
    s = (d * scale)
    
    # Setup figure and axes
    fig = plt.figure()
    ax = plt.axes(
        xlim=(p[0, :, 0].min(), p[-1, :, 0].max()),
        ylim=(p[-1, :, 1].min(), p[-1, :, 1].max())
    )
    scatter = ax.scatter(p[1, :, 0], p[1, :, 1], c=c, cmap='bwr', s=s, alpha=alpha)

    # Start animation and display
    _ = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=True, repeat=False)
    plt.show()

# Run
if __name__ == "__main__":
    Plot(*GeneratePoints())