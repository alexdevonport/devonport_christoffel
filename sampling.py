import numpy as np 
from multiprocessing import Pool, cpu_count
from scipy.integrate import odeint, solve_ivp
from numpy.random import default_rng

def chaotic_oscillator(y,t, alpha = 0.05, omega = 1.3, gamma = 0.4):
    dydt = [y[1], -alpha*y[1] + y[0] - y[0]**3+gamma*np.cos(omega*t)]
    return dydt


def planar_quadrotor(y,t,u1, u2, g=9.81, K=0.89/1.4, d0=70, d1=17, n0=55):
    x, xdot, h, hdot, theta, thetadot = y
    dydt = [xdot, u1*K*np.sin(theta), hdot, -g+u1*K*np.cos(theta), thetadot, -d0*theta-d1*thetadot+n0*u2]
    return dydt

def traffic(y,t,d=0):
    v = 0.5            # free-flow speed, in links/period
    w = 1/6            # congestion-wave speed, in links/period
    c = 40             # capacity (max downstream flow), in vehicles/period
    xbar = 320         # max occupancy when jammed, in vehicles
    b = 1              # fraction of vehicle staying on the network after each link
    T = 30             # time step for the continuous-time model
    nx = np.size(y)
    dy = np.zeros(np.shape(y))
    dy[0] = 1/T*(d - min(c , v*y[0] , 2*w*(xbar-y[1])))
    for i in range(1,nx-1):
        dy[i] = 1/T*(b*min(c , v*y[i-1] , w*(xbar-y[i]))-min(c , v*y[i] , w/b*(xbar-y[i+1])))
    dy[-1] = 1/T*(b*min(c , v*y[-2] , w*(xbar-y[-1])) - min(c , v*y[-1]))
    return dy

def make_sample_n(sample_fn, parallel=True, pool=None):
    def sample_n(n, pool=pool):
        if parallel:
            if pool is None:
                #print(r'Using {} CPUs'.format(cpu_count()))
                p = Pool(cpu_count())
            else:
                p = pool
            return np.array(list(p.map(sample_fn, np.arange(n))))
        else:
            return np.array([sample_fn() for i in range(n)])
    return sample_n

def sample_oscillator(x=None):
    t = np.linspace(0, 100, 1001)
    rng = default_rng()
    y0 = np.array([rng.uniform(0.95, 1.05), rng.uniform(-0.05, 0.05)])
    sol = odeint(chaotic_oscillator, y0, t)
    succ = sol[-1]
    return np.flip(succ)

def sample_traffic(x=None, nx=6):
    rng = default_rng()
    t = np.linspace(0, 100, 10001)
    y0 = 100 * np.ones(nx) + 100 * rng.uniform(0,1,size=(nx,))
    sol = odeint(traffic, y0, t)
    return sol[-1]

def sample_traffic_end(x=None, nx=6):
    rng = default_rng()
    t = np.linspace(0, 100, 10001)
    y0 = 100 * np.ones(nx) + 100 * rng.uniform(0,1,size=(nx,))
    sol = odeint(traffic, y0, t)
    successor = sol[-1]
    return successor[-2:]

def sample_quadrotor(x=None):
    g=9.81
    K=0.89/1.4
    rng = default_rng()
    ru = rng.uniform
    t = np.linspace(0, 5, 5001)
    y0 = np.array([ru(-1.7, 1.7), ru(-0.8, 0.8), 
                   ru(0.3, 2.0),  ru(-1.0, 1.0),
                   ru(-np.pi/12, np.pi/12), ru(-np.pi/2, np.pi/2)])
    sol = odeint(planar_quadrotor, y0, t, args=(ru(-1.5+g/K, 1.5+g/K), ru(-np.pi/4, np.pi/4)))
    return sol[-1]

def sample_quadrotor_xh(x=None):
    g=9.81
    K=0.89/1.4
    rng = default_rng()
    ru = rng.uniform
    t = np.linspace(0, 5, 5001)
    y0 = np.array([ru(-1.7, 1.7), ru(-0.8, 0.8), 
                   ru(0.3, 2.0),  ru(-1.0, 1.0),
                   ru(-np.pi/12, np.pi/12), ru(-np.pi/2, np.pi/2)])
    sol = odeint(planar_quadrotor, y0, t, args=(ru(-1.5+g/K, 1.5+g/K), ru(-np.pi/4, np.pi/4)))
    successor = sol[-1]

    return np.array([successor[0],successor[2]])
    
sample_oscillator_n = make_sample_n(sample_oscillator)
sample_quadrotor_n = make_sample_n(sample_quadrotor)
sample_quadrotor_xh_n = make_sample_n(sample_quadrotor_xh)
sample_traffic_n = make_sample_n(sample_traffic)
sample_traffic_end_n = make_sample_n(sample_traffic_end)
