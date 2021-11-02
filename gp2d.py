import numpy as np
import matplotlib.pyplot as plt

# - generate x and y grid
# - generate covariance matrix for RVs at each grid point
# - generate instance
# - plot it

def grid2cov(xgrid, ygrid, kfun):
    nx = np.size(xgrid)
    ny = np.size(ygrid)
    cmat = np.zeros((nx*ny,nx*ny))
    for (i1, x1) in enumerate(xgrid):
        for (j1, y1) in enumerate(ygrid):
            cmat_idx_1 = j1 + ny*i1
            for (i2, x2) in enumerate(xgrid):
                for (j2, y2) in enumerate(ygrid):
                    cmat_idx_2 = j2 + ny*i2
                    z1 = np.array([x1,y1])
                    z2 = np.array([x2,y2])
                    cmat[cmat_idx_1,cmat_idx_2] = kfun(z1,z2)
    return cmat

def kfun(z1,z2):
    #return np.exp(-np.linalg.norm(z1-z2)**2)
    return np.exp(np.dot(z1,z2))

gridsize=25
gridwidth = 5
xgrid = np.linspace(-gridwidth,gridwidth,gridsize)
ygrid = np.linspace(-gridwidth,gridwidth,gridsize)
nx = np.size(xgrid)
ny = np.size(ygrid)

def gpgrid(xgrid, ygrid, cmat_chol):
    nx = np.size(xgrid)
    ny = np.size(ygrid)
    cmat = grid2cov(xgrid, ygrid, kfun)
    cmat_chol = np.linalg.cholesky(cmat + 1e-9*np.eye(nx*ny))
    standard_normals = np.random.normal(size=nx*ny)
    gp_grid_values = np.matmul(cmat_chol,standard_normals)
    return np.resize(gp_grid_values, (nx,ny))


cmat = grid2cov(xgrid, ygrid, kfun)
cmat_chol = np.linalg.cholesky(cmat + 1e-9*np.eye(nx*ny))


for w in range(10):
    print(w)
    plt.clf()
    standard_normals = np.random.normal(size=nx*ny)
    gp_grid_values = np.matmul(cmat_chol,standard_normals)
    gp_grid_values = np.resize(gp_grid_values, (nx,ny))
    Z = np.power(gp_grid_values, 2)
    X,Y = np.meshgrid(xgrid,ygrid)
    #plt.contour(X, Y, gpg, levels=[0])
    #plt.contour(X, Y, gp_grid_values, levels=[1])
    plt.contour(X, Y, Z, levels=[np.exp(10)])
    plt.savefig('run{:d}.png'.format(w))
