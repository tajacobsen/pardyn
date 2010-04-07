"""Define some pickle helper classes"""
import os
import numpy as np
import cPickle as pickle
from gpaw.mpi import world, MASTER

class PickleTraj:
    """Class for reading the pickle file dumped by ParticleOnPES"""
    def __init__(self, fname):
        self.fname = fname
        self.data = {}
        self.is_read = False

    def read(self):
        """Read the file and sort stuff"""
        fd = open(self.fname, 'rb')
        dummy_rawdata = pickle.load(fd) # Read first line
        good = True
        while good:
            try:
                rawdata = pickle.load(fd)
                for key in rawdata.keys():
                    if self.data.has_key(key):
                        self.data[key].append(rawdata[key])
                    else:
                        self.data[key] = [rawdata[key]]
            except EOFError:
                good = False

        for key in self.data.keys():
            self.data[key] = np.asarray(self.data[key])

        self.is_read = True

class Writer:
    def __init__(self, particle, dyn, name, fmt='-%04i.pickle', directory=None):
        self.dyn = dyn
        self.particle = particle
        if directory is None:
            directory = ''
        else:
            if not directory.endswith('/'):
                directory += '/'
            if not os.path.isdir(directory):
                if world.rank == MASTER:
                    os.mkdir(directory)
            world.barrier()
        self.fmt = directory + name + fmt
    def write(self):
        self.particle.write(self.fmt % self.dyn.nsteps)

def normalize(wf):
    return wf / np.sqrt(np.dot(wf.reshape(-1).conj(), wf.reshape(-1)))

def get_overlaps(wfs0, wfs1):
    overlaps = np.empty((len(wfs0), len(wfs1)))
    for i in range(len(wfs0)):
        for j in range(len(wfs1)):
            overlaps[i, j] = np.dot(wfs0[i].conj(), wfs1[j])

    return overlaps

def get_random_number():
    if world.rank == MASTER:
        rand = np.random.rand(1)
    else:
        rand = np.empty(1)
    world.broadcast(rand, MASTER)
    rand = rand[0]
    return rand

def spline(x, f, df=None):
    n = len(x) - 1  # Number of knots
    h = np.diff(x)     # n-vector with knot spacings
    v = np.diff(f)/h   # n-vector with derived differences

    A = np.zeros( (n+1, n+1) )
    r = np.zeros( (n+1, 1) )

    for i in range(0, n-1):
        A[i+1, i:i+3] = np.array( [h[i+1], 2*(h[i] + h[i+1]), h[i]] )
        r[i+1] = 3*( h[i+1]*v[i] + h[i]*v[i+1] )

    if df is not None:
        A[0, 0] = 1
        r[0] = df[0]
        A[n, n] = 1
        r[n] = df[1]
    else:
        A[0, 0:2] = np.array([2, 1])
        r[0] = 3*v[0]
        A[n, n-1:n+1] = np.array([1, 2])
        r[n] = 3*v[n-1]

    ds = np.linalg.solve(A, r)
    p = np.zeros( (n, 4) )
    for i in range(n):
        p[i,0] = f[i]
        p[i,1] = h[i]*ds[i]
        p[i,2] = 3*( f[i+1] - f[i]) - h[i]*(2*ds[i] + ds[i+1])
        p[i,3] = 2*( f[i] - f[i+1]) + h[i]*(ds[i] + ds[i+1])

    return p

def splineeval(x, p, t):
    s = np.zeros( t.shape )
    for i in range(len(x) - 1):
        k = np.nonzero( (x[i] <= t) & (t <= x[i+1]) )
        if len(k):
            u = (t[k] - x[i])/(x[i+1] - x[i])
            s[k] = p[i,0] + u*(p[i,1] + u*(p[i,2] + u*p[i,3]))
    return s
