"""Define a Particle on a PES

Allow for particles on a single PES or a surface hopping particle
"""
import numpy as np
import cPickle as pickle
from ase import units
from gpaw.mpi import world, MASTER
from pardyn.tools import normalize, get_overlaps, get_random_number

# Ideas for improvements:
#   ND functionality

hbar = units._hbar * units.second * units.kJ * 1e-3

def get_effective_mass_1d(particle, eps=1e-3):
    """Return current effective mass"""
    atomsf = particle.atoms_constructor(particle.x + eps)
    xposf = atomsf.get_positions().reshape(-1)

    atomsb = particle.atoms_constructor(particle.x - eps)
    xposb = atomsb.get_positions().reshape(-1)

    dxdlambda = (xposf - xposb) / (2 * eps)

    masses = np.zeros_like(xposf)
    for i in range(len(atomsf)):
        masses[(3 * i):(3 * (i + 1))] = atomsf.get_masses()[i]

    m = np.sum(masses * dxdlambda**2)
    return m

def get_acceleration_1d(particle, **kwargs):
    """Return current acceleration"""
    gradV = particle.pes.get_derivative(particle.x)
    m = get_effective_mass_1d(particle, **kwargs)

    a = - gradV / m
    return a

def get_effective_mass_2d(particle, eps=1e-3):
    """Return current effective mass"""
    atomsf0 = particle.atoms_constructor(particle.x + np.array([eps, 0.0]))
    xposf0 = atomsf0.get_positions().reshape(-1)

    atomsb0 = particle.atoms_constructor(particle.x - np.array([eps, 0.0]))
    xposb0 = atomsb0.get_positions().reshape(-1)

    dxdlambda0 = (xposf0 - xposb0) / (2 * eps)

    atomsf1 = particle.atoms_constructor(particle.x + np.array([0.0, eps]))
    xposf1 = atomsf1.get_positions().reshape(-1)

    atomsb1 = particle.atoms_constructor(particle.x - np.array([0.0, eps]))
    xposb1 = atomsb1.get_positions().reshape(-1)

    dxdlambda1 = (xposf1 - xposb1) / (2 * eps)

    masses = np.zeros_like(xposf1)
    for i in range(len(atomsf1)):
        masses[(3 * i):(3 * (i + 1))] = atomsf1.get_masses()[i]

    m = np.zeros((2, 2))
    m[0, 0] = np.sum(masses * dxdlambda0**2)
    m[0, 1] = np.sum(masses * dxdlambda0 * dxdlambda1)
    m[1, 0] = np.sum(masses * dxdlambda1 * dxdlambda0)
    m[1, 1] = np.sum(masses * dxdlambda1**2)

    return m

def get_acceleration_2d(particle, **kwargs):
    """Return current acceleration"""
    gradV = np.zeros_like(particle.x)
    gradV[0] = particle.pes.get_derivative(particle.x, dx=[1, 0])
    gradV[1] = particle.pes.get_derivative(particle.x, dx=[0, 1])

    m = get_effective_mass_2d(particle, **kwargs)

    a = - np.linalg.solve(m, gradV) # - m**(-1) * gradV
    return a

class ParticleOnPES:
    """A particle on an adiabatic PES"""
    def __init__(self, x0, pes, atoms_constructor, v0=None, obs=None):
        """Parameters:

        x0: array
            Initial position
        pes: PES
            Potential Energy Surface particle is moving on
        atoms_constructor: function
            A function returning an atoms object as a function of x
        v0: array
            Initial velocity
        obs: string/file
            Filename/file to dump information in
        """
        self.x = x0
        if v0 is None:
            v0 = np.zeros_like(x0)
        self.v = v0
        self.pes = pes
        self.atoms_constructor = atoms_constructor

        if isinstance(obs, str):
            obs = open(obs, 'wb')
        self.obs = obs
        self.data = {}

    def get_position(self):
        """Return current position"""
        return self.x.copy()

    def set_position(self, x):
        """Set position"""
        # Implicity copy
        self.x = np.array(x)

    def get_velocity(self):
        """Return current velocity"""
        return self.v.copy()

    def set_velocity(self, v):
        """Set velocity"""
        # Implicity copy
        self.v = np.array(v)

    def get_acceleration(self, **kwargs):
        """Return current acceleration"""
        if self.pes.dim == 1:
            return get_acceleration_1d(self, **kwargs)
        elif self.pes.dim == 2:
            return get_acceleration_2d(self, **kwargs)

    def get_energy(self):
        """Return energy at current position"""
        return self.pes.get_value(self.x)

    def call_obs(self):
        """Write information to observer file"""
        if self.obs is not None:
            self.data['x'] = self.x.copy()
            self.data['v'] = self.v.copy()
            self.data['E'] = self.get_energy()
            if world.rank == MASTER:
                pickle.dump(self.data, self.obs, protocol=-1)
                self.obs.flush()

class MultiPESParticle(ParticleOnPES):
    """Particle allowed to move on multiple PESs"""
    def __init__(self, x0, peslist, nlist, atoms_constructor, calc,
                 surf=1, v0=None, c0=None, obs=None, dt=units.fs,
                 newint=True, switch=True):
        """Parameters:

        peslist: list of PES
            Surfaces to move on
        nlist: list of int
            Band indices of surfaces (to calculate overlaps)
        calc: GPAW
            Calculator to calculate overlaps
        surf: int
            Surface to start on
        c0: list of float
            Initial expansion coefficients
        dt: float
            Timestep (to calculate time derivatives)
        newint: bool
            Use new integration scheme
        switch: bool
            Switch surfaces to follow geometry

        See ParticleOnPES for:
            x0, atoms_constructor, v0, obs
        """
        if len(nlist) != len(peslist):
            raise ValueError, "Shape mismatch"

        Npes = len(peslist)

        if surf not in range(Npes):
            raise IndexError, "Specified surface not available"

        pes = peslist[surf]
        ParticleOnPES.__init__(self, x0, pes, atoms_constructor,
                               v0=v0, obs=obs)

        self.peslist = peslist
        self.nlist = nlist
        self.calc = calc
        self.surf = surf
        self.dt = dt
        self.newint = newint
        self.switch = switch

        if c0 is None:
            c0 = np.zeros((Npes,))
            c0[surf] = 1.0
        self.c = c0

        self.initialized = False
        self.wfs = [None for pes in peslist]

        # These variables accounts for surfaces switching around
        self.idx = np.arange(len(peslist), dtype=int)
        self.permutation_matrix = np.eye(len(peslist), dtype=int)

    def initialize(self):
        """Initialize

        This is here so no calculations is done in the regular constructor
        """
        if not self.initialized:
            self.wfs = self._get_wfs_raw()
            self.initialized = True

    def _get_wfs_raw(self):
        """Calculate wavefunctions for all surfaces

        Unsorted and phase not fixed"""
        atoms = self.atoms_constructor(self.x)
        atoms.set_calculator(self.calc)
        atoms.get_potential_energy()
        wfs = []
        for n in self.nlist:
            wf = self.calc.get_pseudo_wave_function(band=n).reshape(-1)
            wf = normalize(wf)
            wfs.append(wf)
        return wfs

    def _get_wfs(self):
        """Calculate wavefunctions for all surfaces"""
        # Get wavefunctions and sort them
        wfs = self._get_wfs_raw()
        wfs = [wfs[i] for i in self.idx]

        # Fix phase and ordering 
        wfsold = self.wfs
        S = get_overlaps(wfs, wfsold)

        if self.switch:
            # See if surfaces have switched places
            this_permutation = np.abs(np.array(S.round(), dtype=int))
            if world.rank == MASTER:
                print '# DEBUG ######################'
                print 'S:', S.reshape(-1)
                print 'perm:', this_permutation.reshape(-1)
            assert np.abs(np.linalg.det(this_permutation)) == 1
            # Save the redordered index for later use 
            self.idx = np.dot(this_permutation,
                         self.idx)
            self.data['idx'] = self.idx.copy()
            self.permutation_matrix = np.dot(this_permutation,
                                             self.permutation_matrix)
            self.data['permutation_matrix'] = self.permutation_matrix.copy()

            # Re-sort if necessary
            this_idx = np.dot(this_permutation,
                              np.arange(len(self.idx), dtype=int))
            wfs = [wfs[i] for i in this_idx]
            S_permuted = np.dot(this_permutation, S)
            S = get_overlaps(wfs, wfsold)
            assert np.max(np.abs(S - S_permuted)) < 1e-3

        # Change sign of wavefunctions if overlaps with itself is negative
        for i in range(S.shape[0]):
            if S[i, i] < 0.0:
                wfs[i] *= -1
                S = get_overlaps(wfs, wfsold)
        self.data['S'] = S.copy()

        return wfs

    def set_position(self, x):
        """Set position"""
        self.initialize()
        ParticleOnPES.set_position(self, x)

        wfsold = self.wfs
        wfs = self._get_wfs()

        # <psi_i|d/dt|psi_j>
        ddt_wfs = []
        for i in range(len(wfs)):
            ddt_wf = (wfs[i] - wfsold[i]) / self.dt
            ddt_wfs.append(ddt_wf)
        V = get_overlaps(wfs, ddt_wfs)
        self.data['V'] = V.copy()

        # Energies
        Epes = []
        for i in self.idx:
            pes = self.peslist[i]
            Epes.append(pes.get_value(self.x))
        self.data['Epes'] = Epes

        # Hamiltonian
        H = np.diag(Epes) - 1.0j * V
        self.data['H'] = H.copy()

        # Propagate expansion coefficients
        cold = self.c.copy()

        if not self.newint:
            ### Old integration
            c = -1.0j / hbar * np.dot(H, cold) * self.dt + cold
        else:
            ### New integration
            #   d             d
            # i -- c = H c => -- c = -i H c
            #   dt            dt
            # 
            # e_i, v_i is eigenvals/vectors of (-i H)
            #
            # then
            #
            # c(t) = k_1 v_1 exp(e_2 t) + k_2 v_2 exp(e_2 t)
            #
            # b.c: v k = c => k = v\c
            if world.rank == MASTER:
                print 'New integration'
            e, v = np.linalg.eig(-1.0j * H / hbar)
            k = np.linalg.solve(v, cold)
            c = k[0] * v[:, 0] * np.exp(e[0] * self.dt) + \
                k[1] * v[:, 1] * np.exp(e[1] * self.dt)
        #
        self.data['c'] = c.copy()

        c /= np.linalg.norm(c)
        self.data['cnorm'] = c.copy()
        self.c = c.copy()

        # Save wavefunctions for later
        self.wfs = wfs

        # Stuff done for logging purposes
        self.data['nlist'] = self.nlist[:]
        self.data['eigenvalues'] = \
                [self.calc.get_eigenvalues()[n] for n in self.nlist]
        self.data['potential_energy'] = self.calc.get_potential_energy()

class SurfaceHoppingParticle(MultiPESParticle):
    """Surface hopping dynamics particle"""
    def __init__(self, *args, **kwargs):
        self.newint = kwargs.pop('newint', True)
        self.collapse = kwargs.pop('collapse', False)
        MultiPESParticle.__init__(self, *args, **kwargs)

    def set_position(self, x):
        """Set position"""
        MultiPESParticle.set_position(self, x)

        # Populations
        c = self.c
        P = np.real(c.conj() * c)
        self.data['P'] = P.copy()

        assert np.abs(1.0 - P.sum()) < 1e-5

        # Determine if we should hop
        rand = get_random_number()
        self.data['rand'] = rand

        # Choose surface like this    
        #         
        # 0|  p0   | p1 | p2  |  p3  |...|1
        #            ^
        #           rand
        for i, p in enumerate(P.cumsum()):
            if rand < p:
                surf = i
                if self.collapse and surf != self.surf:
                    self.c = np.zeros((len(self.peslist),))
                    self.c[surf] = 1.0
                self.surf = surf
                self.pes = self.peslist[self.idx[i]]
                break
        self.data['surf'] = self.surf

    def write(self, f):
        f = open(f, 'wb')
        data = (self.x, self.v, self.c, self.wfs, self.surf)
        pickle.dump(data, f, protocol=-1)
        f.close()

    def read(self, f):
        f = open(f, 'rb')
        (self.x, self.v, self.c, self.wfs, self.surf) = \
            pickle.load(f)
        f.close()
        self.initialized = True

class EhrenfestParticle(MultiPESParticle):
    """Ehrenfest dynamics particle"""
    def __init__(self, *args, **kwargs):
        MultiPESParticle.__init__(self, *args, **kwargs)

    def get_energy(self):
        """Return energy at current position"""
        c = self.c
        P = np.real(c.conj() * c)
        E = 0.0

        for i, p in enumerate(P):
            pes = self.peslist[self.idx[i]]
            E += p * pes.get_value(self.x)

        return E

    def get_acceleration(self, **kwargs):
        """Return current acceleration"""
        c = self.c
        P = np.real(c.conj() * c)
        A = np.zeros((len(self.x),))

        for i, p in enumerate(P):
            self.pes = self.peslist[self.idx[i]]
            if self.pes.dim == 1:
                A += p * get_acceleration_1d(self, **kwargs)
            elif self.pes.dim == 2:
                A += p * get_acceleration_2d(self, **kwargs)

        return A

    def write(self, f):
        f = open(f, 'wb')
        data = (self.x, self.v, self.c, self.wfs)
        pickle.dump(data, f, protocol=-1)
        f.close()

    def read(self, f):
        f = open(f, 'rb')
        (self.x, self.v, self.c, self.wfs) = \
            pickle.load(f)
        f.close()
        self.initialized = True
