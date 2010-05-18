#!/usr/bin/env python
import os
import sys
import time

import numpy as np
from ase.parallel import paropen, rank

class Verlet:
    """Verlet integrator for molecular dynamics"""
    def __init__(self, particle, dt,
                 trajectory=None,
                 logfile=None,
                ):
        """Parameters:

        particle: ParticleOnPES (or subclass)
            Particle to integrate the equations of motion for.
        dt: float
            Time step
        trajectory: str
            File storing stuff stuff during calculation
        logfile: str
            Logfile
        """
        self.particle = particle
        self.dt = dt
        self.nsteps = 0
        self.observers = []

        self.xold = None

        if rank != 0:
            logfile = None
        elif isinstance(logfile, str):
            if logfile == '-':
                logfile = sys.stdout
            else:
                if os.path.isfile(logfile):
                    os.rename(logfile, logfile + '.old')
                logfile = paropen(logfile, 'w')
        self.logfile = logfile

        if isinstance(trajectory, str):
            trajectory = paropen(trajectory, 'w')
        self.trajectory = trajectory

    def step(self):
        particle = self.particle
        dt = self.dt
        x = particle.get_position()
        xold = self.xold
        v = particle.get_velocity()
        a = particle.get_acceleration()

        # First timestep
        # x(dt) = x(0) + v(0)dt + 0.5*a(0)dt**2 
        # Other timesteps
        # x(t+dt) = 2*x(t) - x(t-dt) + a(t)dt**2
        if self.nsteps == 0:
            xnew = x + v * dt + 0.5 * a * dt**2
        else:
            xnew = 2 * x - xold + a * dt**2

        # v(t) =  0.5(x(t+dt)-x(t-dt))/dt   # Midtpoint formula
        # v(t+dt) = (x(t+dt)-x(t))/dt       # Backward gradient
        # v(t) = (x(t)-x(t-dt))/dt
        vnew = (xnew - x) / dt

        particle.set_position(xnew)
        particle.set_velocity(vnew)
        self.xold = x

    def run(self, steps=100):
        if self.nsteps == 0:
            self.log()
            self.call_observers()

        step = 0
        while step < steps:
            self.step()
            self.nsteps += 1
            step += 1
            self.log()
            self.call_observers()

    def log(self):
        if self.logfile is not None:
            T = time.localtime()
            self.logfile.write('%5i  %02d:%02d:%02d %10.4f\n' %
                               (self.nsteps, T[3], T[4], T[5], self.particle.get_energy()))
            self.logfile.flush()
        if self.trajectory is not None:
            poslist = []
            for p in self.particle.get_position():
                poslist.append('%15.10f' % p)
            poslist.append('%15.10f' % self.particle.get_energy())
            self.trajectory.write('\t'.join(poslist) + '\n')
            self.trajectory.flush()

    def attach(self, function, interval=1, *args, **kwargs):
        """Attach callback function.

        At every *interval* steps, call *function* with arguments
        *args* and keyword arguments *kwargs*."""

        if not callable(function):
            function = function.write
        self.observers.append((function, interval, args, kwargs))

    def call_observers(self):
        for function, interval, args, kwargs in self.observers:
            if self.nsteps % interval == 0:
                function(*args, **kwargs)
