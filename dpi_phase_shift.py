import numpy as np
import matplotlib.pyplot as plt
from math import pi

def lam(x,y,z):
    return x**2. + y**2. + z**2. - 2.*(x*y+y*z+z*x)

class UChiPT_Potential:
    def __init__(self, a, h24, h35, h4p, h5p):
        self.a = a
        self.lam = 1000.
        self.h0 = 0.01368400
        self.h1 = 0.42
        self.fpi = 92.21
        self.h24 = h24
        self.h35 = h35
        self.h4p = h4p
        self.h5p = h5p
        self.mpi = (134.98 + 2. * 139.57) / 3.
        self.mK = (493.68 + 497.61) / 2.
        self.meta = 547.862
        self.mD = (1869.66 + 1864.84) / 2.
        self.mDs = 1968.300
        self.mDav = (self.mD + self.mDs) / 2.
        self.delta_onehalf = np.vectorize(self._delta_onehalf)

        # Propagator matrix
    def G(self,s,m1,m2):
        sig = np.sqrt(lam(s,m1**2,m2**2))
        l1 = np.log(s - m1**2 + m2**2 + sig)
        l2 = np.log(-s - m1**2 + m2**2 + sig)
        l3 = np.log(s + m1**2 - m2**2 + sig)
        l4 = np.log(-s + m1**2 - m2**2 + sig)
        return 1./16./pi**2.*(self.a + np.log(m1*m2/self.lam**2.) + (m1**2.-m2**2.)/2./s*np.log(m1**2./m2**2.) + sig/2./s*(l1-l2+l3-l4))

    def GMat_0_12(self,s):
        return np.diag(np.array([self.G(s,self.mD,self.mpi),self.G(s,self.mD,self.meta),self.G(s,self.mDs,self.mK)]))

    # Potential matrix
    def VMat_0_12(self,s):
        MH = np.array([self.mD,self.mD,self.mDs])
        ML = np.array([self.mpi,self.meta,self.mK])
        CLO = np.array([[-2.,0.,-np.sqrt(6.)/2.],[0.,0.,-np.sqrt(6.)/2.],[-np.sqrt(6.)/2.,-np.sqrt(6.)/2.,-1.]])
        C0 = np.diag([self.mpi**2.,self.meta**2.,self.mK**2.])
        C1 = np.array([[-self.mpi**2.,-self.mpi**2.,-np.sqrt(6.)/4.*(self.mpi**2.+self.mK**2.)],[-self.mpi**2.,-self.mpi**2./3.,np.sqrt(6.)*(5.*self.mK**2.-3.*self.mpi**2.)/12.],[-np.sqrt(6.)/4.*(self.mpi**2.+self.mK**2.),np.sqrt(6.)*(5.*self.mK**2.-3.*self.mpi**2.)/12.,-self.mK**2.]])
        C35 = np.array([[1.,1.,np.sqrt(6.)/2.],[1.,1./3.,-1./np.sqrt(6.)],[np.sqrt(6.)/2.,-1./np.sqrt(6.),1.]])
        E_L = (s + ML**2. - MH**2.)/2./np.sqrt(s)
        E_H = (s + MH**2. - ML**2.)/2./np.sqrt(s)
        p2sq = lam(s,ML**2.,MH**2.)/4./s
   
        p24_Sw = np.outer(E_L,E_L)
        p14p23_Sw = np.outer(E_L*E_H,E_L*E_H) + np.outer(p2sq,p2sq)/3.
        p12p34_Sw = np.outer(s-MH**2.-ML**2.,s-MH**2.-ML**2.)/4.
        u_Sw = np.column_stack((MH**2.,MH**2.,MH**2.)) + np.vstack((ML**2.,ML**2.,ML**2.)) - 2.*np.outer(E_H,E_L)

        H24_sw = 2.*self.h24*p24_Sw + self.h4p/self.mDav**2.*(p14p23_Sw + p12p34_Sw - 2.*self.mDav**2.*p24_Sw)
        H35_sw = self.h35*p24_Sw + self.h5p/self.mDav**2.*(p14p23_Sw + p12p34_Sw - 2.*self.mDav**2.*p24_Sw)

        # Note, these are not matrix multiplications, but element-wise!!!
        V = 1./self.fpi**2.*(CLO*(s-u_Sw)/4. - 4.*C0*self.h0 + 2.*C1*self.h1 - 2.*np.eye(3)*H24_sw + 2.*C35*H35_sw)

        return V

    def V_0_32(self,s):
        MH = self.mD
        ML = self.mpi
        CLO = 1
        C0 = self.mpi**2.
        C1 = - self.mpi**2.
        C35 = 1
        E_L = (s + ML**2. - MH**2.)/2./np.sqrt(s)
        E_H = (s + MH**2. - ML**2.)/2./np.sqrt(s)
        p2sq = lam(s,ML**2.,MH**2.)/4./s

        p24_Sw = E_L ** 2
        p14p23_Sw = E_L ** 2 * E_H ** 2 + p2sq**2/3.
        p12p34_Sw = (s-MH**2.-ML**2.)**2/4.
        u_Sw = MH**2 + ML**2 - 2 * E_H * E_L
        H24_sw = 2.*self.h24*p24_Sw + self.h4p/self.mDav**2.*(p14p23_Sw + p12p34_Sw - 2.*self.mDav**2.*p24_Sw)
        H35_sw = self.h35*p24_Sw + self.h5p/self.mDav**2.*(p14p23_Sw + p12p34_Sw - 2.*self.mDav**2.*p24_Sw)
        V = 1./self.fpi**2.*(CLO*(s-u_Sw)/4. - 4.*C0*self.h0 + 2.*C1*self.h1 - 2.*H24_sw + 2.*C35*H35_sw)

        return V

    # T-matrix
    def TMat_0_12(self,s):
        G = self.GMat_0_12(s)
        V = self.VMat_0_12(s)
        den = np.eye(3) - V@G
        return np.linalg.inv(den)@V

    def T_0_32(self,s):
        G = self.G(s,self.mD,self.mpi)
        V = self.V_0_32(s)
        den = 1 - V*G
        return V/den

    # T-matrix projected onto the D pi channe;
    def _TMat_proj_Dpi(self,s):
        return self.TMat_0_12(s * 1000.**2 + 0j)[0]@np.array([np.sqrt(3./2.),np.sqrt(1./6.),1.])

    def _delta_onehalf(self,E):
        return np.angle(self._TMat_proj_Dpi(E**2)) + np.pi

    def delta_threehalf(self,E):
        return np.angle(self.T_0_32(E**2 * 1000.**2 + 0j))
