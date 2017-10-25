import numpy as np
import scipy.linalg as la
from scipy.misc import comb

# specific

def rot2(*args):

    if len(args) == 1:
        theta = args[0]

    elif len(args) == 2:
        v = args[0]
        theta = args[1]

    else:
        assert len(args) != 1 or len(args) !=2, 'must have one or two arguments'

    ct = np.cos(theta)
    st = np.sin(theta)

    rot_mat = np.array([ct,-st,st,ct]).reshape((2,2))

    if len(args) == 1:
        return rot_mat

    if len(args) == 2:
        return np.dot(rot_mat,v)

def rotx(*args):

    if len(args) == 1:
        theta = args[0]

    elif len(args) == 2:
        v = args[0]
        theta = args[1]

    else:
        assert len(args) != 1 or len(args) != 2, 'must have one or two arguments'

    ct= np.cos(theta)
    st = np.sin(theta)
    rot_max = np.array([1,0,0,0,ct,-st,0,st,ct]).reshape((3,3))

    if len(args) == 1:
        return rot_max

    if len(args) == 2:
        return np.dot(rot_max, v)

def roty(*args):

    if len(args) == 1:
        theta = args[0]

    elif len(args) == 2:
        v = args[0]
        theta = args[1]

    else:
        assert len(args) != 1 or len(args) != 2, 'must have one or two arguments'


    ct = np.cos(theta)
    st = np.sin(theta)
    rot_max = np.array([ct,0,st,0,1,0,-st,0,ct]).reshape((3,3))

    if len(args) == 1:
        return rot_max

    if len(args) == 2:
        return np.dot(rot_max, v)

def rotz(*args):

    if len(args) == 1:

        theta = args[0]

    elif len(args) == 2:
        v = args[0]
        theta = args[1]

    else:
        assert len(args) != 1 or len(args) != 2, 'must have one or two arguments'

    ct = np.cos(theta)
    st = np.sin(theta)
    rot_max = np.array([ct,-st,0,st,ct,0,0,0,1]).reshape((3,3))

    if len(args) == 1:

        return rot_max

    if len(args) == 2:

        return np.dot(rot_max, v)

# general subr

def cycle(n, index):

    array = np.arange(1, n + 1)

    index =  np.mod((index -1),n)

    return array[int(index)]

def dofb(dim):

    return dim * (dim - 1) / 2

def dofc(dim):

    return 2 ** dim / 2 - 1

def v0(points):

    return np.vstack(points)

def T(d):

   dim = len(d)

   t =  np.eye(dim,dim)

   t[-1] = d

   return t

def mainPlanes(dim):

    # Returns the number main planes

    return comb(dim, 2)

def numTowards(dim):

    return dim-1

def dimAxis(ndim):

# Returns the dimension of a rotation 'axis' in a ndim dimentional space

    return ndim - 2

# general

def rotationMatrixMainAxis(theta,dim,*args):


    axis = args[0]

    if len(args) == 1:

        towards = int(cycle(dim,axis+1))

    elif len(args) == 2:

        towards = args[1]

    else:
        exit()


    c = np.cos(theta)
    s = np.sin(theta)

    r = np.eye(dim, dim)

    axisIndex = axis-1

    towardsIndex = towards -1

    r[axis-1,axis-1] = c

    r[towards-1,towards-1] = c

    r[axis-1,towards-1] = -1*s

    r[towards-1,axis-1] = s

    return r


def rotationMatrixGeneral(theta,v0,n):

    M1 = T(-1*v0[0])

    v1 = v0 @ M1

    M = M1

    vk = v1

    k = 1

    for r in range(1,n):

        for c in np.arange(n-1,r-1,-1):

            k = k + 1

            theta = np.arctan2(vk[r,c],vk[r,c-1])

            Mk = rotationMatrixMainAxis(theta,len(v0[0]),c,c-1)

            vk = vk @ Mk

            M = M @ Mk


    M = M @ rotationMatrixMainAxis(theta,n-1,n) @ la.inv(M)

    return M




print(rotationMatrixGeneral(0.5,v0(([0,0,0],[0,0,1])),3))

