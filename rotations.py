import numpy as np

# specific

def rot2(*args):

    if len(args) == 1:
        theta = args[0]

    elif len(args) == 2:
        v = args[0]
        theta = args[1]

    else:
        assert len(args) !=1 or len(args) !=2, 'must have one or two arguments'

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

def v0(points):

    return np.vstack(points)

def T(d):

   t =  np.eye(d,d)

   t[-1,:-1] = d

def vk(v0,k):

    vk = v0

    for i in np.arange(1,k+1):

        vk = vk* Mk(i)

    return vk

def Mk(v0,k,j):

    gt = generalTheta(v0,k,j)

    rotationMatrixMain(gt,j,j-1)

def mainPlanes(dim):
    return comb(dim, 2)

def numTowards(dim):
    pass

def dimAxis(dim):
    return dim - 2

# general

def rotationMatrixMainAxis(theta,*args):


    axis = args[0]

    dim = len(axis)

    if len(args) == 1:

        towards = cycle(mainPlanes(dim),axis+1)

    elif len(args) == 2:

        towards = args[2]

    else:
        exit()


    c = np.cos(theta)
    s = np.sin(theta)

    r = np.eye(dim, dim)

    r[axis,axis] = c

    r[towards,towards] = c

    r[axis,towards] = -1*s

    r[towards,axis] = s

    return r


def rotationMatrixGeneral(theta,v0,n):

    M1 = T(-v0[0])

    v1 = v0 * M1

    M = M1

    vk = v1

    k = 1

    for r in np.arange(1,n):

        for c in np.arange(n-1,r,-1):

            k = k + 1

            Mk = rotationMatrixMain(np.arctan2(vk[r,c],vk[r,c-1]),c,c-1)

            vk = vk @ Mk

            M = M @ Mk


    M = M @ rotationMatrixMain(theta,n-1,n) @ la.inv(M)

    return M



def rot(*args,**kwargs):
    pass