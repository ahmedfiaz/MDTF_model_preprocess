'''

PURPOSE: To bin 3D depending on user preference
         To wrap the slowest number-crunching part in a function that uses cython

AUTHOR: Fiaz Ahmed

DATE:   03/11
 
'''

import numpy as np
cimport numpy as np
import cython

DTYPE = np.float
DTYPE1 = np.int
ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE1_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)


def bin_1d_reg(np.ndarray[DTYPE1_t, ndim=1] xind,
np.ndarray[DTYPE1_t, ndim=1] yind,
np.ndarray[DTYPE_t, ndim=2] op1):

    cdef unsigned int vector_size = xind.size        
    cdef Py_ssize_t i 
    
    for i in range(vector_size-1):
        op1[xind[i],yind[i]]+=1
            
def bin_5d_precip(np.ndarray[DTYPE_t, ndim=1] z,
np.ndarray[DTYPE1_t, ndim=1] uind,np.ndarray[DTYPE1_t, ndim=1] vind,
np.ndarray[DTYPE1_t, ndim=1] wind, np.ndarray[DTYPE1_t, ndim=1] xind, 
np.ndarray[DTYPE1_t, ndim=1] yind, np.ndarray[DTYPE_t, ndim=5] op1, 
np.ndarray[DTYPE_t, ndim=5] op2, np.ndarray[DTYPE_t, ndim=5] op3, 
DTYPE_t thresh):

    cdef unsigned int vector_size = z.size
        
    cdef Py_ssize_t i 
#     print op1.size
    for i in range(vector_size-1):
#         print i
#         print uind[i], vind[i], wind[i],xind[i],yind[i]
        op1[uind[i],vind[i],wind[i],xind[i],yind[i]]+=z[i]
        op2[uind[i],vind[i],wind[i],xind[i],yind[i]]+=1
        if z[i]>=thresh:
            op3[uind[i],vind[i],wind[i],xind[i],yind[i]]+=1

def bin_4d_vert(np.ndarray[DTYPE1_t, ndim=1] uind, 
np.ndarray[DTYPE1_t, ndim=1] wind, 
np.ndarray[DTYPE1_t, ndim=1] yind, 
np.ndarray[DTYPE1_t, ndim=1] zind, 
np.ndarray[DTYPE_t, ndim=2] var1, 
np.ndarray[DTYPE_t, ndim=2] var2,
np.ndarray[DTYPE_t, ndim=5] op1, 
np.ndarray[DTYPE_t, ndim=5] op2, 
np.ndarray[DTYPE_t, ndim=4] op3, 
np.ndarray[DTYPE1_t, ndim=1] lev):

    cdef unsigned int vector_size = zind.size
    cdef unsigned int lev_size = lev.size    
    cdef Py_ssize_t i 
    
    for i in range(vector_size-1):
        op3[uind[i],wind[i],yind[i],zind[i]]+=1 
        for j in range(lev_size):
#         print uind[i], wind[i],yind[i],zind[i]
            op1[j,uind[i],wind[i],yind[i],zind[i]]+=var1[j,i]
            op2[j,uind[i],wind[i],yind[i],zind[i]]+=var2[j,i]
                
def bin_4d_precip_mod(np.ndarray[DTYPE_t, ndim=1] z, np.ndarray[DTYPE1_t, ndim=1] uind,
np.ndarray[DTYPE1_t, ndim=1] wind, np.ndarray[DTYPE1_t, ndim=1] xind, 
np.ndarray[DTYPE1_t, ndim=1] yind, np.ndarray[DTYPE_t, ndim=4] op1, 
np.ndarray[DTYPE_t, ndim=4] op2, np.ndarray[DTYPE_t, ndim=4] op3, 
DTYPE_t thresh):

    cdef unsigned int vector_size = z.size
        
    cdef Py_ssize_t i 
    
    for i in range(vector_size-1):
#         print uind[i], wind[i],xind[i],yind[i]
        op1[uind[i],wind[i],xind[i],yind[i]]+=z[i]
        op2[uind[i],wind[i],xind[i],yind[i]]+=1
        if z[i]>=thresh:
            op3[uind[i],wind[i],xind[i],yind[i]]+=1 
                
def bin_3d_precip_mod(np.ndarray[DTYPE_t, ndim=1] z,
np.ndarray[DTYPE1_t, ndim=1] wind, np.ndarray[DTYPE1_t, ndim=1] xind, 
np.ndarray[DTYPE1_t, ndim=1] yind, 
np.ndarray[DTYPE_t, ndim=3] op1, np.ndarray[DTYPE_t, ndim=3] op2, 
np.ndarray[DTYPE_t, ndim=3] op3, DTYPE_t thresh):

    cdef unsigned int vector_size = wind.size
        
    cdef Py_ssize_t i 
    
    for i in range(vector_size-1):
        #print wind[i],xind[i],yind[i]
        op1[wind[i],xind[i],yind[i]]+=z[i]
        op2[wind[i],xind[i],yind[i]]+=1
        if z[i]>=thresh:
            op3[wind[i],xind[i],yind[i]]+=1
            
def bin_2d_precip_mod(np.ndarray[DTYPE_t, ndim=1] z,
np.ndarray[DTYPE1_t, ndim=1] xind, 
np.ndarray[DTYPE1_t, ndim=1] yind, 
np.ndarray[DTYPE_t, ndim=2] op1, np.ndarray[DTYPE_t, ndim=2] op2,
np.ndarray[DTYPE_t, ndim=2] op3, DTYPE_t thresh):

    cdef unsigned int vector_size = xind.size
        
    cdef Py_ssize_t i 
    
    for i in range(vector_size-1):
        op1[xind[i],yind[i]]+=z[i]
        op2[xind[i],yind[i]]+=1
        if z[i]>=thresh:
            op3[xind[i],yind[i]]+=1

def bin_1d_precip_mod(np.ndarray[DTYPE_t, ndim=1] w,
np.ndarray[DTYPE_t, ndim=1] z, np.ndarray[DTYPE1_t, ndim=1] wind,
np.ndarray[DTYPE_t, ndim=1] op1, np.ndarray[DTYPE_t, ndim=1] op2, 
np.ndarray[DTYPE_t, ndim=1] op3,DTYPE_t thresh):

    cdef unsigned int vector_size = w.size
        
    cdef Py_ssize_t i 
    
    for i in range(vector_size-1):
        op1[wind[i]]+=z[i]
        op2[wind[i]]+=1
        if z[i]>=thresh:
            op3[wind[i]]+=1
            
