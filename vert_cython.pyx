'''

PURPOSE: To bin 3D depending on user preference

AUTHOR: Fiaz Ahmed

DATE:   05/6
 
'''

import numpy as np
cimport numpy as np
from libc.math cimport abs
import cython

DTYPE = np.float
DTYPE1 = np.int
ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE1_t

cdef extern from "math.h":
    bint isfinite(double x)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
 
            
def bin_vert_struct(np.ndarray[DTYPE1_t, ndim=1] lev,np.ndarray[DTYPE1_t, ndim=1] zind,np.ndarray[DTYPE1_t, ndim=1] aind,
np.ndarray[DTYPE1_t, ndim=1] wind,np.ndarray[DTYPE_t, ndim=2] xt,np.ndarray[DTYPE_t, ndim=2] yq,
np.ndarray[DTYPE_t, ndim=4] op1, np.ndarray[DTYPE_t, ndim=4] op2, np.ndarray[DTYPE_t, ndim=4] op3):

    cdef unsigned int vector_size = zind.size
    cdef unsigned int ht_size = lev.size
        
    cdef Py_ssize_t i,j
    
    for i in range(vector_size-1):
        for j in range(lev.size-1):
            
            op1[j,zind[i],aind[i],wind[i]]+=xt[j,i]
            op2[j,zind[i],aind[i],wind[i]]+=yq[j,i]
            op3[i,zind[i],aind[i],wind[i]]+=1
            


def vert_integ_variable_bl(np.ndarray[DTYPE_t, ndim=2] var,
np.ndarray[DTYPE_t, ndim=1] var_ps,
np.ndarray[DTYPE_t, ndim=2] var1,
np.ndarray[DTYPE1_t, ndim=1] pbl_ind,
np.ndarray[DTYPE_t, ndim=1] lev, 
np.ndarray[DTYPE_t, ndim=2] dp,
 
np.ndarray[DTYPE_t, ndim=1] op1,
np.ndarray[DTYPE_t, ndim=1] op2,
np.ndarray[DTYPE_t, ndim=1] op3,
np.ndarray[DTYPE1_t, ndim=1] ind_low):

#       cdef unsigned int vector_size = var_ps.size
#       cdef unsigned int ht_size = lev.size

    cdef unsigned int vector_size = len(var_ps)
    cdef unsigned int ht_size = len(lev)

    cdef Py_ssize_t i,j,ind,il,ctr
    cdef Py_ssize_t im

    #       il=ind_low
#   im=ind_low
          
    for i in range(vector_size):
  
        ind=pbl_ind[i]
        il=ind_low[i]
        op1[i]=var_ps[i]
  
        for j in range(ht_size-1):

        ## PBL ##            
            if j>=ind:
                op1[i]+=var[j,i]*dp[j,i]

        ## Mid and low-level ##
            if (j<ind) & (j>=il):
                op2[i]+=var[j,i]*dp[j,i]
                op3[i]+=var1[j,i]*dp[j,i]
                
# def vert_integ_variable_bl(np.ndarray[DTYPE_t, ndim=2] var,
# np.ndarray[DTYPE1_t, ndim=1] lev):
# 
# 
# #   cdef unsigned int vector_size = var_ps.size
#   cdef unsigned int ht_size = lev.size
# 
# #   cdef unsigned int vector_size = len(var_ps)
# #   cdef unsigned int ht_size = len(lev)
# # 
# #   cdef Py_ssize_t i,j,ind,ctr
# #   cdef Py_ssize_t im
# #   
# # #       il=ind_low
# #   im=ind_low
# #               
# #   for i in range(vector_size):
# #       
# #       ind=pbl_ind[i]
# #       il=ind_low[i]
# #       op1[i]=var_ps[i]
# #       
# #       for j in range(ht_size-1):
# # 
# #           ## PBL ##            
# #           if j>=ind:
# #               op1[i]+=var[j,i]*dp[j,i]
# # 
# #           ## Low-level ##
# # #              if (j<ind) & (j>=il):
# # #                  op2[i]+=var1[j,i]*dp[j,i]
# #  
# #           ## Mid-level ##
# # #              if (j<il) & (j>=im):
# # #                  op3[i]+=var1[j,i]*dp[j,i]
# # #                   
# #            ## Mid and low-level ##
# #           if (j<ind) & (j>=im):
# #               op2[i]+=var[j,i]*dp[j,i]
# #               op3[i]+=var1[j,i]*dp[j,i]
 

# def vert_integ_variable_trad(np.ndarray[DTYPE_t, ndim=2] var,
# np.ndarray[DTYPE_t, ndim=2] var1,
# np.ndarray[DTYPE1_t, ndim=1] pbl_ind,
# np.ndarray[DTYPE_t, ndim=1] lev, 
# np.ndarray[DTYPE_t, ndim=2] dp,
#  
# np.ndarray[DTYPE_t, ndim=1] op1,
# np.ndarray[DTYPE_t, ndim=1] op2,
# 
# np.ndarray[DTYPE1_t, ndim=1] ind_mid,
# np.ndarray[DTYPE1_t, ndim=1] ind_low):
# 
#   cdef unsigned int vector_size = len(pbl_ind)
#   cdef unsigned int ht_size = len(lev)
# 
#   cdef Py_ssize_t i,j,ind,ctr
#   cdef Py_ssize_t il,im
#   
#   im=ind_mid
#               
#   for i in range(vector_size):
#       
#       ind=pbl_ind[i]
#       il=ind_low[i]
#       
#       for j in range(ht_size-1):
# 
#           ## Mid and low-level ##
#           if (j<ind) & (j>=im):
#               op2[i]+=var[j,i]*dp[j,i]
# 
#           ## Low-level ##
#           if (j<ind) & (j>=il):
#               op1[i]+=var1[j,i]*dp[j,i]
                  


def vert_integ_variable_bl(np.ndarray[DTYPE_t, ndim=2] var,
np.ndarray[DTYPE_t, ndim=1] var_ps,
np.ndarray[DTYPE_t, ndim=2] var1,
np.ndarray[DTYPE1_t, ndim=1] pbl_ind,
np.ndarray[DTYPE_t, ndim=1] lev, 
np.ndarray[DTYPE_t, ndim=2] dp,
 
np.ndarray[DTYPE_t, ndim=1] op1,
np.ndarray[DTYPE_t, ndim=1] op2,
np.ndarray[DTYPE_t, ndim=1] op3,
# np.ndarray[DTYPE_t, ndim=1] op4,
# np.ndarray[DTYPE_t, ndim=1] op5,
# DTYPE1_t ind_mid,
np.ndarray[DTYPE1_t, ndim=1] ind_low):

    cdef unsigned int vector_size = len(var_ps)
    cdef unsigned int ht_size = len(lev)

    cdef Py_ssize_t i,j,ind,ctr
    cdef Py_ssize_t il,im
    
##     il=ind_low
#     im=ind_mid
                
    for i in range(vector_size):
    
        ind=pbl_ind[i]
        il=ind_low[i]
        op1[i]=var_ps[i]
        
        for j in range(ht_size-1):

            ## PBL ##            
            if j>=ind:
                op1[i]+=var[j,i]*dp[j,i]

            ## Low-level ##
            if (j<ind) & (j>=il):
                op2[i]+=var[j,i]*dp[j,i]
                op3[i]+=var1[j,i]*dp[j,i]
 
            ## Mid-level ##
#             if (j<il) & (j>=im):
#                 op4[i]+=var[j,i]*dp[j,i]
#                 op5[i]+=var1[j,i]*dp[j,i]
            

def vert_integ_exneri_variable_bl(np.ndarray[DTYPE_t, ndim=2] var,
np.ndarray[DTYPE_t, ndim=1] var_ps,
np.ndarray[DTYPE1_t, ndim=1] pbl_ind,
np.ndarray[DTYPE_t, ndim=1] lev, 
np.ndarray[DTYPE_t, ndim=2] dp,
 
np.ndarray[DTYPE_t, ndim=1] op1,
np.ndarray[DTYPE_t, ndim=1] op2,
np.ndarray[DTYPE1_t, ndim=1] ind_low):

    cdef unsigned int vector_size = len(var_ps)
    cdef unsigned int ht_size = len(lev)

    cdef Py_ssize_t i,j,ind,ctr
    cdef Py_ssize_t il
    
                
    for i in range(vector_size):
    
        ind=pbl_ind[i]
        il=ind_low[i]
        op1[i]=var_ps[i]
        
        for j in range(ht_size-1):

            ## PBL ##            
            if j>=ind:
                op1[i]+=var[j,i]*dp[j,i]

            ## Low-level ##
            if (j<ind) & (j>=il):
                op2[i]+=var[j,i]*dp[j,i]
 

def vert_integ_lt_variable_bl(np.ndarray[DTYPE_t, ndim=2] var1,
np.ndarray[DTYPE_t, ndim=2] var2,
np.ndarray[DTYPE1_t, ndim=1] pbl_ind,
np.ndarray[DTYPE_t, ndim=1] lev, 
np.ndarray[DTYPE_t, ndim=2] dp,
 
np.ndarray[DTYPE_t, ndim=1] op1,
np.ndarray[DTYPE_t, ndim=1] op2,
np.ndarray[DTYPE1_t, ndim=1] ind_low):

    cdef unsigned int vector_size = len(pbl_ind)
    cdef unsigned int ht_size = len(lev)

    cdef Py_ssize_t i,j,ind,ctr
    cdef Py_ssize_t il
    
                
    for i in range(vector_size):
    
        ind=pbl_ind[i]
        il=ind_low[i]
        
        for j in range(ht_size-1):

            ## Low-level ##
            if (j<ind) & (j>=il):
                op1[i]+=var1[j,i]*dp[j,i]
                op2[i]+=var2[j,i]*dp[j,i]
                
                
def find_closest_index(np.ndarray[DTYPE_t, ndim=1] pres,
np.ndarray[DTYPE_t, ndim=1] lev,
np.ndarray[DTYPE1_t, ndim=1] ind_lev):

    cdef unsigned int vector_size = len(pres)
    cdef unsigned int ht_size = len(lev)
    cdef double delta1,delta2

    cdef Py_ssize_t i,j
    
    for i in range(vector_size):
        ind_lev[i]=0
        for j in range(ht_size-1):
            delta1=abs(lev[ind_lev[i]]-pres[i])
            delta2=abs(lev[j]-pres[i])
            if (delta2<delta1):
                ind_lev[i]=j


    
