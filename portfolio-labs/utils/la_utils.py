import numpy as np
import numpy.linalg as LA

def get_matrix_properties(a, b, c, d):
    """
    Calculate essential linear algebra properties for matrix M = [[a,b],[c,d]].
    """
    M = np.array([[a, b], [c, d]])
    
    det = LA.det(M)
    trace = np.trace(M)
    cond = LA.cond(M)
    
    eig_vals, eig_vecs = LA.eig(M)
    
    is_real_eigen = np.all(np.isreal(eig_vals))
    
    return {
        "matrix": M,
        "det": det,
        "trace": trace,
        "cond": cond,
        "eig_vals": eig_vals,
        "eig_vecs": eig_vecs,
        "is_real_eigen": is_real_eigen
    }

def get_preset_matrix(name, t=0.0):
    """
    Returns matrix values (a,b,c,d) for given preset name.
    't' is a parameter for some presets (like angle).
    """
    if name == "Identity":
        return 1.0, 0.0, 0.0, 1.0
        
    elif name == "Rotation (45°)":
        theta = np.radians(45)
        c, s = np.cos(theta), np.sin(theta)
        return c, -s, s, c
    
    elif name == "Shear (X)":
        return 1.0, 1.0, 0.0, 1.0
        
    elif name == "Scaling (2x)":
        return 2.0, 0.0, 0.0, 2.0
    
    elif name == "Reflection (Y-axis)":
        return -1.0, 0.0, 0.0, 1.0
        
    elif name == "Singular (Projection X)":
        return 1.0, 0.0, 0.0, 0.0
        
    elif name == "Near Singular":
        # Determinant is very small
        return 1.0, 0.99, 1.0, 1.0
        
    return 1.0, 0.0, 0.0, 1.0

def interpolate_matrices(M_start, M_end, t):
    """
    Linear interpolation between two matrices: M(t) = (1-t)M_start + t*M_end
    """
    return (1 - t) * M_start + t * M_end
