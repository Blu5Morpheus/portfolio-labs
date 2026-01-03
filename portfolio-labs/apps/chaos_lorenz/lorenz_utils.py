import numpy as np

def lorenz_deriv(state, sigma, rho, beta):
    """
    Compute the derivatives for the Lorenz system.
    
    Args:
        state (array): [x, y, z] state vector
        sigma (float): Prandtl number
        rho (float): Rayleigh number
        beta (float): Geometric factor
        
    Returns:
        np.array: [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def rk4_step(state, dt, sigma, rho, beta):
    """
    Perform a single RK4 integration step.
    """
    k1 = lorenz_deriv(state, sigma, rho, beta)
    k2 = lorenz_deriv(state + 0.5 * dt * k1, sigma, rho, beta)
    k3 = lorenz_deriv(state + 0.5 * dt * k2, sigma, rho, beta)
    k4 = lorenz_deriv(state + dt * k3, sigma, rho, beta)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_with_divergence(x0, epsilon, T, dt, sigma, rho, beta):
    """
    Simulate two Lorenz trajectories starting 'epsilon' apart.
    
    Args:
        x0 (array): Initial state [x, y, z] for trajectory 1
        epsilon (float): Initial separation in X direction for trajectory 2
        T (float): Total simulation time
        dt (float): Time step
        sigma, rho, beta: Lorenz parameters
        
    Returns:
        tuple: (time_array, traj1_array, traj2_array, divergence_array)
    """
    steps = int(T / dt)
    t = np.linspace(0, T, steps)
    
    # Initialize arrays
    traj1 = np.zeros((steps, 3))
    traj2 = np.zeros((steps, 3))
    
    # Set initial conditions
    traj1[0] = x0
    x0_prime = np.array(x0) + np.array([epsilon, 0.0, 0.0])
    traj2[0] = x0_prime
    
    curr1 = x0
    curr2 = x0_prime
    
    # Run simulation
    for i in range(1, steps):
        curr1 = rk4_step(curr1, dt, sigma, rho, beta)
        curr2 = rk4_step(curr2, dt, sigma, rho, beta)
        traj1[i] = curr1
        traj2[i] = curr2
        
    # Calculate divergence (Euclidean distance at each step)
    diff = traj1 - traj2
    divergence = np.linalg.norm(diff, axis=1)
    
    return t, traj1, traj2, divergence

def calculate_horizon(t, divergence, threshold):
    """
    Find the time t at which divergence first exceeds the threshold.
    """
    idx = np.where(divergence > threshold)[0]
    if len(idx) > 0:
        return t[idx[0]]
    return None

def estimate_lyapunov(t, divergence, window_fraction=0.3):
    """
    Estimate Lyapunov exponent from the slope of log(divergence).
    This is a rough heuristic. We look for a linear region in the log plot.
    Usually this is in the beginning but after transient, before saturation.
    
    For this MVP, we'll try to fit a line to the first 'window_fraction' of points
    where divergence is small but growing.
    """
    # Filter for reasonable range (e.g. between 1e-8 and 1.0 divergence)
    # to avoid numerical noise at start and saturation at end
    valid_indices = np.where((divergence > 1e-10) & (divergence < 5.0))[0]
    
    if len(valid_indices) < 10:
        return 0.0
        
    # Take a slice of valid indices
    limit = int(len(valid_indices) * window_fraction)
    if limit < 5: 
        limit = len(valid_indices)
        
    use_idx = valid_indices[:limit]
    
    log_div = np.log(divergence[use_idx])
    time_slice = t[use_idx]
    
    # Linear fit: log(delta) ~ lambda * t + C
    slope, _ = np.polyfit(time_slice, log_div, 1)
    return slope
