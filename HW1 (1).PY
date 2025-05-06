import numpy as np
from scipy.optimize import fsolve, root

# Define the system of equations
def system_eqs(x):
    x1, x2, x3 = x
    eq1 = 10*x1 + 2*x2**2 - 3*x3 - 9
    eq2 = x1**2 + 3*x2 - x3**2 + 2
    eq3 = x1*x2*x3 - 6
    return [eq1, eq2, eq3]

# Gradient of the system (Jacobian for steepest descent)
def gradient(x):
    x1, x2, x3 = x
    grad_f1 = [10, 4*x2, -3]
    grad_f2 = [2*x1, 3, -2*x3]
    grad_f3 = [x2*x3, x1*x3, x1*x2]
    return np.array([grad_f1, grad_f2, grad_f3])

# Steepest Descent method (improved version)
def steepest_descent(system_eqs, initial_guess, max_iter=1000, tol=1e-6, alpha=0.1):
    x = np.array(initial_guess)
    for i in range(max_iter):
        # Compute the residuals
        res = np.array(system_eqs(x))
        
        # Compute the gradient (Jacobian)
        grad = gradient(x)
        
        # Update the solution
        step = np.linalg.lstsq(grad, -res, rcond=None)[0]  # Solve for step in the direction of the gradient
        x = x + alpha * step  # Update the solution with step size alpha
        
        # Check for convergence
        if np.linalg.norm(res) < tol:
            break
    
    return x

# Continuation method (refined version)
def continuation_method(system_eqs, initial_guess, param_range, max_iter=100, tol=1e-6):
    x = np.array(initial_guess)
    previous_solution = x
    for param in param_range:
        # Gradually modify the system equations based on the parameter (smooth transition)
        def parametric_system(x):
            x1, x2, x3 = x
            # Smoothly modify constants based on the parameter
            eq1 = 10*x1 + 2*x2**2 - 3*x3 - (9 + param)  # Gradually modify constants
            eq2 = x1**2 + 3*x2 - x3**2 + (2 + param)  # Gradually modify constants
            eq3 = x1*x2*x3 - (6 + param)  # Gradually modify constants
            return [eq1, eq2, eq3]
        
        # Solve the system for the current parameter value
        solution = fsolve(parametric_system, x)
        
        # Prevent solutions from diverging too much
        if np.linalg.norm(solution - previous_solution) > 1e3:  # If the solution diverges too much, reset
            print(f"Warning: Solution diverging at param={param}. Reverting to previous solution.")
            solution = previous_solution
        
        x = solution  # Update the solution for the next iteration
        previous_solution = solution  # Keep track of the previous solution for divergence control

        # Check for convergence
        if np.linalg.norm(system_eqs(x)) < tol:
            break
    
    return x

# Solve the system using different methods

# Initial guess for the solution
initial_guess = [1, 1, 1]

# 1. Newton's Method using fsolve (which uses Newton's method internally)
solution_newton = fsolve(system_eqs, initial_guess)
print(f"Newton's method solution: x1 = {solution_newton[0]}, x2 = {solution_newton[1]}, x3 = {solution_newton[2]}")

# 2. Broyden's Method using root (broyden1)
solution_broyden = root(system_eqs, initial_guess, method='broyden1')
print(f"Broyden's method solution: x1 = {solution_broyden.x[0]}, x2 = {solution_broyden.x[1]}, x3 = {solution_broyden.x[2]}")

# 3. Steepest Descent Method
solution_steepest = steepest_descent(system_eqs, initial_guess)
print(f"Steepest Descent solution: x1 = {solution_steepest[0]}, x2 = {solution_steepest[1]}, x3 = {solution_steepest[2]}")

# 4. Continuation Method (parameter range from 0 to 2)
param_range = np.linspace(0, 2, 10)  # Adjust the range for smoother transition
solution_continuation = continuation_method(system_eqs, initial_guess, param_range)
print(f" Continuation method solution: x1 = {solution_continuation[0]}, x2 = {solution_continuation[1]}, x3 = {solution_continuation[2]}")
