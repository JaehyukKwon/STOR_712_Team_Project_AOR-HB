import numpy as np

#Runs Gradient Descent with a fixed step size (alpha) for at most max_iter or until norm(gradient) <= epsilon
#x0 defines the starting point and fun is used to compute the gradient (should be given as a lambda function, see below)
#Outputs a vector of the iterates (x_all), the vector of the norm of the gradient at each step (g_all), and the number of
#iterations (i)
def gradient_descent_fixed(x0, fun, alpha, epsilon, max_iter):
    #Set current x to initial x
    x = x0

    #Compute the gradient using fun (mode = 2)
    gradient = fun(x, 2)
    #Calculate the norm of the gradient
    normg = np.linalg.norm(gradient)

    #Initialize the vectors to be output
    x_all = [x0]
    g_all = [normg]
    i = 0
    
    #Run until we have performed all iterations or found a point with sufficiently small norm
    while normg > epsilon and i < max_iter:
        #Gradient descent update
        x = x - alpha * gradient
        i += 1

        #Compute gradient at new point
        gradient = fun(x, 2)
        normg = np.linalg.norm(gradient)

        #Record the new point for output
        x_all.append(x)
        g_all.append(normg)
    return x_all, g_all, i


def heavy_ball_method(x0, fun, alpha, beta, epsilon, max_iter) : 
    #Set current x to initial x
    x = x0
    x_previous = x0

    #Compute the gradient using fun (mode = 2)
    gradient = fun(x, 2)
    #Calculate the norm of the gradient
    normg = np.linalg.norm(gradient)

    #Initialize the vectors to be output
    x_all = [x0]
    g_all = [normg]
    i = 0
    
    #Run until we have performed all iterations or found a point with sufficiently small norm
    while normg > epsilon and i < max_iter:
        #Gradient descent update
        x_new = x - alpha * gradient + beta * (x - x_previous)
        x_previous = x
        x = x_new
        i += 1

        #Compute gradient at new point
        gradient = fun(x, 2)
        normg = np.linalg.norm(gradient)

        #Record the new point for output
        x_all.append(x)
        g_all.append(normg)
    return x_all, g_all, i


def nesterov_accelerated_gradient(x0, fun, alpha, beta, epsilon, max_iter) :
    #Set current x to initial x
    x = x0
    y = x0
    x_previous = x0

    #Compute the gradient using fun (mode = 2)
    gradient = fun(x, 2)
    #Calculate the norm of the gradient
    normg = np.linalg.norm(gradient)

    #Initialize the vectors to be output
    x_all = [x0]
    y_all = [x0]
    g_all = [normg]
    i = 0
    
    #Run until we have performed all iterations or found a point with sufficiently small norm
    while normg > epsilon and i < max_iter:
        #Gradient descent update
        y = x + beta * (x - x_previous)
        x_previous = x
        x = y - alpha * fun(y, 2)
        i += 1

        #Compute gradient at new point
        gradient = fun(x, 2)
        normg = np.linalg.norm(gradient)

        #Record the new point for output
        x_all.append(x)
        y_all.append(y)
        g_all.append(normg)
    return x_all, y_all, g_all, i