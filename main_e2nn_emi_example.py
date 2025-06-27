################################################################################
################################################################################
"""                          (0) PRELIMINARY SETUP                           """
################################################################################
################################################################################

################################################################################
"""IMPORT PACKAGES"""
import numpy as np

from numpy.random import default_rng
rng = default_rng()

from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import t

import scipy
from scipy.optimize import minimize

from pyDOE3 import lhs, ff2n

import pickle
import os  # for making folders to put plots in

from build_ensemble import build_ensemble
from predict_ensemble import predict_ensemble

from user_defined_test_functions import uninformative_nd_lf
from user_defined_test_functions import fun3d_simulation_lift_force

from user_defined_test_functions import get_rosen_emulator, \
    get_rosen_biv_emulator, return_sum_emulator

# e2nn_models, emulator_functions, x_scale_obj, y_scale_obj
# e2nn_ensembles, emulator_function_lists, x_scale_objs, y_scale_objs


# fit_gpr
# predict_gpr
################################################################################
"""HANDLING FIGURES"""

# enable plotting
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

# Improve figure appearence
import matplotlib as mpl
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)


global global_parallel_acquisitions
global_parallel_acquisitions = 0


################################################################################
"""GAUSSIAN PROCESS MODELS"""

# def fit_gpr(X_train_raw, Y_train_raw):
#     """
# 
#     """
# 
#     ################################################################################
#     """STANDARDIZE TRAINING DATA"""
# 
#     # scale data to 0 mean and unit variance using sk-learn
#     from sklearn.preprocessing import StandardScaler
#     x_stand_obj = StandardScaler().fit(X_train_raw) 
#     X_train = x_stand_obj.transform(X_train_raw)
#     # (Y data is scaled internally, so it doesn't need to be changed)
# 
#     ################################################################################
#     """FIT GAUSSIAN PROCESS REGRESSION MODEL TO DATA"""
# 
#     print("Building GPR model")
# 
#     # construct gpr model and train on data
#     from sklearn.gaussian_process import GaussianProcessRegressor
#     from sklearn.gaussian_process.kernels import Matern, RBF
# 
#     # # Version 1: default optimizer
#     # GPR = GaussianProcessRegressor( # kernel = Matern()
#     #         kernel=1.0*RBF(1.0), normalize_y=True, alpha=1e-10, optimizer="fmin_l_bfgs_b",
#     #         n_restarts_optimizer=250) #25) #25
#     # GPR.fit(X_train, Y_train_raw)
# 
#     # Version 2: default optimizer but with extra objective evaluations
#     callable_opt = lambda obj_func, th0, bounds : scipy.optimize.fmin_l_bfgs_b(obj_func, x0=th0, bounds=bounds, factr=10000000.0, pgtol=1e-05, epsilon=1e-08, maxfun=150_000, maxiter=150_000)[0:2] # default: maxfun=15_000, maxiter=15_000
#     GPR = GaussianProcessRegressor( # kernel = Matern()
#             kernel=1.0*RBF(1.0), normalize_y=True, alpha=1e-10, optimizer=callable_opt,
#             n_restarts_optimizer=250) #32) #25) #
#     GPR.fit(X_train, Y_train_raw)
# 
#     # print("kernel theta value: ", np.exp(GPR.kernel_.theta))
#     # print("log_marginal_likelihood_value_: ", GPR.log_marginal_likelihood_value_)
# 
#     print("Done building GPR model")
# 
#     return(GPR, x_stand_obj)


# def predict_gpr(gpr_model, X_pred_raw, x_stand_obj):
#     """
#     Outputs of gpr_model.predict() are flat regardless of the input shape
#     """
#     X_pred = x_stand_obj.transform(X_pred_raw)
#     Y_pred_raw, Sig_pred_raw = gpr_model.predict(X_pred, return_std=True)
#     return(Y_pred_raw, Sig_pred_raw)


# def EI_acquisition(x, GPR_MODEL, x_stand_obj):
#     """
#     Compute Expected Improvement of GPR model
#     """
#     mu, sig = predict_gpr(GPR_MODEL, x, x_stand_obj)
#     f_best = np.min(Y_train_raw)
#     z = (f_best-mu)/sig
#     EI = (f_best-mu)*norm.cdf(z) + sig*norm.pdf(z)
#     return(EI)


#  def EGRA(x, GPR_models, x_stand_objs):
#      """ Original Bichon activation function """
#      ys = [] #np.zeros(len(GPR_models))
#      sigs = [] #np.zeros(len(GPR_models))
#      for c, (GPR_model, x_stand_obj) in enumerate(zip(GPR_models, x_stand_objs)):
#          y_c, sig_c = predict_gpr(GPR_model, x, x_stand_obj)
#          y_c = y_c.reshape(-1, 1)
#          sig_c = sig_c.reshape(-1, 1)
#          ys.append(y_c)
#          sigs.append(sig_c)
#    
#      ys = np.hstack(ys)
#      sigs = np.hstack(sigs)
#    
#      c_idx = np.argmax(ys, axis=1)
#      pt_idx = np.arange(ys.shape[0])
#      mu = ys[pt_idx,c_idx]
#      sig = sigs[pt_idx,c_idx]
#  
#      eps = np.finfo("float64").eps
#      alpha = 2
#      zbar = 0
#      zp =  alpha*sig
#      zm = -alpha*sig
#  
#      u1 = -mu/(sig+eps)
#      u2 = (zp-mu)/(sig+eps)
#      u3 = (zm-mu)/(sig+eps)
#  
#      CLS = mu*(2*norm.cdf(u1)-norm.cdf(u2)-norm.cdf(u3)) - sig*(2*norm.pdf(u1) - norm.pdf(u2)-norm.pdf(u3)) + alpha*sig*(norm.cdf(u2)-norm.cdf(u3))
#    
#      return(CLS, c_idx)


# def ALT(x, GPR_models, x_stand_objs):
#     """My alternative acquisition function"""
#     ys = [] #np.zeros(len(GPR_models))
#     sigs = [] #np.zeros(len(GPR_models))
#     for c, (GPR_model, x_stand_obj) in enumerate(zip(GPR_models, x_stand_objs)):
#         y_c, sig_c = predict_gpr(GPR_model, x, x_stand_obj)
#         y_c = y_c.reshape(-1, 1)
#         sig_c = sig_c.reshape(-1, 1)
#         ys.append(y_c)
#         sigs.append(sig_c)
#     
#     ys = np.hstack(ys)
#     sigs = np.hstack(sigs)
#     
#     c_idx = np.argmax(ys, axis=1)
#     pt_idx = np.arange(ys.shape[0])
#     mu = ys[pt_idx,c_idx]
#     sig = sigs[pt_idx,c_idx]
# 
#     eps = np.finfo("float64").eps
# 
#     const = 1
#     # alt = -(np.abs(mu)+1)/(sig+eps)
#     alt = (sig+eps)/(np.abs(mu)+1)
#     return(alt, c_idx)


# def EMI(x, GPR_models, x_stand_objs):
#     """Expected Magnitude of Incorrectness (EMI)"""
#     eps = np.finfo("float64").eps
# 
#     ys = [] #np.zeros(len(GPR_models))
#     sigs = [] #np.zeros(len(GPR_models))
#     for c, (GPR_model, x_stand_obj) in enumerate(zip(GPR_models, x_stand_objs)):
#         y_c, sig_c = predict_gpr(GPR_model, x, x_stand_obj)
#         y_c = y_c.reshape(-1, 1)
#         sig_c = sig_c.reshape(-1, 1)
#         ys.append(y_c)
#         sigs.append(sig_c+eps)
#   
#     ys = np.hstack(ys)
#     sigs = np.hstack(sigs)
#   
#     # choose index with highest probability of failure (not closest to failure)
#     c_idx = np.argmax(ys/sigs, axis=1)
#     pt_idx = np.arange(ys.shape[0])
#     mu = ys[pt_idx,c_idx]
#     sig = sigs[pt_idx,c_idx]
#   
#     # calculate EMI
#     emi = -np.abs(mu)*norm.cdf(-np.abs(mu/sig))+sig*norm.pdf(mu/sig)
#     return(emi, c_idx)


def EMI(X_unscaled, e2nn_ensembles, emulator_function_lists, x_scale_objs, y_scale_objs):
    """Expected Magnitude of Incorrectness (EMI)"""
    eps = np.finfo("float64").eps

    ys = [] #np.zeros(len(e2nn_ensembles))
    sigs = [] #np.zeros(len(e2nn_ensembles))
    dfs = []
    for c, (e2nn_models, emulator_functions, x_scale_obj, y_scale_obj) in enumerate(zip(e2nn_ensembles, emulator_function_lists, x_scale_objs, y_scale_objs)):
        y_c, sig_c, df = predict_ensemble(X_unscaled, e2nn_models, emulator_functions, x_scale_obj, y_scale_obj)
        y_c = y_c.reshape(-1, 1)
        sig_c = sig_c.reshape(-1, 1)
        ys.append(y_c)
        sigs.append(sig_c+eps)
        dfs.append(df)
  
    ys = np.hstack(ys)
    sigs = np.hstack(sigs)
    dfs = np.hstack(dfs)
    dfs = np.tile(dfs, [ys.shape[0], 1])

    # choose index with highest probability of failure (not closest to failure)
    p_fails = 1 - t.cdf((0 - ys) / sigs, dfs)
    c_idx = np.argmax(p_fails, axis=1)
    pt_idx = np.arange(ys.shape[0])
    mu = ys[pt_idx,c_idx]    # mu is a column vector
    sig = sigs[pt_idx,c_idx]    # sig is a column vector
    df = dfs[pt_idx, c_idx]    # df column vector
  
    # calculate EMI
    z = mu/sig
    emi = -np.abs(mu)*t.cdf(-np.abs(z), df) + df/(df-1)*(1+z**2/df)*sig*t.pdf(z, df)
    return(emi, c_idx)



# def EMRI(x, GPR_models, x_stand_objs):
#     """
#     Expected Magnitude of Incorrectness (EMI) 
#     Now includes probability that changing a constraint to be feasible 
#     changes nothing (because other constraints are infeasible)
#     """
#     eps = np.finfo("float64").eps
# 
#     ys = [] #np.zeros(len(GPR_models))
#     sigs = [] #np.zeros(len(GPR_models))
#     for c, (GPR_model, x_stand_obj) in enumerate(zip(GPR_models, x_stand_objs)):
#         y_c, sig_c = predict_gpr(GPR_model, x, x_stand_obj)
#         y_c = y_c.reshape(-1, 1)
#         sig_c = sig_c.reshape(-1, 1)
#         ys.append(y_c)
#         sigs.append(sig_c+eps)
#   
#     ys = np.hstack(ys)
#     sigs = np.hstack(sigs)
#   
#     # choose index with highest probability of failure (not closest to failure)
#     p_failure = norm.cdf(ys/sigs)
#     c_idx = np.argmax(p_failure, axis=1)
# 
#     pt_idx = np.arange(ys.shape[0])
#     mu = ys[pt_idx,c_idx]
#     sig = sigs[pt_idx,c_idx]
#   
# 
#     # Get probabilities that all other constraints are feasible
#     p_feasible = 1-p_failure
#     mask_array = np.full(p_feasible.shape, True)
#     mask_array[pt_idx, c_idx] = False
#     new_shape = (mask_array.shape[0], mask_array.shape[1]-1)
#     p_other_c_feasible = p_feasible[mask_array].reshape(new_shape)
#     p_all_other_c_feasible = np.prod(p_other_c_feasible, axis=1)
#     p_incorrectness_is_relevant = p_all_other_c_feasible
# 
#     # Incorrectness has to be relevent if current prediction is feasible
#     pred_feasible_idx = mu<0
#     p_incorrectness_is_relevant[pred_feasible_idx] = 1 
# 
#     # calculate EMI
#     emi = -np.abs(mu)*norm.cdf(-np.abs(mu/sig))+sig*norm.pdf(mu/sig)
#     emi = emi * p_incorrectness_is_relevant
#     return(emi, c_idx)



# def NRMSE(y_pred, y_true):
#     """
#     Normalized root mean squared error
#     """
#     y_pred = y_pred.flatten()
#     y_true = y_true.flatten()
#     y_mean = np.mean(y_true)
#     additive_NRMSE = np.sqrt(np.sum((y_pred-y_true)**2)/np.sum((y_mean-y_true)**2))
#     return(additive_NRMSE)

# def sum_log_likelihood_norm(y_true, y_pred, scale):
#     """
#     returns the sum of log likelihoods for a normal distribution
#     an error metric that takes model uncertainty or PDF into account
#     """
#     SLL = np.sum(norm.logpdf(y_true, loc=y_pred, scale=scale))
#     return(SLL)

############################################################################
""" OPTIMIZATION USING GAUSSIAN SCATTER"""


def minimize_gaussian_scatter(obj_fn, x0, step0, bounds, Nscatter=10_000, ftol=1e-6, xtol=1e-6):# maxfun=15000, maxiter=15000):
    """
    Local optimization using Gaussian Scatter method
    ftol: if all evaluations yield the same value to within ftol, converge (region flat)
    xtol: if the standard deviation drops to this level, converge (optimum approached)
    """
    Ndim = len(x0)
    stdev = step0
    y0 = obj_fn(x0)

    pdfs_at_x0 = np.array([])

    # bounds were scaled from [[0, 1],[0, 1], [0, 1], ... ]
    lb = bounds[:,0]
    ub = bounds[:,1]

    lb = np.tile(lb, (Nscatter, 1))
    ub = np.tile(ub, (Nscatter, 1))

    converged = False
    global global_parallel_acquisitions


    while not converged:

        # find optimum
        x_scatter = rng.normal(loc = x0, scale = stdev/np.sqrt(Ndim), size = (Nscatter, Ndim))
        low = x_scatter<lb
        high = x_scatter>ub
        x_scatter[low] = lb[low]
        x_scatter[high] = ub[high]

        y_scatter = obj_fn(x_scatter)

        idx = np.argmin(y_scatter)
        x_opt = x_scatter[idx,:]
        y_opt = y_scatter[idx]

        # check convergence
        if stdev < xtol:
            converged = True
        if np.max(y_scatter) - np.min(y_scatter) < ftol:
            converged = True

        global_parallel_acquisitions += 1
        print("global_parallel_acquisitions: ", global_parallel_acquisitions)

        if obj_fn(x_opt) < obj_fn(x0):
            PDF_opt = multivariate_normal.pdf(x_opt, x0, stdev/np.sqrt(Ndim)*np.eye(Ndim))
            stdev = (PDF_opt * Nscatter)**-(1/Ndim) # density-based
            # # stdev = np.sum((x_opt - x0)**2)**(1/Ndim) # step-based
            x0 = x_opt
            y0 = y_opt
        else:
            # if no improvement, shrink search area
            stdev *= 0.5 # halve distance
            # stdev *= 0.5**(1/Ndim) # halve hypervolume

        x_dist = np.sqrt(np.sum((x0 - 0.75)**2))

    return(x_opt, y_opt)



def global_optimization(obj_fn, lb, ub, Ndim, n_scatter_init = 100_000,  n_local_opts = 10, previous_local_xopt = np.array([]), n_scatter_gauss = 128):
    """
    Meant for optimizing functions that are cheap to evaluate in parallel but difficult with many local minima
    """

    ############################################################################
    """Scatter points"""
    print(f"Scattering {n_scatter_init} points")
    print("Start LHS")
    X_scatter = lhs(Ndim, samples=n_scatter_init, criterion="maximin", iterations=10)
    print("End LHS")
    X_scatter = (ub-lb)*X_scatter + lb
    Y_scatter = obj_fn(X_scatter)
    sort_idx = np.argsort(Y_scatter.flatten())

    Y_scatter = Y_scatter[sort_idx]
    X_scatter = X_scatter[sort_idx,:]

    from sklearn.preprocessing import MinMaxScaler
    x_lim = np.vstack([lb, ub])
    xscale_obj = MinMaxScaler(feature_range=(0, 1)).fit(x_lim)
    X_scatter_scaled = xscale_obj.transform(X_scatter)
    print("done scattering points")

    ############################################################################
    """Select local minima"""
    print("Selecting local minima")
    # initial: hypercube exclusion zone side length is 2 times (Volume/Npt)**1/Ndim
    #          scales up volume by 2**Ndim
    #  (later try 4 times, scaling up volume by 4**Ndim)
    # Select top 10 local optima

    # exclusion_dist = 0.5 * Ndim * (1/n_scatter_init)**(1/Ndim) # half length
    exclusion_dist = Ndim * (1/n_scatter_init)**(1/Ndim) # manhattan exclusion distance (diag len is double this)
    # exclusion_dist = 2*Ndim * (1/n_scatter_init)**(1/Ndim) # double exclusion distance to allow some margin

    """only check N_check best optima"""
    # # scale down problem size (scalability)
    # N_check = 100
    # X_scatter_scaled = X_scatter_scaled[:Ncheck, :]

    local_optima_x = X_scatter_scaled[0,:].reshape([1,-1])
    local_optima_y = np.array([Y_scatter[0]])
    exclusion_points = local_optima_x.copy()

    for ii in range(1, n_scatter_init):

        # check convergence
        if local_optima_y.size >= n_local_opts:
            break

        x_check = X_scatter_scaled[ii,:]
        manhattan_dists = np.sum(np.abs(x_check - exclusion_points), axis = 1)
        if np.all(manhattan_dists > exclusion_dist):
            local_optima_x = np.vstack([local_optima_x, x_check])
            local_optima_y = np.vstack([local_optima_y, Y_scatter[ii]])
        exclusion_points = np.vstack([exclusion_points, x_check])

    local_optima_x_init = local_optima_x.copy()
    local_optima_y_init = local_optima_y.copy()

    """ check all Nscatter points, but discard more quickly """
    # Track local optimas and exclusion hyperspheres around them. Hypersphere radius
    # is distance to furthest point discarded by hypershphere, plus some exclusion
    # distance (either n_scatter_init**(-1/Ndim) or double that). Each scatter point
    # is checked against all local optima hyperspheres, and discarded by the nearest
    # if it falls within any.

    """combine local optimas found using both scalable methods"""
    # use the "unique points" thing with a tighter tolerance (1e-9 or something)

    print("Done selecting local minima")

    ############################################################################
    """Perform optimization to improve local minima"""

    # # option 1: L-BFGS-B
    #
    # # bounds = np.hstack([lb.reshape(-1,1), ub.reshape(-1, 1)])
    # print("np.zeros([lb.size, 1]): ", np.zeros([lb.size, 1]))
    # print("np.ones([ub.size, 1]): ", np.ones([ub.size, 1]))
    # bounds = np.hstack([np.zeros([lb.size, 1]), np.ones([ub.size, 1])])
    # # obj_fn_scaled = lambda x : obj_fn(xscale_obj.inverse_transform(x))
    # obj_fn_scaled = lambda x : obj_fn(xscale_obj.inverse_transform(x.reshape([-1, Ndim])))
    #
    # for ii in range(local_optima_y.size):
    #     x0 = local_optima_x[ii,:]
    #     # x0 = x0.reshape((1, -1))
    #     print("x0: ", x0)
    #     y0 = local_optima_y[ii]
    #     res = minimize(obj_fn_scaled, x0 = x0, bounds = bounds, method="L-BFGS-B", options={"maxcor": 2*Ndim, "ftol": 1e-6, "eps": 1e-03, "maxfun": 15000, "maxiter": 15000}) # "ftol": 1e-6, "eps": 1e-06
    #     # res = minimize(obj_fn_scaled, x0 = x0, bounds = bounds, method="SLSQP", options={"ftol": 1e-7, "eps": 1e-3}) # "ftol": 1e-7, "eps": 1e-7
    #
    #     # update with the refined local optimum, but only if it is an improvement
    #     if res.fun < y0:
    #         local_optima_x[ii,:] = res.x
    #         local_optima_y[ii] = res.fun
    #         print("Optimum improved for local optimum "+str(ii+1)+" of "+str(local_optima_y.size))
    #     else:
    #         local_optima_x[ii,:] = x0
    #         local_optima_y[ii] = y0
    #         print("Optimation failed to improve for local optimum "+str(ii+1)+" of "+str(local_optima_y.size))


    # option 2: Gaussian scatter (must choose standard deviation) [based off of local sample density from last iter]
    # standard deviation = stacked spacing
    obj_fn_scaled = lambda x : obj_fn(xscale_obj.inverse_transform(x.reshape([-1, Ndim])))
    bounds = np.hstack([np.zeros([lb.size, 1]), np.ones([ub.size, 1])])
    step0 = n_local_opts**-(1/Ndim) # n_scatter_init**-(1/Ndim)
    for ii in range(local_optima_y.size):
        # in case only a single local optima found, change to 2D
        if len(local_optima_x.shape)==1:
            local_optima_x = local_optima_x.reshape([1,-1])
        x0 = local_optima_x[ii,:]

        # gaussian scatter
        [x_opt, y_opt] = minimize_gaussian_scatter(obj_fn_scaled, x0, step0, bounds, Nscatter = n_scatter_gauss, ftol = 1e-6, xtol = 1e-4) #, ftol = 1e-6, xtol = 1e-5)

        local_optima_x[ii,:] = x_opt
        local_optima_y[ii] = y_opt

    ############################################################################
    """Select final optimum"""

    x_opts = xscale_obj.inverse_transform(local_optima_x)
    y_opts = local_optima_y

    idx = np.argmin(y_opts)
    x_opt = x_opts[idx, :]
    y_opt = y_opts[idx]

    # de-scale
    exclusion_points = xscale_obj.inverse_transform(exclusion_points)
    local_optima_x_init = xscale_obj.inverse_transform(local_optima_x_init)

    return(x_opt, y_opt, x_opts, y_opts, exclusion_points, X_scatter, Y_scatter, local_optima_x_init, local_optima_y_init)




################################################################################
"""USER-DEFINED SAMPLING AND FUNCTION"""

def rosenbrock(X, c1=100.0, c2=1.0):
    """
    Takes in a 2D matrix of samples and outputs 2D column matrix of
    Rosenbrock function responses
    The function must be 2D or higher
    """
    Y = np.sum(c1*(X[:,1:] - X[:,:-1]**2.0)**2.0 +c2* (1 - X[:,:-1])**2.0, axis=1)
    return(Y)

def constr_rosen_nd(X):
    d = X.shape[1]
    boundary = rosenbrock(-np.ones(d).reshape(1,-1))
    return rosenbrock(X)-boundary

def constr_rosen_2d(X):
    boundary = 10
    return rosenbrock(X) - boundary


def g1(X):
    """ first constraint """
    if len(X.shape)==1: # single point
        x1 = X[0]
        x2 = X[1]
    else: 
        x1 = X[:,0]
        x2 = X[:,1]
    Y = (x1**2+4)*(x2-1)/20 - np.sin(5/2*x1) - 2
    return(Y)

def g2(X):
    """ second constraint """
    if len(X.shape)==1: # single point
        x1 = X[0]
        x2 = X[1]
    else: 
        x1 = X[:,0]
        x2 = X[:,1]
    Y = (x1+2)**4 - x2 + 4
    return(Y)

def g3(X):
    """ third constraint """
    if len(X.shape)==1: # single point
        x1 = X[0]
        x2 = X[1]
    else: 
        x1 = X[:,0]
        x2 = X[:,1]
    Y = (x1-4)**3 - x2 + 2
    return(Y)

def vehicle_side_impact(x, idx):
    """
    Failure modes:
    L: Abdomon load
    F: Pubic symphysis force
    Du: Rib deflection upper
    Dm: Rib deflection middle
    Dl: Rib deflection lower
    VCu: Viscous creiteria upper
    VCm: Viscous creiteria middle
    VCl:Viscous creiteria lower
    VB: Velocity at B-pillar
    VD: Velocity at door
    """
    x1 = x[:,0]
    x2 = x[:,1]
    x3 = x[:,2]
    x4 = x[:,3]
    x5 = x[:,4]
    x6 = x[:,5]
    x7 = x[:,6]
    x8 = x[:,7]
    x9 = x[:,8]
    x10 = x[:,9]
    x11 = x[:,10]
    L = 1.16-0.3717*x2*x4-0.00931*x2*x10-0.484*x3*x9+0.01343*x6*x10
    F = 4.72-0.5*x4-0.19*x2*x3-0.0122*x4*x10+0.009325*x6*x10+0.000191*x11**2
    Du = 28.98+3.818*x3-4.2*x1*x2+0.0207*x5*x10+6.63*x6*x9-7.7*x7*x8+0.32*x9*x10
    Dm = 33.86+2.95*x3+0.1792*x10-5.057*x1*x2-11.0*x2*x8-0.0215*x5*x10-9.98*x7*x8+22.0*x8*x9
    Dl=46.36-9.9*x2-12.9*x1*x8+0.1107*x3*x10
    VCu = 0.261-0.0159*x1*x2-0.188*x1*x8-0.019*x2*x7+0.0144*x3*x5+0.0008757*x5*x10+0.08045*x6*x9+0.00139*x8*x11+0.00001575*x10*x11
    VCm = 0.214+0.00817*x5-0.131*x1*x8-0.0704*x1*x9+0.03099*x2*x6-0.018*x2*x7+0.0208*x3*x8+0.121*x3*x9-0.00364*x5*x6+0.0007715*x5*x10-0.0005354*x6*x10+0.00121*x8*x11
    VCl = 0.74-0.61*x2-0.163*x3*x8+0.001232*x3*x10-0.166*x7*x9+0.227*x2**2
    VB = 10.58-0.674*x1*x2-1.95*x2*x8+0.02054*x3*x10-0.0198*x4*x10+0.028*x6*x10
    VD = 16.45-0.489*x3*x7-0.843*x5*x6+0.0432*x9*x10-0.0556*x9*x11-0.000786*x11**2
    vals = [L, F, Du, Dm, Dl, VCu, VCm, VCl, VB, VD]
    lims = [1.0, 4.01, 32.0, 32.0, 32.0, 0.32, 0.32, 0.32, 9.9, 15.69]
    g_constr = vals[idx] - lims[idx]
    return(g_constr)


def constr1(x):
    return vehicle_side_impact(x, 0)

def constr2(x):
    return vehicle_side_impact(x, 1)

def constr3(x):
    return vehicle_side_impact(x, 2)

def constr4(x):
    return vehicle_side_impact(x, 3)

def constr5(x):
    return vehicle_side_impact(x, 4)

def constr6(x):
    return vehicle_side_impact(x, 5)

def constr7(x):
    return vehicle_side_impact(x, 6)

def constr8(x):
    return vehicle_side_impact(x, 7)

def constr9(x):
    return vehicle_side_impact(x, 8)

def constr10(x):
    return vehicle_side_impact(x, 9)


def run_egra_problem(acquisition_function_name, problem_iter, ratio):


    ################################################################################
    """Make folder for plots"""

    folder_name = f"{acquisition_function_name}_example_{problem_iter}"
    os.makedirs(folder_name, exist_ok=True)

    ################################################################################
    """GET FUNCTIONS"""

    # Ndim = 2 #1 #10 #3 #2 #3 #1 #2
    # Nsamp = 8 #(Ndim+1)*(Ndim+2)//2 #8 #20 #3000 #2000 #1500 #1000 #500 #300 #200 #10*Ndim #6 #20 #8

    # Ntest = 10_000 #1000 #1000 # use fewer when plotting test error convergence
    n_scatter_gauss = 128 #32 #64 #256 #

    # # F = nonstationary_1d_hf
    # # # Fe = [nonstationary_1d_lf]
    # # # Fe = [uninformative_1d]
    # # lb = np.zeros([Ndim]) #0.0 # -2 + np.zeros([Ndim]) #np.array([-2, -2])
    # # ub = np.ones([Ndim]) #1.0 #  2 + np.zeros([Ndim]) #np.array([ 2,  2])
    # # 
    # # # Rosenbrock (nd)
    # # F = rosenbrock
    # # # Fe = [uninformative_nd_lf]
    # # lb = -2*np.ones([Ndim]) #0.0 # -2 + np.zeros([Ndim]) #np.array([-2, -2])
    # # ub = 2*np.ones([Ndim]) #1.0 #  2 + np.zeros([Ndim]) #np.array([ 2,  2])
    # # Nemulator = len(Fe)


    # # multimodal system
    # Ndim = 2 #1 #10 #3 #2 #3 #1 #2
    # Nsamp = 8 #(Ndim+1)*(Ndim+2)//2 #8 #20 #3000 #2000 #1500 #1000 #500 #300 #200 #10*Ndim #6 #20 #8
    # Fs = [g1, g2, g3]
    # lb = np.array([-3., 0.])
    # ub = np.array([7., 10.])
    # N_constr = len(Fs)
    # Ntest = 10_000
    # HF_EXPENSIVE = False
    # emulator_function_lists = N_constr*[[uninformative_nd_lf]]

    # # vehicle side impact
    # #Fs = [lambda x: vehicle_side_impact(x, idx) for idx in range(10)]
    # Fs = [constr1, constr2, constr3, constr4, constr5, constr6, constr7, constr8, constr9, constr10]
    # x_mu = np.array([0.5, 1.31, 0.5, 1.395, 0.875, 1.2, 0.4, 0.345, 0.192, 0.0, 0.0])
    # x_sig = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.006, 0.006, 10.0, 10.0])
    # lb = x_mu - 5*x_sig
    # ub = x_mu + 5*x_sig
    # N_constr = len(Fs)
    # Ndim = 11
    # Nsamp = (Ndim+1)*(Ndim+2)//2 #8 #20  #1000 #
    # Ntest = 10_000
    # HF_EXPENSIVE = False
    # emulator_function_lists = N_constr*[[uninformative_nd_lf]]

    # # 2D rosenbrock
    # Ndim = 2 #1 #10 #3 #2 #3 #1 #2
    # Nsamp = 8 #(Ndim+1)*(Ndim+2)//2 #8 #20 #3000 #2000 #1500 #1000 #500 #300 #200 #10*Ndim #6 #20 #8
    # Fs = [constr_rosen_2d]
    # lb = -2*np.ones([Ndim])
    # ub = 2*np.ones([Ndim])
    # N_constr = len(Fs)
    # Ntest = 10_000
    # HF_EXPENSIVE = False
    # emulator_function_lists = N_constr*[[uninformative_nd_lf]]

    # # D-dim Rosenbrock
    # Ndim = 20  #10  #1  #
    # Nsamp = (Ndim+1)*(Ndim+2)//2 #20 #1000
    # # Fs = [rosenbrock]
    # Fs = [constr_rosen_nd]
    # # Fe = [uninformative_nd_lf]
    # lb = -2*np.ones([Ndim]) #0.0 # -2 + np.zeros([Ndim]) #np.array([-2, -2])
    # ub = 2*np.ones([Ndim]) #1.0 #  2 + np.zeros([Ndim]) #np.array([ 2,  2])
    # N_constr = len(Fs)
    # Ntest = 10_000
    # HF_EXPENSIVE = False
    # # emulator_function_lists = N_constr*[[uninformative_nd_lf]]
    # univ_em = False
    # if univ_em:
    #     emulator_functions = []
    #     # univariate emulators
    #     for ii in range(Ndim):
    #         emulator = get_rosen_emulator(dim=ii, N_DIM=Ndim)
    #         emulator_functions.append(emulator)
    #     # sum emulator
    #     emulator_sum = return_sum_emulator(emulator_functions.copy())
    #     emulator_functions.append(emulator_sum)
    # else:
    #     # univariate emulators
    #     em_functions_u = []
    #     for ii in range(Ndim):
    #         emulator = get_rosen_emulator(dim=ii, N_DIM=Ndim)
    #         em_functions_u.append(emulator)
    #     em_functions_b = []
    #     # bivariate case
    #     for ii in range(Ndim-1):
    #         emulator = get_rosen_biv_emulator(dims=(ii, ii+1), N_DIM=Ndim)
    #         em_functions_b.append(emulator)
    #     em_sum_b = return_sum_emulator(em_functions_b.copy())
    #     # v0, no additional emulators (beats v2 for float64)
    #     emulator_functions = em_functions_u + em_functions_b
    #     # # v1, additional emulator to approximate true function
    #     # em_sum_u = return_sum_emulator(em_functions_u.copy())
    #     # em_sum = lambda x: em_sum_b(x) - em_sum_u(x)
    #     # emulator_functions = em_functions_u + em_functions_b + [em_sum]
    #     print(f"Number of emulators: {len(emulator_functions)}")
    # emulator_function_lists = [emulator_functions]

    # # Aircraft lift force
    # Ndim = 3
    # Nsamp = (Ndim+1)*(Ndim+2)//2
    # Ntest = 4000
    # subfolder2 = "GHV_34k_i_CGNS"  #"GHV_300k_i_CGNS"   #
    # out_name2  = "GHV_fgrid_coarse01"  #"GHV02_300k"   #
    # MINIMUM_LIFT = 25e6  #N
    # lift_constr = lambda x: fun3d_simulation_lift_force(
    #     x, out_name2, subfolder2).flatten() - MINIMUM_LIFT
    # Fs = [lift_constr]
    # lb = np.zeros(Ndim)
    # ub = np.ones(Ndim)
    # N_constr = len(Fs)
    # HF_EXPENSIVE = True
    # HF_NAME = "inv30k"
    # emulator_function_lists = N_constr*[[uninformative_nd_lf]]


    # Aircraft lift force multi-fidelity
    Ndim = 3
    Nsamp = (Ndim+1)*(Ndim+2)//2
    Ntest = 4000
    subfolder2 = "GHV_300k_i_CGNS"   #"GHV_34k_i_CGNS"  #
    out_name2  = "GHV02_300k"   #"GHV_fgrid_coarse01"  #
    MINIMUM_LIFT = 25e6  #N
    lift_constr = lambda x: fun3d_simulation_lift_force(
        x, out_name2, subfolder2).flatten() - MINIMUM_LIFT
    Fs = [lift_constr]
    lb = np.zeros(Ndim)
    ub = np.ones(Ndim)
    N_constr = len(Fs)
    HF_EXPENSIVE = True
    HF_NAME = "inv300k"   #"vis300k"   #

    # ---------- begin lf part
    filename_x = "X_test_E2NN_200_3D_samples"   #"X_test_E2NN_4000_3D_samples"   #
    filename_y = "Y_test_E2NN_200_3D_samples"   #"Y_test_inv30k_E2NN_4000_3D_samples"  #
    with open(filename_x, "rb") as infile:
        X_lf_train = pickle.load(infile)
        # X_lf_train = (ub-lb)*X_lf_train + lb
    with open(filename_y, "rb") as infile:
        y_lf_lift = pickle.load(infile)[0]
    # filename_train = 'ycl_cd_lf_inviscid_3d.pkl'
    # try:
    #     with open(filename_train, "rb") as infile:
    #         y_lf_cl_cd = pickle.load(infile)
    # except:
    #     out_name_lf = "GHV_fgrid_coarse01"
    #     subfolder_lf = "GHV_34k_i_CGNS"
    #     fun_i_ratio = lambda x: -fun3d_simulation_cl_cd(x, out_name_lf,  subfolder_lf)
    #     y_lf_cl_cd = fun_i_ratio(X_lf_train)
    #     with open(filename_train, "wb") as outfile:
    #         pickle.dump(y_lf_cl_cd, outfile)
    
    # construct gpr model and train on data
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF
    
    GPR = GaussianProcessRegressor(   # kernel = Matern()
            kernel=1.0*RBF(1.0), alpha=1e-10, optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=250) #25) #25
    
    GPR.fit(X_lf_train, y_lf_lift) # LD Ratio
    
    LF_GPR = lambda x : GPR.predict(x, return_std=False).flatten()
    emulator_functions = [LF_GPR]
    # ---------- end lf part
    emulator_function_lists = [emulator_functions]   #N_constr*[[uninformative_nd_lf]]



    # emulator_function_lists = N_constr*[[uninformative_nd_lf]]
    e2nn_ensembles = N_constr*[None]
    x_scale_objs = N_constr*[None]
    y_scale_objs = N_constr*[None]

    fourier_factors_1L = N_constr*[1.0]
    fourier_factors_2L = N_constr*[1.0]


    N_COPY_ARCH = 2  # number of copies of each NN architecture
    ensemble_settings = N_COPY_ARCH*[
        {"n_layers": 1, "activation": "fourier", "freq_factor": 1.0},
        {"n_layers": 1, "activation": "fourier", "freq_factor": 1.1},
        {"n_layers": 1, "activation": "fourier", "freq_factor": 1.2},
        {"n_layers": 1, "activation": "swish", "freq_factor": np.nan},
    
        {"n_layers": 2, "activation": "fourier", "freq_factor": 1.0},
        {"n_layers": 2, "activation": "fourier", "freq_factor": 1.1},
        {"n_layers": 2, "activation": "fourier", "freq_factor": 1.2},
        {"n_layers": 2, "activation": "swish", "freq_factor": np.nan},
    ]


    azim_2d = -120 #-150 #-60 # -60 (default) #
    elev_2d = 20 # 30 (default) #
    # ax.azim = -60
    # ax.dist = 10
    # ax.elev = 30

    ################################################################################
    """GET TRAINING DATA"""

    if Ndim == 1:
        X_train = np.linspace(0, 1, Nsamp) #np.array([0, 0.5, 1]) # this should be 2D
        X_train = X_train.reshape(-1, 1) # make 2D
    else:
        # X_train = lhs(Ndim, samples=Nsamp, criterion="maximin", iterations=1_000)#100_000)#20_000)#20)
        # outfile_name = f"x_train_{Nsamp}_pts_{Ndim}D.pkl"
        # with open(outfile_name, "wb") as outfile:
        #     pickle.dump(X_train, outfile)

        filename_train = f"x_train_{Nsamp}_pts_{Ndim}D_{problem_iter}.pkl" #f"x_train_{Nsamp}_pts_{Ndim}D.pkl" #
        try:
            # load X
            with open(filename_train, "rb") as infile:
                X_train = pickle.load(infile)
        except:
            # Get samples using LHS
            print("Generating training samples using LHS")
            X_train = lhs(Ndim, samples=Nsamp, criterion="maximin", iterations=1_000) 
            #100_000) #
            print("DOE of training points complete")

            # save X
            with open(filename_train, "wb") as outfile:
                pickle.dump(X_train, outfile)
            # END Get samples using LHS

    X_train_raw = (ub-lb)*X_train + lb
    # Y_train_raw = F(X_train_raw)
    # # reshape to 2D for scaling
    # Y_train_raw = Y_train_raw.reshape(-1, 1)



    ################################################################################
    """GET TEST DATA"""

    if Ndim == 1:
        X_test = np.linspace(0, 1, 201)
        X_test = X_test.reshape(-1, 1) # make 2D
        X_test_raw = (ub-lb)*X_test + lb
        # Y_test_raw = F(X_test_raw)

    elif Ndim == 2:
        X1_test_unscaled = np.linspace(lb[0], ub[0], 65)#33)#17)
        X2_test_unscaled = np.linspace(lb[1], ub[1], 65)#33)#17)
        (X1_test_unscaled, X2_test_unscaled) = np.meshgrid(X1_test_unscaled, X2_test_unscaled)

        test_plot_shape = X1_test_unscaled.shape

        X1_test_unscaled = np.reshape(X1_test_unscaled, (-1, 1)) # 2D column
        X2_test_unscaled = np.reshape(X2_test_unscaled, (-1, 1)) # 2D column

        X_test_raw = np.concatenate( ( X1_test_unscaled, X2_test_unscaled ), axis=1) # fuse
        X1_test_unscaled, X2_test_unscaled = X1_test_unscaled.flatten(), X2_test_unscaled.flatten() # 1D

        # Y_test_raw = F(X_test_raw)

    else:
        # X_test = lhs(Ndim, samples=Ntest, criterion="maximin", iterations=1000) #10_000)
        # X_test_raw = (ub-lb)*X_test + lb
        # Y_test_raw = F(X_test_raw)

        filename_test = f"X_test_E2NN_{Ntest}_{Ndim}D_samples"
        try:
            # load X test
            with open(filename_test, "rb") as infile:
                X_test = pickle.load(infile)
        except:
            # Get samples using LHS
            print("Generating test samples using LHS")
            X_test = lhs(Ndim, samples=Ntest, criterion="maximin", iterations=1000)#20000)#20)
            print("DOE of test points complete")

            # save X test
            with open(filename_test, "wb") as outfile:
                pickle.dump(X_test, outfile)
            # END Get samples using LHS
        X_test_raw = (ub-lb)*X_test + lb
        # Y_test_raw = F(X_test_raw)





    ################################################################################
    ################################################################################
    """                           DO EGRA OPTIMIZATION                           """
    ################################################################################
    ################################################################################


    Xs_train_raw = N_constr*[np.zeros([0, Ndim])]
    Ys_train_raw = N_constr*[np.array([])]

    Xs_test_raw = N_constr*[X_test_raw]
    # Ys_test_raw = [Fs[ii](Xs_test_raw[ii]) for ii in range(N_constr)]

    if not HF_EXPENSIVE:
        Ys_test_raw = [Fs[ii](Xs_test_raw[ii]) for ii in range(N_constr)]
    else:
        filename_test = f"Y_test_{HF_NAME}_E2NN_{Ntest}_{Ndim}D_samples"
        try:
            # load values
            with open(filename_test, "rb") as infile:
                Ys_test_raw = pickle.load(infile)
                Ys_test_raw = [val.flatten() for val in Ys_test_raw]
        except:
            Ys_test_raw = [Fs[ii](Xs_test_raw[ii]) for ii in range(N_constr)]
            # save Y test
            with open(filename_test, "wb") as outfile:
                pickle.dump(Ys_test_raw, outfile)

    # Tolerances
    if "EGRA" in acquisition_function_name:
        EFF_tol = 1e-5  #1e-2  #
    else:
        EFF_tol = 1e-6  #1e-3  # This depends on the data scale
    N_STEP = 1
    MAX_STEP = 300 #1000 #np.inf
    STEPS_TO_CONVERGE = 3 #1 #
    converged_steps = 0

    new_X = N_constr*[X_train_raw]


    # GPR_models = N_constr*[None]
    # x_stand_objs = N_constr*[None]



    EFFs = []
    NRMSEs = []
    SumLogLikelihoods = []
    Precisions = [] # TP/(TP+FP)
    Recalls = [] # TP/(TP+FN)
    F1_scores = [] # 2*P*R/(P+R)
    MCCs = []
    total_samples = []
    # min_idx = np.argmin(Y_train_raw)
    # Yopts = [np.min(Y_train_raw)]
    # Xopts = [X_train_raw[min_idx,:]]

    adaptive_sampling_converged = False

    while not adaptive_sampling_converged:

        # for each constraint
        for c in range(N_constr):
            # evaluate new point(s) for constraint
            if new_X[c].size: # if array of new points not empty
                new_Yc = Fs[c](new_X[c])
                Xs_train_raw[c] = np.vstack([Xs_train_raw[c], new_X[c]])
                Ys_train_raw[c] = np.hstack([Ys_train_raw[c], new_Yc])
                # new_X[c] = np.array([]) # replace with empty array

            ######################### build meta-model for constraint
            # GPR_models[c], x_stand_objs[c] = fit_gpr(Xs_train_raw[c], Ys_train_raw[c])
            
            frequencies_good = False
            while not frequencies_good:
                (e2nn_ensembles[c], x_scale_objs[c], y_scale_objs[c], \
                fail_1L, fail_2L, *_) = build_ensemble(
                    emulator_function_lists[c], Xs_train_raw[c], Ys_train_raw[c].reshape(-1,1), lb, ub,
                    ensemble_settings=ensemble_settings, 
                    fourier_factor_1L=fourier_factors_1L[c], fourier_factor_2L=fourier_factors_2L[c],
                )
                
                if fail_1L:
                    fourier_factors_1L[c] *= 1.2 #+= 1 #*= 2 #
                if fail_2L:
                    fourier_factors_2L[c] *= 1.2 #+= 1 #*= 2 #

                if not (fail_1L or fail_2L):
                    frequencies_good = True

        
        #  # plot actual vs predicted for each constraint
        #  for c in range(N_constr):
        #      
        #      Y_gpr_test, Sig_gpr_test = predict_gpr(GPR_models[c], Xs_test_raw[c], x_stand_objs[c])
        #      Y_test_raw = Ys_test_raw[c]
        #      gpr_test_nrmse = NRMSE(Y_gpr_test, Y_test_raw)
        #      # gpr_test_sum_log_likelihood = sum_log_likelihood_norm(Y_test_raw, Y_gpr_test, Sig_gpr_test)

        #      yerr=2*Sig_gpr_test
        #      fig = plt.figure(figsize=(6.4, 4.8))
        #      ax = fig.add_subplot(111)
        #      ax.errorbar(Y_test_raw.flatten(), Y_gpr_test.flatten(), yerr=yerr.flatten(), fmt = "none", color = "g", marker=None, capsize = 4, elinewidth=0.5, alpha = 0.2, label="$\pm\sigma$ error bars", zorder=1) #capthick= , elinewidth=
        #      ax.scatter(Y_test_raw, Y_gpr_test, c="b", marker=".", label="test predictions", zorder=2)
        #      ax.plot(Y_test_raw, Y_test_raw, "k-", linewidth=1, label = "true", zorder=3)
        #      ax.set_title(f"GPR fit (Test NRMSE={gpr_test_nrmse})")
        #      ax.set_xlabel("Actual", fontsize=18)
        #      ax.set_ylabel("Predicted", fontsize=18)
        #      ax.legend(loc="lower right")
        #      plt.savefig(f"{folder_name}/Actual vs Predicted GPR for constraint {c} (step {N_STEP}).png", dpi=300)
        #      # plt.show()
        #      plt.close(fig)

        # GET OTHER ERROR METRICS

        # get predictions
        Yc_preds = N_constr*[None]
        for c in range(N_constr):
            # Yc_pred, __ = predict_gpr(GPR_models[c], Xs_test_raw[c], x_stand_objs[c])
            Yc_pred, *_ = predict_ensemble(Xs_test_raw[c], e2nn_ensembles[c], emulator_function_lists[c], x_scale_objs[c], y_scale_objs[c])
            Yc_preds[c] = Yc_pred.flatten()
        # 
        


        print("Ys_test_raw[0].shape: ", Ys_test_raw[0].shape)
        print("Yc_preds[0].shape: ", Yc_preds[0].shape)

        Ys_true = np.vstack(Ys_test_raw)
        Ys_feasible_true = Ys_true<=0
        Ys_feasible_true = np.all(Ys_feasible_true, axis=0)

        Ys_pred = np.vstack(Yc_preds)
        Ys_feasible_pred = Ys_pred<=0
        Ys_feasible_pred = np.all(Ys_feasible_pred, axis=0)

        TP = np.sum(Ys_feasible_true & Ys_feasible_pred)
        FP = np.sum(~Ys_feasible_true & Ys_feasible_pred)
        FN = np.sum(Ys_feasible_true & ~Ys_feasible_pred)
        TN = np.sum(~Ys_feasible_true & ~Ys_feasible_pred)
        
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        # Calculate F1 score
        if not TP: # avoid NaN from division by 0
            F1_score = 0.0
            Precision = 0.0
            Recall = 0.0
        else:
            F1_score = 2*Precision*Recall/(Precision+Recall)
        # Calculate MCC
        if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) == 0:
            MCC = 0.0
        else:
            MCC = (TP*TN - FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5

        Precisions.append(Precision) # TP/(TP+FP)
        Recalls.append(Recall) # TP/(TP+FN)
        F1_scores.append(F1_score) # 2*P*R/(P+R)
        MCCs.append(MCC)
        
        Nsamp = [Ys_train_raw[c].size for c in range(N_constr)]
        net_samp = int(np.sum(Nsamp))
        total_samples.append(net_samp)

        ################################################################################
        """ ACQUISITION TO GET NEW SAMPLE for each constraint """

        print("Beginning acquisition optimization")

        # # EFF ACQUISITION
        # if acquisition_function_name == "EGRA":
        #     EFF = EGRA
        # if acquisition_function_name == "EGRA4":
        #     EFF = EGRA
        # elif acquisition_function_name == "ALT":
        #     EFF = ALT
        # elif acquisition_function_name == "EMI":
        #     EFF = EMI
        # elif acquisition_function_name == "EMI4":
        #     EFF = EMI
        # elif acquisition_function_name == "EMRI":
        #     EFF = EMRI
        # elif acquisition_function_name == "EMRI4":
        #     EFF = EMRI
        # else:
        #     raise ValueError(f"Unknown acquisition function name: {acquisition_function_name}")

        # obj_fn = lambda x : -EFF(x, GPR_models, x_stand_objs)[0]
        EFF = EMI
        obj_fn = lambda x: -EFF(x, e2nn_ensembles, emulator_function_lists, x_scale_objs, y_scale_objs)[0]

        x_opt, f_opt, *__ = global_optimization(obj_fn, lb, ub, Ndim, n_scatter_init = 1_000,  n_local_opts = 10, previous_local_xopt = np.array([]), n_scatter_gauss = n_scatter_gauss)
        x_opt = x_opt.reshape(1, -1) # make x_new 2D

        EFF_opt, c_idx = EFF(x_opt, e2nn_ensembles, emulator_function_lists, x_scale_objs, y_scale_objs)
        EFFs.append(float(EFF_opt.flatten()[0])) # sometimes EFF_opt is a numpy array, sometimes it is a float
        # (probably caused issues with other optimizer, might want to fix underlying issue)
        # (probably an issue of dimensionality)

        eff_vals = np.zeros(N_constr)
        for c in range(N_constr):
            eff_vals[c], __ = EFF(x_opt, [e2nn_ensembles[c]], [emulator_function_lists[c]], [x_scale_objs[c]], [y_scale_objs[c]])

        # # EI ACQUISITION (REQUIRES SINGLE GPR_MODEL)
        # obj_fn = lambda x: - EI_acquisition(x, GPR_MODEL, x_stand_obj)
        # x_opt, f_opt, *__ = global_optimization(obj_fn, lb, ub, Ndim, n_scatter_init = 1_000,  n_local_opts = 10, previous_local_xopt = np.array([]), n_scatter_gauss = n_scatter_gauss)


        # # UNCERTAINTY ACQUISITION (CAN SAMPLE MULTIPLE)
        # def get_uncertainty(X_unscaled, GPR_MODEL, x_stand_obj):
        #     __, Sig_gpr_test = predict_gpr(GPR_MODEL, X_unscaled, x_stand_obj)
        #     return(Sig_gpr_test)
        # obj_fn = lambda x : -get_uncertainty(x, GPR_MODEL, x_stand_obj)
        # 
        # x_opts = np.zeros(N_constr)
        # f_opts = np.zeros(N_constr)
        # for c in range(N_constr):
        #     x_opts[c], f_opts[c], *__ = global_optimization(obj_fn, lb, ub, Ndim, n_scatter_init = 1_000,  n_local_opts = 10, previous_local_xopt = np.array([]), n_scatter_gauss = n_scatter_gauss)
        # opt_idx = np.argmin(f_opts)
        # x_opt = x_opts[opt_idx]
        # f_opt = f_opts[opt_idx]


        # # NEED CHANGES BECAUSE 3 DIFFERENT NRMSES AND SUMLOGLIKELIHOODS
        # NRMSEs.append(gpr_test_nrmse)
        # # np array when 1D, scalar otherwise
        # SumLogLikelihoods.append(float(gpr_test_sum_log_likelihood))

        ################################################################################
        """ PLOTTING ACQUISITION """

        # 2d plots
        if Ndim == 2:

            X_plot = X_test_raw
            EFF_plot = -obj_fn(X_plot)

            def fix_legend_crash(surf):
                # Fixes a bug where the legend tries to call the surface color and crashes
                surf._facecolors2d=surf._facecolor3d
                surf._edgecolors2d=surf._edgecolor3d
                return(surf)

            X1_test_unscaled = X_test_raw[:,0]#.flatten()
            X2_test_unscaled = X_test_raw[:,1]#.flatten()
            tri = mtri.Triangulation(X1_test_unscaled, X2_test_unscaled)

            # plot acquisition
            fig = plt.figure(figsize=(6.4, 4.8))
            ax = fig.add_subplot(111, projection="3d")
            ax.grid(False)
            surf = ax.plot_trisurf(X1_test_unscaled, X2_test_unscaled, EFF_plot, \
                            triangles=tri.triangles, color="k", alpha = 0.5, linewidth=0.2, label = f"{acquisition_function_name}")
            surf = fix_legend_crash(surf)

            ax.scatter(x_opt[:,0], x_opt[:,1], EFF_opt, c="r", marker="o", label="optimum")
            # ax.scatter(X_EFF_opts[:,0], X_EFF_opts[:,1], EFF_local_opts, c="g", marker="x", label="opts final")
            ax.scatter(X_train_raw[:,0], X_train_raw[:,1], np.zeros(np.shape(X_train_raw[:,0])), c="gray", marker="o", label="data locations")

            ax.set_title(f"Expected Improvement (step {N_STEP})")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel(f"{acquisition_function_name}")
            ax.legend(loc="upper right")
            # ax.azim = azim_2d
            # ax.elev = elev_2d

            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("w")
            ax.yaxis.pane.set_edgecolor("w")
            ax.zaxis.pane.set_edgecolor("w")

            plt.savefig(f"{folder_name}/Acquisition ({N_STEP} steps).png", dpi=300)
            # plt.show()
            plt.close(fig)


        # SAVE DATA
        for c in range(N_constr):
            doc1 = open(f"{folder_name}/sample data step {N_STEP} constr{c}.txt","w")
            # doc1.write(f"Xs_train_raw[c]: {Xs_train_raw[c]}\n")
            # doc1.write(f"Ys_train_raw[c]: {Ys_train_raw[c]}\n")
            doc1.write(f"EFFs: {EFFs}\n")
            doc1.write(f"Precisions: {Precisions}\n")
            doc1.write(f"Recalls: {Recalls}\n")
            doc1.write(f"F1_scores: {F1_scores}\n")
            doc1.write(f"MCCs: {MCCs}\n")
            doc1.write(f"total_samples: {total_samples}\n")
            # doc1.write("Yopts: "+str(Yopts)+"\n")
            # doc1.write("NRMSEs: "+str(NRMSEs)+"\n")
            # doc1.write("SumLogLikelihoods: "+str(SumLogLikelihoods)+"\n")
            doc1.close()


        ################################################################################
        """ ADD NEW SAMPLE UNLESS CONVERGED"""
        c_idx = int(c_idx[0]) # was numpy array
        data_range = np.max(Ys_train_raw[c_idx])-np.min(Ys_train_raw[c_idx])

        doc1 = open(f"{folder_name}/sample data step {N_STEP} constr{c}.txt","a")
        doc1.write(f"data_range: {data_range}\n")
        doc1.close()

        # check tolerance
        if EFF_opt < EFF_tol*data_range:
            converged_steps += 1
        else:
            converged_steps = 0

        # Check convergence
        if converged_steps >= STEPS_TO_CONVERGE or N_STEP >= MAX_STEP:
            adaptive_sampling_converged = True
        else:
            N_STEP += 1

        # Select new evaluations
        for c, eff_val in enumerate(eff_vals):
            data_range = np.max(Ys_train_raw[c])-np.min(Ys_train_raw[c])
            if eff_val < EFF_tol*data_range or ratio*eff_val<EFF_opt:
                new_X[c] = np.array([])
            else:
                new_X[c] = x_opt

            # y_new = F(x_new)
            # X_train_raw = np.vstack([X_train_raw, x_new])
            # Y_train_raw = np.vstack([Y_train_raw, y_new])

            # min_idx = np.argmin(Y_train_raw)
            # Yopts.append(np.min(Y_train_raw))
            # Xopts.append(X_train_raw[min_idx,:])


        # plot constraints and new points

        # The first time spines are removed it doesn't work. Call to "warm-start"
        fig = plt.figure()
        
        plt.close(fig)

        if Ndim == 2:
            from numpy import ma

            X_plot1 = X_test_raw[:,0].reshape(test_plot_shape)
            X_plot2 = X_test_raw[:,1].reshape(test_plot_shape)

            Yc_preds = N_constr*[None]
            for c in range(N_constr):
                Yc_pred, *_ = predict_ensemble(Xs_test_raw[c], e2nn_ensembles[c], emulator_function_lists[c], x_scale_objs[c], y_scale_objs[c]) #GPR_models[c], Xs_test_raw[c], x_stand_objs[c])
                Yc_pred = Yc_pred.reshape(test_plot_shape)
                Yc_preds[c] = Yc_pred

            plt.rcParams["axes.spines.right"] = False
            plt.rcParams["axes.spines.top"] = False
            # plt.rcParams["axes.spines.left"] = False
            # plt.rcParams["axes.spines.bottom"] = False

            fig = plt.figure()#figsize=(6.4, 4.8))
            
            # plt.axis("equal")
            plt.axis("square")

            pad = 0.2
            plt.xlim([lb[0]-pad, ub[0]+pad])
            plt.ylim([lb[1]-pad, ub[1]+pad])
            
            # plt.axis("off")

            # plt.grid(False)
            colors = ["r", "b", "g"]
            markers = [mpl.markers.MarkerStyle("o", fillstyle="none"),"x","+"] #["1","2","3"] #["x","+","."] #["o", "+", "x"] #["o","P","X"] #


            # shade predicted infeasible
            shader = np.zeros(test_plot_shape)
            for Yc_pred in Yc_preds:
                Gshade = ma.masked_where(Yc_pred<0, shader)
                plt.contourf(X_plot1, X_plot2, Gshade, alpha=0.5, colors="gray")

            # true contour (black) 
            for Y_test_raw in Ys_test_raw:
                plt.contour(X_plot1, X_plot2, Y_test_raw.reshape(test_plot_shape), [0], linewidths=2, colors="k")

            # predicted constraints (color)
            for c, Yc_pred in enumerate(Yc_preds):
                plt.contour(X_plot1, X_plot2, Yc_pred, [0], linewidths=2, colors=colors[c])

            # current and acquisition points
            for c in range(N_constr):

                plt.scatter(Xs_train_raw[c][:,0], Xs_train_raw[c][:,1], c="k", marker=markers[c])#, label="data locations") #
                
                if len(new_X[c]):
                    plt.scatter(new_X[c][:,0], new_X[c][:,1], c=colors[c], marker=markers[c], zorder=100+c)#, label="new") #


            # plt.title(f"Constraint contours (step {N_STEP})")
            plt.xlabel("$x_1$")
            plt.ylabel("$x_2$")
            # plt.legend(loc="upper right")
            plt.savefig(f"{folder_name}/Constraint plottings (step {N_STEP}).png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            plt.rcParams["axes.spines.right"] = True
            plt.rcParams["axes.spines.top"] = True





    ################################################################################
    """ PLOT OPTIMIZATION RESULTS"""

    # Xopts = np.array(Xopts)
    # Yopts = np.array(Yopts)
    EFFs = np.array(EFFs)
    Niters = len(EFFs)
    iters = np.array(range(1, Niters+1))

    Precisions = np.array(Precisions)
    Recalls = np.array(Recalls)
    F1_scores = np.array(F1_scores)
    MCCs = np.array(MCCs)
    total_samples = np.array(total_samples)

    filename = f"plotting_data_{acquisition_function_name}_iter_{problem_iter}.pkl"

    data = {"EFFs": EFFs,
            "iters": iters,
            "total_samples": total_samples,
            "Precisions": Precisions,
            "Recalls": Recalls,
            "F1_scores": F1_scores,
            "MCCs": MCCs,
    }
    with open(filename, "wb") as outfile:
        pickle.dump(data, outfile)

    # # Optimal xs
    # fig1 = plt.figure(figsize=(6.4, 4.8))
    # ax = fig1.add_subplot(111)
    # for ii in range(Ndim):
    #     ax.plot(iters, Xopts[:,ii], "-", label = "Opt $X_"+str(ii)+"$", linewidth=2)
    # ax.set_title("X history")
    # ax.set_xlabel("iteration")
    # ax.set_ylabel("Optimums")
    # ax.legend(loc="upper left")
    # plt.savefig("History X.png", dpi=300)
    # plt.draw()


    # # Optimal ys
    # fig2 = plt.figure(figsize=(6.4, 4.8))
    # ax = fig2.add_subplot(111)
    # ax.plot(iters, Yopts, "k.-", label = "Yopt", linewidth=2)
    # ax.set_title("Y history")
    # ax.set_xlabel("iteration")
    # ax.set_ylabel("Optimums")
    # ax.legend(loc="upper right")
    # plt.savefig("History Y.png")
    # plt.draw()


    # # Optimal ys log
    # if np.all(Yopts>0):
    #     fig3 = plt.figure(figsize=(6.4, 4.8))
    #     ax = fig3.add_subplot(111)
    #     ax.plot(iters, Yopts, "k.-", label = "Yopt", linewidth=2)
    #     ax.set_title("Y history")
    #     ax.set_xlabel("iteration")
    #     ax.set_ylabel("Optimums")
    #     ax.legend(loc="upper right")
    #     plt.yscale("log")
    #     plt.savefig("History Y log.png")
    #     plt.draw()



    # EFF
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(iters, EFFs, "r.-", label = f"{acquisition_function_name}", linewidth=2)
    ax.set_title(f"{acquisition_function_name} history")
    ax.set_xlabel("iteration")
    ax.set_ylabel(f"{acquisition_function_name}")
    ax.legend(loc="upper right")
    plt.savefig(f"{folder_name}/History {acquisition_function_name}.png")
    plt.close(fig)


    # EFF log
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(iters, EFFs, "r.-", label = f"{acquisition_function_name}", linewidth=2)
    ax.set_title(f"{acquisition_function_name} history")
    ax.set_xlabel("iteration")
    ax.set_ylabel(f"{acquisition_function_name}")
    ax.legend(loc="upper right")
    plt.yscale("log")
    plt.savefig(f"{folder_name}/History {acquisition_function_name} log.png")
    plt.close(fig)



    # Precision
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(total_samples, Precisions, "r.-", label = "Precision", linewidth=2)
    ax.set_title("Precision history")
    ax.set_xlabel("constraint evaluations")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower right")
    plt.savefig(f"{folder_name}/History Precision.png")
    plt.close(fig)

    # Recall
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(total_samples, Recalls, "g.-", label = "Recall", linewidth=2)
    ax.set_title("Recall history")
    ax.set_xlabel("constraint evaluations")
    ax.set_ylabel("Recall")
    ax.legend(loc="lower right")
    plt.savefig(f"{folder_name}/History Recall.png")
    plt.close(fig)

    # F1 score
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(total_samples, F1_scores, "k.-", label = "F1 score", linewidth=2)
    ax.set_title("F1 score history")
    ax.set_xlabel("constraint evaluations")
    ax.set_ylabel("F1 score")
    ax.legend(loc="lower right")
    plt.savefig(f"{folder_name}/History F1 score.png")
    plt.close(fig)

    # Matthews Correlation Coefficient (MCC)
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(total_samples, MCCs, "k.-", label = "MCC", linewidth=2)
    ax.set_title("MCC history")
    ax.set_xlabel("constraint evaluations")
    ax.set_ylabel("MCC")
    ax.legend(loc="lower right")
    plt.savefig(f"{folder_name}/History MCC.png")
    plt.close(fig)


    # # NRMSE
    # fig6 = plt.figure(figsize=(6.4, 4.8))
    # ax = fig6.add_subplot(111)
    # ax.plot(iters, NRMSEs, "g.-", label = "NRMSE", linewidth=2)
    # ax.set_title("NRMSE history")
    # ax.set_xlabel("iteration")
    # ax.set_ylabel("NRMSE")
    # ax.legend(loc="upper right")
    # plt.savefig("History NRMSE.png")
    # plt.draw()


    # # NRMSE log
    # fig7 = plt.figure(figsize=(6.4, 4.8))
    # ax = fig7.add_subplot(111)
    # ax.plot(iters, NRMSEs, "g.-", label = "NRMSE", linewidth=2)
    # ax.set_title("NRMSE history")
    # ax.set_xlabel("iteration")
    # ax.set_ylabel("NRMSE")
    # ax.legend(loc="upper right")
    # plt.yscale("log")
    # plt.savefig("History NRMSE log.png")
    # plt.draw()

    return(None)

problem_iters = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# problem_iters = [1,2,3,4,5,6,7,8,9,10] #[6,7,8,9,10]
# problem_iters = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,\
#                  17,18,19,20,21,22,23,24,25,26,27,28,29,30,\
#                  31,32]
# problem_iters = [25,26,27,28,29,30,31,32]
#folder_name_root = "ALT_example" #"EGRA_example" #"EMI_example" #"EMI_adv_example" #
# acquisition_function_name = "EGRA"  #"EMRI"  #"EMI"  #"ALT"  #

acquisition_function_names = ["EMI_E2NN"]
ratio = 4.0   # sample all within factor of 4

# acquisition_function_names = ["EMI_E2NN_safe"]
# ratio = np.inf   # sample all nonconverged

if __name__ == "__main__":
    for acquisition_function_name in acquisition_function_names:
        for problem_iter in problem_iters:
            run_egra_problem(acquisition_function_name, problem_iter, ratio=ratio)


print("\nProgram completed successfully")






#
