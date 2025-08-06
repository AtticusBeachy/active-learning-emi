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


################################################################################
""" USER SELECTED OPTIONS """

problem_iters = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
ratio = 4.0   # sample all within factor of 4
acquisition_function_names = ["E2NN-EMI"]

# ratio = np.inf   # sample all nonconverged
# acquisition_function_names = ["EMI_E2NN_safe"]

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



def run_active_learning(acquisition_function_name, problem_iter, ratio):


    ################################################################################
    """Make folder for plots"""

    folder_name = f"{acquisition_function_name}_example_{problem_iter}"
    os.makedirs(folder_name, exist_ok=True)

    n_scatter_gauss = 128 #32 #64 #256 #

    ################################################################################
    """PROBLEM SELECTION (UNCOMMENT THE DISIRED ONE)"""

    # (1) 2D rosenbrock
    Ndim = 2 #1 #10 #3 #2 #3 #1 #2
    Nsamp = 8 #(Ndim+1)*(Ndim+2)//2 #
    Fs = [constr_rosen_2d]
    lb = -2*np.ones([Ndim])
    ub = 2*np.ones([Ndim])
    N_constr = len(Fs)
    Ntest = 10_000
    HF_EXPENSIVE = False
    emulator_function_lists = N_constr*[[uninformative_nd_lf]]

    # # (2) multimodal system
    # Ndim = 2 #1 #10 #3 #2 #3 #1 #2
    # Nsamp = 8 #(Ndim+1)*(Ndim+2)//2 #8 #20 #
    # Fs = [g1, g2, g3]
    # lb = np.array([-3., 0.])
    # ub = np.array([7., 10.])
    # N_constr = len(Fs)
    # Ntest = 10_000
    # HF_EXPENSIVE = False
    # emulator_function_lists = N_constr*[[uninformative_nd_lf]]

    # # (3) Aircraft lift force
    # Ndim = 3
    # Nsamp = (Ndim+1)*(Ndim+2)//2
    # Ntest = 4000
    # subfolder2 = "GHV_34k_i_CGNS"  #"GHV_300k_i_CGNS"   #
    # out_name2  = "GHV_fgrid_coarse01"  #"GHV02_300k"   #
    # MINIMUM_LIFT = 1e4  #25e6  #N
    # lift_constr = lambda x: fun3d_simulation_lift_force(
    #     x, out_name2, subfolder2).flatten() - MINIMUM_LIFT
    # Fs = [lift_constr]
    # lb = np.zeros(Ndim)
    # ub = np.ones(Ndim)
    # N_constr = len(Fs)
    # HF_EXPENSIVE = True
    # HF_NAME = "inv30k"
    # emulator_function_lists = N_constr*[[uninformative_nd_lf]]


    ################################################################################
    """ NEURAL NETWORK SETUP"""

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
        #filename_test = f"Y_test_{HF_NAME}_E2NN_{Ntest}_{Ndim}D_samples"
        filename_test = f"Y_test_E2NN_{Ntest}_{Ndim}D_samples"
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


        # get predictions
        Yc_preds = N_constr*[None]
        for c in range(N_constr):
            # Yc_pred, __ = predict_gpr(GPR_models[c], Xs_test_raw[c], x_stand_objs[c])
            Yc_pred, *_ = predict_ensemble(Xs_test_raw[c], e2nn_ensembles[c], emulator_function_lists[c], x_scale_objs[c], y_scale_objs[c])
            Yc_preds[c] = Yc_pred.flatten()

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

    return(None)



################################################################################
"""GAUSSIAN PROCESS MODELS"""

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

if __name__ == "__main__":
    for acquisition_function_name in acquisition_function_names:
        for problem_iter in problem_iters:
            run_active_learning(acquisition_function_name, problem_iter, ratio=ratio)


print("\nProgram completed successfully")


