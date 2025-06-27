import matplotlib.pyplot as plt
import numpy as np
import pickle

# Decide what to plot

# problem_iters = [1,2,3,4,5,6,7,8,9,10]
# acquisition_function_names = ["EGRA", "EMI", "EMRI", "EMI4", "EGRA4", "EMRI4", "sEMI"] #  "sALT", "sEGRA", "sEMRI", "ALT"


# problem_iters = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]  #[1]  #
# acquisition_function_names = ["EGRA", "EMI", "EMRI"]


# problem_iters = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# acquisition_function_names = ["EGRA4", "EMI4", "EMRI4"]


problem_iters = [1, 2, 3, 4, 5]  #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]  #
acquisition_function_names = ["EGRA", "EMI", "EMI_E2NN"]  #, "EMRI4"]

# problem_iters = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,\
#                  21,22,23,24,25,26,27,28,29,30,31,32]
# acquisition_function_names = ["EGRA", "EMI", "EMRI", "EMI4", "EGRA4", "EMRI4"] #, "sEMI"] #  "sALT", "sEGRA", "sEMRI", "ALT"

# acquisition_function_names = ["EGRA", "sEGRA"] #  "sALT", "sEGRA", "sEMI", "sEMRI"
# acquisition_function_names = ["EMI", "sEMI"] #  "sALT", "sEGRA", "sEMI", "sEMRI"
# acquisition_function_names = ["EMRI", "sEMRI"] #  "sALT", "sEGRA", "sEMI", "sEMRI"
# acquisition_function_names = ["sEGRA", "sEMI", "sEMRI"] #  "sALT", "sEGRA", "sEMI", "sEMRI", "ALT"
# problem_iters = [1,2,3,4,5,6,7,8,9,10]
# problem_iters = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,\
#                  21,22,23,24,25,26,27,28,29,30,31,32]
# problem_iters = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# problem_iters = [1,2]


# Decide colors

RED = '#D7191C'
BLUE = '#2C7BB6'

colors = {
    "EGRA": "r",
    "EMI": "b",
    "EMRI": "k",
    "EGRA4": "tab:orange",
    "EMI4": "g",
    "EMRI4": "tab:gray",
    "EMI_E2NN": "darkviolet",
    # "c", "tab:purple",
}


linestyles = {
    "EGRA": "-",
    "EMI": "-",
    "EMRI": "-",
    "EGRA4": "-",
    "EMI4": "-",
    "EMRI4": "-",
    "EMI_E2NN": "-",
}



# Extract pickled results

iters_dict = {}
total_samples_dict = {}
mccs_dict = {}


for acquisition_function_name in acquisition_function_names:
    
    iters_list = []
    total_samples_list = []
    mccs_list = []

    for problem_iter in problem_iters:
        filename = f"plotting_data_{acquisition_function_name}_iter_{problem_iter}.pkl"
    
        # during plotting
        with open(filename, "rb") as infile:
            #print(f"{type(infile)=}")
            #print(f"{infile=}")
            data = pickle.load(infile)
            #print(f"{type(data)=}")
            #print(f"{data=}")
        EFFs = data["EFFs"]
        iters = data["iters"]
        total_samples = data["total_samples"]
        Precisions = data["Precisions"]
        Recalls = data["Recalls"]
        F1_scores = data["F1_scores"]
        MCCs = data["MCCs"]

        iters_list.append(iters)
        total_samples_list.append(total_samples)
        mccs_list.append(MCCs)

    iters_dict[acquisition_function_name] = iters_list
    total_samples_dict[acquisition_function_name] = total_samples_list
    mccs_dict[acquisition_function_name] = mccs_list





############################################################################
""" Individual runs only """

plt.figure()

for acquisition_function_name in acquisition_function_names:

    iters_list = iters_dict[acquisition_function_name]
    total_samples_list = total_samples_dict[acquisition_function_name]
    mccs_list = mccs_dict[acquisition_function_name]

    color = colors[acquisition_function_name]
    linestyle = linestyles[acquisition_function_name]

    first_iter = True

    for problem_iter in problem_iters:

        plt.plot(total_samples_list[problem_iter-1], 
                 mccs_list[problem_iter-1], 
                 color=color, linestyle=linestyle,
                 linewidth=1, alpha=0.5, 
                 label=f"{acquisition_function_name}" if first_iter else None
        )
        first_iter = False

plt.xlabel("Total samples")
plt.ylabel("MCC")
plt.legend()

plt.savefig("_vs_".join(acquisition_function_names)+"_all_runs_comparison.png", dpi=300)
# plt.show()
plt.close()




############################################################################
""" Individual runs and the average """



def get_between_values(desired_idx, idx_vals, y_vals):
    """
    returns desired values at desired_idxs
    if the information doesn't exist in idx_vals, returns the y_vals value for the next smallest idx
    """
    y_desired = np.zeros(desired_idx.shape)
    for ii, d_idx in enumerate(desired_idx):
        for jj, idx in enumerate(idx_vals):
            if idx == d_idx:
                val = y_vals[jj]
                break
            elif idx>d_idx:
                break
            else: 
                val = y_vals[jj]
        y_desired[ii] = val
    return(y_desired)



plt.figure()

# Plot individual values
for acquisition_function_name in acquisition_function_names:

    iters_list = iters_dict[acquisition_function_name]
    total_samples_list = total_samples_dict[acquisition_function_name]
    mccs_list = mccs_dict[acquisition_function_name]

    color = colors[acquisition_function_name]
    linestyle = linestyles[acquisition_function_name]

    first_iter = True

    for problem_iter in problem_iters:

        plt.plot(total_samples_list[problem_iter-1], 
                 mccs_list[problem_iter-1], 
                 color=color, linestyle=linestyle, 
                 linewidth=1, alpha=0.5, 
                 #label=f"{acquisition_function_name}" if first_iter else None
        )
        first_iter = False


# Plot mean value
for acquisition_function_name in acquisition_function_names:

    iters_list = iters_dict[acquisition_function_name]
    total_samples_list = total_samples_dict[acquisition_function_name]
    mccs_list = mccs_dict[acquisition_function_name]

    color = colors[acquisition_function_name]
    linestyle = linestyles[acquisition_function_name]

    min_samp = np.max([np.min(total_samples) for total_samples in total_samples_list])
    max_samp = np.max([np.max(total_samples) for total_samples in total_samples_list])

    total_samples_plot = np.arange(min_samp, max_samp+1)
    mccs_plot = [get_between_values(total_samples_plot, total_samples, mccs) for total_samples, mccs in zip(total_samples_list, mccs_list)]
    print(f"{mccs_plot=}")
    mccs_plot = np.mean(np.vstack(mccs_plot), axis=0)
    print(f"{mccs_plot=}")
    print(f"{total_samples_plot=}")
    plt.plot(total_samples_plot, mccs_plot, 
             color=color, marker="o", linestyle=linestyle, 
             linewidth=3, label=f"{acquisition_function_name}")


plt.xlabel("Total samples")
plt.ylabel("MCC")
plt.legend()

plt.savefig("_vs_".join(acquisition_function_names)+"_all_runs_and_average_comparison.png", dpi=300)
# plt.show()
plt.close()





############################################################################
""" Average only """



plt.figure()


# Plot mean value
for acquisition_function_name in acquisition_function_names:

    iters_list = iters_dict[acquisition_function_name]
    total_samples_list = total_samples_dict[acquisition_function_name]
    mccs_list = mccs_dict[acquisition_function_name]

    color = colors[acquisition_function_name]
    linestyle = linestyles[acquisition_function_name]

    min_samp = np.max([np.min(total_samples) for total_samples in total_samples_list])
    max_samp = np.max([np.max(total_samples) for total_samples in total_samples_list])

    total_samples_plot = np.arange(min_samp, max_samp+1)
    mccs_plot = [get_between_values(total_samples_plot, total_samples, mccs) for total_samples, mccs in zip(total_samples_list, mccs_list)]
    print(f"{mccs_plot=}")
    mccs_plot = np.mean(np.vstack(mccs_plot), axis=0)
    print(f"{mccs_plot=}")
    print(f"{total_samples_plot=}")
    plt.plot(total_samples_plot, mccs_plot, 
             color=color, marker="o", linestyle=linestyle, 
             linewidth=3, label=f"{acquisition_function_name}")


plt.xlabel("Total samples")
plt.ylabel("MCC")
plt.legend()

plt.savefig("_vs_".join(acquisition_function_names)+"_average_comparison.png", dpi=300)
# plt.show()
plt.close()











#     egras = [get_between_values(samps_egra, idx_vals, y_vals) for (idx_vals, y_vals) in zip(SAMPS_E, EGRAS)]
#     emis = [get_between_values(samps_emi, idx_vals, y_vals) for (idx_vals, y_vals) in zip(SAMPS_A, emiS)]
# 
# egra_means = np.mean(np.vstack(egras), axis=0)
# 
# 
#     samps_mean = np.arange(min_samp, max_samp+1)
# 
#     for problem_iter in problem_iters:
# 
#         total_samples_list[problem_iter]
#         mccs_list[problem_iter]
# 
# 
# EGRAS = [egra1, egra2, egra3, egra4, egra5, egra6, egra7, egra8, egra9, egra10]
# emiS = [emi1, emi2, emi3, emi4, emi5, emi6, emi7, emi8, emi9, emi10]
# 
# SAMPS_E = [samp_e1, samp_e2, samp_e3, samp_e4, samp_e5, samp_e6, samp_e7, samp_e8, samp_e9, samp_e10]
# SAMPS_A = [samp_a1, samp_a2, samp_a3, samp_a4, samp_a5, samp_a6, samp_a7, samp_a8, samp_a9, samp_a10]
# 
# min_samp_egra = np.max([np.min(samp_e) for samp_e in SAMPS_E])
# min_samp_emi = np.max([np.min(samp_a) for samp_a in SAMPS_A])
# 
# max_samp_egra = np.min([np.max(samp_e) for samp_e in SAMPS_E])
# max_samp_emi = np.min([np.max(samp_a) for samp_a in SAMPS_A])
# 
# samps_egra = np.arange(min_samp_egra, max_samp_egra+1)
# samps_emi = np.arange(min_samp_emi, max_samp_emi+1)
# 
# 
# 
# egras = [get_between_values(samps_egra, idx_vals, y_vals) for (idx_vals, y_vals) in zip(SAMPS_E, EGRAS)]
# emis = [get_between_values(samps_emi, idx_vals, y_vals) for (idx_vals, y_vals) in zip(SAMPS_A, emiS)]
# 
# egra_means = np.mean(np.vstack(egras), axis=0)
# emi_means = np.mean(np.vstack(emis), axis=0)
# 
# print("samps_egra: \n", samps_egra)
# print("egras[1]: \n", egras[1])

# num_egra = min(egra1.size, egra2.size, egra3.size, egra4.size, egra5.size)
# num_emi = min(emi1.size, emi2.size, emi3.size, emi4.size, emi5.size)
# 
# 
# data_a = [np.array(a) for a in zip(egra1, egra2, egra3, egra4, egra5)]
# data_b = [np.array(b) for b in zip(emi1, emi2, emi3, emi4, emi5)]
# 
# 
# egra_means = np.array([np.mean(a) for a in data_a])
# emi_means = np.array([np.mean(b) for b in data_b])


# num_egra = min(egra1.size, egra2.size, egra3.size, egra4.size, egra5.size)
# num_emi = min(emi1.size, emi2.size, emi3.size, emi4.size, emi5.size)

# egras = [egra1, egra2, egra3, egra4, egra5]
# emis = [emi1, emi2, emi3, emi4, emi5]
# for ii, egrai in enumerate(egras):
#     egras[ii] = egrai[:num_egra]
# for ii, emii in enumerate(emis):
#     emis[ii] = emii[:num_emi]

# plt.figure()
# 
# for egra in egras:
#     plt.plot(samps_egra, egra, 'r-', linewidth=1, alpha=0.5)
# for emi in emis:
#     plt.plot(samps_emi, emi, 'b-', linewidth=1, alpha=0.5)
# 
# plt.plot(samps_egra, egra_means, 'ro-', linewidth=3, label="EGRA")
# plt.plot(samps_emi, emi_means, 'bo-', linewidth=3, label="EMI")
# plt.xlabel("Acquisition step")
# plt.ylabel("F1 score")
# plt.legend()
# plt.savefig('EGRA_vs_EMI_mean_points_comparison.png', dpi=300)
# # plt.show()
# plt.close()


