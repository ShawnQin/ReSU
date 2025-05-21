import numpy as np
import seaborn as sns
from typing import List, Dict
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


### This part is written by Gizem
""" Module for plotting the results. """

def plot_svd_comparison(*args, **kwargs):
    """
    Makes a 3 by 3 subplot, each row corresponds to a different method,
    first column: the singular values
    second column: the first left singular vector
    third column: the first right singular vector

    If args are not empty, then assign U, S, V to the first set of args
    """
    export_path = kwargs.pop("export_path", None)
    matrix_names = ("S1", "S2", "S3", "Vh1", "Vh2", "Vh3", "U1", "U2", "U3")

    if len(args) == 9:
        sub_dic = dict(zip(matrix_names, args[:]))
    else:
        assert isinstance(args[0], dict), "The input should be a dictionary"
        svd_result = args[0]
        sub_dic = {k: svd_result[k] for k in matrix_names}

    figure_titles = [
        "Method 1: PCA (H)",
        "Method 2: Min proj. ($HR^{-1}H^{T}$)",
        "Method 3: CCA ($R^{-1/2}HR^{-1/2T}$)",
    ]
    fig, ax = plt.subplots(3, 3, figsize=(9, 6), sharex=True)

    # plot the singular values, subset the contain S1, S2, S3
    for ix, (_, value) in enumerate(sub_dic.items()):
        if ix < 3:
            ax[ix, 0].plot(value, color="black")
            ax[ix, 0].set_yscale("log")
            # first 3 singular values
            first_three = np.sum(value[0:3]) / np.sum(value)
            ax[ix, 0].set_title(f"First 3 S.V: {first_three*100:.1f} %")
            # ax[ix,0].set_title(key)
        elif ix < 6:
            ax[ix - 3, 1].plot(value[0:3, :].T)
            ax[ix - 3, 1].axhline(0, color="black", linestyle="--", linewidth=0.5)
        #     ax[ix-3,1].set_title(key)
        else:
            ax[ix - 6, 2].plot(value[:, 0], label="1st")
            ax[ix - 6, 2].plot(value[:, 1], label="2nd")
            ax[ix - 6, 2].plot(value[:, 2], label="3rd")
            ax[ix - 6, 2].axhline(0, color="black", linestyle="--", linewidth=0.5)
            # ax[ix-6,2].set_title(key)
            ax[ix - 6, 2].legend(ncols=2)

    xlabels = ["Singular value index", "Lag", "Lag"]
    ylabels = [
        "Singular value",
        "Amplitude (L. sing. vec.)",
        "Amplitude (R. sing. vec.)",
    ]

    for i in range(3):
        ax[-1, i].set_xlabel(xlabels[i])
        ax[0, i].set_ylabel(ylabels[i])
        ax[1, i].set_ylabel(ylabels[i])
        ax[2, i].set_ylabel(ylabels[i])
    # set the titles
    for i in range(3):
        ax[i, 1].set_title(figure_titles[i])

    plt.tight_layout()

    # Commenting this because the function is not clear
    # for key, value in kwargs.items():
    # print(f'{key}: {value}')
    # save the figure as pdf
    if export_path is not None:
        plt.savefig(export_path, bbox_inches="tight")
        print(f"Plot saved in {export_path}")

    plt.show()
    return fig, ax


def plot_filters(*args, **kwargs):
    """
    If the input is a tuple of arrays, it should be
    in the order of filter 1, 2, 3.
    If the input is a dictionary, then it should have the
    following keys: filter1, fitler2, filter3
    """
    export_path = kwargs.pop("export_path", None)
    normalize_filters = kwargs.pop("normalize_filters", False)
    matrix_names = ("filter1", "filter2", "filter3")
    # examine if the input is dictionary
    if len(args) == 3:
        sub_dic = dict(zip(matrix_names, args[:]))
    else:
        assert isinstance(args[0], dict), "The input should be a dictionary"
        svd_result = args[0]
        sub_dic = {k: deepcopy(svd_result[k]) for k in matrix_names}

    figure_titles = [
        "$V_{1}$ (PCA)",
        "$U_{3}HR^{-1}$ (Min. proj.)",
        "$V_{2}^{T} R^{-1/2}$ (CCA)",
    ]
    legends = ["1st", "2nd"]
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    for i, (_, value) in enumerate(sub_dic.items()):
        if normalize_filters:
            for row in range(value.shape[0]):
                value[row, :] = value[row, :] / np.max(np.abs(value[row, :]))

        ax[i].plot(value[:2, :].T, label=legends)
        ax[i].legend(frameon=False)
        ax[i].axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax[i].set_title(figure_titles[i])
        ax[i].set_xlabel("Lag")

    ax[0].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.suptitle("Filters for three methods", y=1.02)
    # for key, value in kwargs.items():
    #     print(f'{key}: {value}')
    #     # save the figure as pdf
    #     plt.savefig(f'./figures/{value}.pdf', bbox_inches='tight')
    if export_path is not None:
        plt.savefig(export_path, bbox_inches="tight")

    plt.show()
    return fig, ax


def plot_covariance(*args, **kwargs):
    """
    Plot the covariance matrix and compare the three different methods
    Either the input is a dictionary or a list of matrices
    """
    export_path = kwargs.pop("export_path", None)
    matrix_names = ("Rp", "Rp_half", "H", "W2", "W3")

    if len(args) == 5:
        sub_dic = dict(zip(matrix_names, args[:]))
    else:
        assert isinstance(args[0], dict), "The input should be a dictionary"
        svd_result = args[0]
        sub_dic = {k: svd_result[k] for k in matrix_names}

    title = ["$R$", "$R^{1/2}$", "$H=FP^{T}$", "$HR^{-1}H^{T}$", "$R^{-1/2}HR^{-1/2T}$"]

    # visualize all the matrices
    fig, ax = plt.subplots(2, 3, figsize=(9, 5))
    for ix, (_, value) in enumerate(sub_dic.items()):
        plt.subplot(2, 3, ix + 1)
        plt.imshow(value, cmap="viridis")
        plt.colorbar(shrink=0.7)
        plt.title(title[ix])
        # Add x and y labels
        plt.xlabel("Lag")
        plt.ylabel("Lag")

    plt.subplot(2, 3, 6)
    # resize the axis
    plt.plot(sub_dic["Rp_half"][:, 0], color="black")
    plt.title("$R_{p}^{1/2}$")
    plt.title("first column of $Rp^{1/2}$")
    plt.xlabel("Lag")
    plt.ylabel("Amplitude")

    plt.tight_layout(pad=-0.1)
    # for key, value in kwargs.items():
    #     plt.savefig(f'./figures/{value}.pdf')
    if export_path is not None:
        plt.savefig(export_path, bbox_inches="tight")

    plt.show()
    return fig, ax


def plot_projection(*args, **kwargs):
    """
    Rows of filters correspond to the filters
    P is the matrix of past lag vectors
    """
    export_path = kwargs.pop("export_path", None)
    time_window = kwargs.pop("time_window", 500)
    normalize = kwargs.pop("normalize", False)

    matrix_names = ("filter1", "filter2", "filter3", "P")
    if len(args) == 4:
        sub_dic = dict(zip(matrix_names, args[:]))
    else:
        assert isinstance(args[0], dict), "The input should be a dict"
        svd_result = args[0]
        sub_dic = {k: svd_result[k] for k in matrix_names}

    # The corresponding input data
    lag = sub_dic["P"].shape[0]
    lumi_input = sub_dic["P"][-1, -time_window:]
    times = np.arange(-lag, time_window - lag)
    projections = []

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    figure_titles = ["$H$", "$HR^{-1}H^T$", "$R^{1/2}HR^{-1/2T}$"]
    for ix, (_, value) in enumerate(sub_dic.items()):
        if ix < 3:
            plt.subplot(1, 3, ix + 1)

            proj_resp = value[:2, :] @ sub_dic["P"][:, -time_window:]

            if normalize:
                # min max normalize everything
                lumi_input = (lumi_input - np.min(lumi_input)) / (
                    np.max(lumi_input) - np.min(lumi_input)
                )
                for row in range(proj_resp.shape[0]):
                    proj_resp[row, :] = (
                        proj_resp[row, :] - np.min(proj_resp[row, :])
                    ) / (np.max(proj_resp[row, :]) - np.min(proj_resp[row, :]))

            projections.append(proj_resp)
            # Add transparence to the line
            ax[ix].plot(
                times,
                lumi_input.T,
                color="black",
                linestyle="--",
                linewidth=1,
                label="input",
            )
            ax[ix].plot(proj_resp[0:, :].T, alpha=0.8, label=["1st proj.", "2nd proj."])
            ax[ix].set_title(figure_titles[ix])
            # Add labels
            ax[ix].set_xlabel("Time steps")
        ax[0].set_ylabel("Amplitude")
        # Add legend
        ax[-1].legend()

    plt.tight_layout()
    # for key, value in kwargs.items():
    #     plt.savefig(f'./figures/{value}.pdf', bbox_inches='tight')
    if export_path is not None:
        plt.savefig(export_path, bbox_inches="tight")
    plt.show()

    return fig, ax, lumi_input, projections


def plot_filter_obj(
    obj_scalars: Dict[str, float],
    obj_random: List,
    export_path: str = None,
    ax: Axes = None,
    title: str = '',
    legend: bool = True,
    clip=None,
    color_dist='steelblue'
) -> Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5, 1.2))
    sns.kdeplot(
        obj_random, ax=ax,
        color=color_dist,
        alpha=0.2,
        fill=True,
        clip=clip
    )
    linestyles = ['--', '-.', ':']

    for i, (key, value) in enumerate(obj_scalars.items()):
        ax.axvline(value, lw=1, ls=linestyles[i], label=key, color='black')

    ax.set_xlabel('Objective func. value')
    ax.set_ylabel('Density')
    if legend:
        ax.legend()
    ax.set_title(title)

    if export_path is not None:
        plt.savefig(export_path, dpi=300)
    return ax



### The following is written by Shanshan
def plot_svd_comparison(*args, **kwargs):
    # make a 3 by 3 subplot, each row corresponds to a different method,
    #first column is the singular values, second column is the first left singular vector,
    # if args are not empty, then assign U, S, V to the first set of args
    if len(args) == 9:
        U1, S1, V1, U2, S2, V2, U3, S3, V3 = args[:]
    else:
        assert isinstance(args[0], dict), 'The input should be a dictionary'
        svd_result = args[0]
        sub_dic = {k: svd_result[k] for k in ('S1','S2','S3', 'Vh1','Vh2','Vh3','U1','U2','U3')}

    figure_titles = ['H', '$HR^{-1}H^T$', '$R^{1/2}HR^{-1/2T}$']
    fig, ax = plt.subplots(3, 3, figsize=(11, 8))
    # plot the singular values, subset the contain S1, S2, S3
    for ix, (key, value) in enumerate(sub_dic.items()):
        if ix < 3:
            ax[ix,0].plot(np.log(value))
            # ax[ix,0].set_title(key)
        elif ix < 6:
            ax[ix-3,1].plot(value[0:2,:].T)
            ax[ix-3,1].axhline(0, color='black', linestyle='--', linewidth=0.5)
        #     ax[ix-3,1].set_title(key)
        else:
            ax[ix-6,2].plot(value[:,0:2])
            ax[ix-6,2].axhline(0, color='black', linestyle='--', linewidth=0.5)
            # ax[ix-6,2].set_title(key)

    xlabels = ['Singular value index', 'Lag', 'Lag']
    ylabels = ['Singular value', 'Amplitude', 'Amplitude']
    for i in range(3):
        ax[2, i].set_xlabel(xlabels[i])
        ax[i, 0].set_ylabel(ylabels[i])
    # set the titles
    for i in range(3):
        ax[0, i].set_title(figure_titles[i])

    for key, value in kwargs.items():
        print(f"{key}: {value}")
        # save the figure as pdf
        plt.savefig(f'./figures/{value}.pdf', bbox_inches='tight')
    # plt.savefig('./figures/contrast_trace.pdf', bbox_inches='tight')
    plt.show()
    return fig, ax


def plot_filters(*args, **kwargs):
    '''
    If the input is a dictionary, then keys are the names of the filter1
    '''
    # examine if the input is dictionary
    if len(args) == 3:
        filter1, filter2, filter3 = args[:]
    else:
        assert isinstance(args[0], dict), 'The input should be a dictionary'
        svd_result = args[0]
        keys = ['filter1', 'filter2', 'filter3']
        filters = [svd_result[key] for key in keys]

    figure_titles = ['H', '$HR^{-1}H^T$', '$R^{1/2}HR^{-1/2T}$']
    legends = ['1st', '2nd']
    fig, ax = plt.subplots(1,3, figsize=(12, 3))

    for i in range(3):
        ax[i].plot(filters[i][:2,:].T/np.max(np.abs(filters[i][:2,:].T),axis = 0), label=legends)
        ax[i].legend()
        ax[i].axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax[i].set_title(figure_titles[i])
        ax[i].set_xlabel('Lag')
    ax[0].set_ylabel('Amplitude')

    for key, value in kwargs.items():
        print(f"{key}: {value}")
        # save the figure as pdf
        plt.savefig(f'./figures/{value}.pdf', bbox_inches='tight')

    return fig, ax


def plot_covariance(*args,**kwargs):
    '''
    Plot the covariance matrix and compare the three different methods
    Either the input is a dictionary or a list of matrices
    '''
    if len(args) == 5:
        Rp, Rp_half, Q, W2, W3 = args[:]
    else:
        assert isinstance(args[0], dict), 'The input should be a dictionary'
        svd_result = args[0]
        sub_dic = {k: svd_result[k] for k in ('Rp', 'Rp_half', 'Q', 'W2', 'W3')}
        title = ['R','$R^{1/2}$','H=FP^T','$HR^{-1}H^T$', '$R^{1/2}HR^{-1/2T}$']
        # Rp, Rp_half, Q, W2, W3 = svd_result['Rp'], svd_result['Rp_half'], svd_result['Q'], \
        #     svd_result['W2'], svd_result['W3']
    # visualize all the matrices
    fig, ax = plt.subplots(2, 3, figsize=(10, 7))
    for ix, (key, value) in enumerate(sub_dic.items()):
        plt.subplot(2, 3, ix+1)
        plt.imshow(value, cmap='viridis')
        plt.colorbar(shrink = 0.7)
        plt.title(title[ix])
        # add x and y labels
        plt.xlabel('Lag')
        plt.ylabel('Lag')

    plt.subplot(2, 3, 6)
    # resize the axis
    plt.plot(sub_dic['Rp'][:,0])
    plt.title('$Rp^{1/2}$')
    plt.title('first column of $Rp$')
    plt.xlabel('Lag')
    plt.ylabel('Amplitude')


    for key, value in kwargs.items():
        plt.savefig(f'./figures/{value}.pdf')

    return fig, ax


def plot_projection(*args, **kwargs):
    '''
    Rowd of filters correspond to the filters
    P    matrix of past lag vectors
    '''
    if len(args) == 4:
        filter1, filter2, filter3, P = args[:]
    else:
        assert isinstance(args[0], dict), 'The input should be a dictionary'
        svd_result = args[0]
        sub_dic = {k: svd_result[k] for k in ('filter1', 'filter2', 'filter3', 'P')}

    # the correspoinding input data
    plot_length = 500
    lag = sub_dic['P'].shape[0]
    lumi_input = sub_dic['P'][-1,-plot_length:]
    times = np.arange(-lag,plot_length-lag)

    fig, ax = plt.subplots(1,3, figsize=(12, 3))
    figure_titles = ['H', '$HR^{-1}H^T$', '$R^{1/2}HR^{-1/2T}$']
    for ix, (key, value) in enumerate(sub_dic.items()):
        if ix < 3:
            plt.subplot(1,3, ix+1)
            resp = value[:2,:]@sub_dic['P'][:,-500:]
            # add transparence to the line
            ax[ix].plot(times,lumi_input.T, color='black', linestyle='--', linewidth=1,label='input')
            ax[ix].plot(resp[0:,:].T, alpha=0.8, label=['1st projection', '2nd prjection'])
            ax[ix].set_title(figure_titles[ix])
            # add x labels
            ax[ix].set_xlabel('Time')
            ax[ix].set_ylabel('Amplitude')
            # add lengend   
            ax[ix].legend()
    for key, value in kwargs.items():
        plt.savefig(f'./figures/{value}.pdf', bbox_inches='tight')

    return fig, ax


def plot_project_resp(svd_result,P, fig_show=False, **kwargs):
    '''
    Rowd of filters correspond to the filters
    P    matrix of past lag vectors
    '''

    assert isinstance(svd_result, dict), 'The input should be a dictionary'
    # svd_result = args[0]
    sub_dic = {k: svd_result[k] for k in ('filter1', 'filter2', 'filter3')}
    # P = args[1]  # Past henkel matrix

    # the correspoinding input data
    # plot_length = 500
    lag = sub_dic['filter1'].shape[1]
    lumi_input = P[-1,:]
    times = np.arange(-lag,P.shape[1]-lag)

    fig, ax = plt.subplots(1,3, figsize=(12, 3))
    figure_titles = ['H', '$HR^{-1}H^T$', '$R^{1/2}HR^{-1/2T}$']
    resps = []
    for ix, (key, value) in enumerate(sub_dic.items()):
        if ix < 3:
            plt.subplot(1,3, ix+1)
            resp = value[:2,:]@P
            resps.append(resp)
            # add transparence to the line
            # ax[ix].plot(times,lumi_input.T, color='black', linestyle='--', linewidth=1,label='input')
            ax[ix].plot(resp[:2,:].T, alpha=0.8, label=['1st projection', '2nd prjection'])
            ax[ix].set_title(figure_titles[ix])
            # add x labels
            ax[ix].set_xlabel('Lag time')
            ax[ix].set_ylabel('Amplitude')
            # add lengend   
            ax[ix].legend()
    for key, value in kwargs.items():
        plt.savefig(f'./figures/{value}.pdf', bbox_inches='tight')

    return resps

def plot_compare_noise_dependent(svd_results,noise_std,filename=None):
    '''
    Compare the filter shape when the noise level is different
    '''
        # make plot of the first filter
    line_styles = ['-', '--', '-.', ':']
    colors = sns.color_palette()
    legends = [r"$\sigma$"+' = ' + str(noise) for noise in noise_std]

    fig, ax = plt.subplots(2, 3, figsize=(12, 5))
    filters = ['filter1', 'filter2', 'filter3']
    for i in range(len(noise_std)):
        for j0,flt in enumerate(filters):
            ax[0,j0].plot(svd_results[i][flt][0,:],label=legends[i], linestyle=line_styles[i],color=colors[i])
            ax[1,j0].plot(svd_results[i][flt][1,:],label=legends[i],linestyle=line_styles[i],color=colors[i])

    fig_titles = ['H', r'$HR^{-1}H^T$', r'$R^{1/2}HR^{-1/2T}$']
    ylabels = ['1st filter', '2nd filter']
    for i in range(3):
        ax[0,i].set_title(fig_titles[i])
        # set x labels
        ax[1,i].set_xlabel('Lag time')
    # show the legend
    for i in range(2):
        ax[i,0].set_ylabel(ylabels[i])
        # ax[i,0].legend()
    ax[0,0].legend()

    # save as a pdf 
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight',transparent=True)
    # fig_name = './figures/saccade_scale1_filter_noise_dependent_lag' + str(lag) +  '.pdf'
    # plt.savefig(fig_name, bbox_inches='tight')

    plt.show()
    return fig, ax


def plot_normalized_filter_noise(svd_results, noise_std, which_filter = 3, filename=None):
    '''
    Compare the filter shape when the noise level is different
    '''
    colors = ['#a50f15','#fb6a4a','#fcae91']
    legends = [r'$\sigma$'+' = ' + str(noise) for noise in noise_std]
    # select the filter to plot
    filter_name = 'filter' + str(which_filter)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for i in range(len(noise_std)):
        orignal_flt = svd_results[i][filter_name][0,:]
        ax[0].plot(orignal_flt/np.max(abs(orignal_flt)),label=legends[i], color=colors[i],linewidth=2)
        orignal_flt = svd_results[i][filter_name][1,:]
        ax[1].plot(orignal_flt/np.max(abs(orignal_flt)),label=legends[i], color=colors[i],linewidth=2)
    # add y = 0 line
    ax[0].axhline(y=0, color='0.8', linestyle='--',linewidth=0.5)
    ax[1].axhline(y=0, color='0.8', linestyle='--',linewidth=0.5)
    ax[0].set_title('1st filter')
    ax[1].set_title('2nd filter')
    ax[0].set_xlabel('Lag time')
    ax[1].set_xlabel('Lag time')
    ax[0].set_ylabel('Normalized value')
    ax[1].set_ylabel('Normalized value')
    ax[1].legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight',transparent=True)
    plt.show()
    return fig, ax


# def plot_prediction_sp_tp_filter(test_movie,filter):
#     '''
#     Make a prediction of the upcoming stimulus at the center of the RF
#     '''