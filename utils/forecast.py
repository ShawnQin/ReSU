# pylint: disable=W0612
""" Module containing a class for forecasting methods. """

from copy import deepcopy
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.linalg import sqrtm

from utils import analysis


class PredictFuture:
    """Class to predict the future of a trajectory.
    Specifically, this class performs the following steps:
    1. Perform SVD analysis on the trajectory
    2. Predict the future using PCA
    3. Quantify the MSE of the prediction

    Parameters
    ----------
    trajectory : np.ndarray
        Input trajectory
    memory : int
        Previous time steps to store
    horizon : int
        Future time steps to predict
    rank : int, optional
        Rank of the prediction, by default 2
    method : str, optional
        Method, could be pca, cca, min_proj or regression,
        by default 'pca'


    Example usage:
    --------------
    memory = 50  # past 10 steps
    horizon = 15  # predict 1 step ahead
    trajectory = <some trajectory>

    predict_future = PredictFuture(
            trajectory,
            memory,
            horizon,
            rank=-1,
            method='pca',
    )

    (
        future_prediction,
        x_partial,
        future_filter,
        past_filter
    ) = predict_future.predict_future(method='pca')
    mse = predict_future.quantify_mse()
    predict_future.plot_prediction(
        export_path="./result.png",
        title="PCA future prediction"
    )
    """

    def __init__(
        self,
        trajectory: np.ndarray,
        memory: int,
        horizon: int,
        rank: int = -1,
    ):
        self.trajectory = trajectory
        self.memory = memory
        self.horizon = horizon
        self.n_samples = len(trajectory) - memory - horizon
        self.rank = rank
        self.ridge_regression = None
        self.future_prediction = None
        self.future_matrix = None
        self.past_matrix = None
        self.future_filter = None
        self.past_filter = None
        self.mse = None
        self.corr_coef = None
        self.matrices = {}

    @staticmethod
    def min_max_normalization(data):
        """Min max normalization."""
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

    @staticmethod
    def standard_normalization(data):
        """Standard normalization."""
        data = (data - np.mean(data)) / (np.std(data))
        return data

    def create_past_future_matrices(self):
        """
        Create past and future matrices.
        Past trajectory is the trajectory from 0 to memory.
        Future trajectory is the trajectory
        from memory + 1 to memory + 1 + horizon.

        This method builds block matrices of past and future trajectories
        with <n_samples> many samples.
        """
        future_matrix = np.zeros((self.n_samples, self.horizon))
        past_matrix = np.zeros((self.n_samples, self.memory))
        for i in range(0, self.n_samples):
            past_matrix[i, :] = self.trajectory[i : i + self.memory]
            future_matrix[i, :] = self.trajectory[
                i + self.memory + 1 : i + self.memory + 1 + self.horizon
            ]
        # Flip past matrix so that the last row is the oldest
        past_matrix = np.flip(past_matrix[:,:], axis=1)
        return past_matrix, future_matrix

    def predict_future(
        self, method: Literal["regression", "pca", "cca", "min_proj"], **kwargs
    ):
        # Create past and future matrices
        self.past_matrix, self.future_matrix = self.create_past_future_matrices()
        if method.lower() == "regression":
            self.future_prediction = self.predict_future_regression(**kwargs)
            self.future_filter = self.regression_filters
            return self.future_prediction, self.future_filter
        elif method.lower() == "pca":
            (future_prediction, x_partial, future_filter, past_filter,singular_values) = (
                self.predict_future_pca()
            )
        elif method.lower() == "cca":
            (future_prediction, x_partial, future_filter, past_filter, singular_values) = (
                self.predict_future_cca()
            )
        elif method.lower() == "min_proj":
            (future_prediction, x_partial, future_filter, past_filter, singular_values) = (
                self.predict_future_minproj()
            )
        else:
            raise ValueError(
                "Method not recognized, please select `regression`, `pca`, `cca` or `min_proj`."
            )

        self.future_prediction = future_prediction
        self.future_filter = future_filter
        self.past_filter = past_filter

        return future_prediction, x_partial, future_filter, past_filter, singular_values

    def predict_future_regression(self, **kwargs):
        """Use regression to predict the future of the trajectory."""
        alpha = kwargs.get("alpha", 0.1)

        clf = Ridge(alpha=alpha)
        # Fit a regressor between the past and future matrices
        self.ridge_regression = clf.fit(self.past_matrix, self.future_matrix)
        past_trajectory = self.past_matrix[0, :]

        return self.ridge_regression.predict(self.past_matrix).T

    @property
    def regression_filters(self):
        if self.ridge_regression is None:
            print("No regression model has been trained.")
            return None
        return self.ridge_regression.coef_[:, :]

    @property
    def hankel_matrix(self):
        """returns a matrix of horizon x memory"""
        # hankel matrix = FP^T
        return self.future_matrix.T @ self.past_matrix / self.n_samples

    @property
    def toeplitz_past(self):
        """returns the past covariance matrix"""
        return self.past_matrix.T @ self.past_matrix / self.n_samples

    @property
    def toeplitz_future(self):
        """returns the future covariance matrix"""
        return self.future_matrix.T @ self.future_matrix / self.n_samples

    def predict_future_pca(self):
        """Given past and future number of time steps,
        this functions performs a PCA
        to predict the future <horizon>.
        """
        past_matrix = deepcopy(self.past_matrix).T  # memory x delay
        hankel_matrix = deepcopy(self.hankel_matrix)
        # print(hankel_matrix.shape, past_matrix.shape)

        U_pca, S_pca, V_pca, filter_pca = analysis.pca_method(hankel_matrix)
        singular_matrix = np.diag(S_pca)
        # print(U_pca.shape, S_pca.shape, V_pca.shape, filter_pca.shape)
        future_filter = U_pca[:, :self.rank] if self.rank is not None else U_pca[:,:]
        singular_matrix = singular_matrix[:self.rank, :self.rank] if self.rank is not None else singular_matrix[:,:]
        filter_pca = filter_pca[:self.rank, :] if self.rank is not None else filter_pca[:,:]

        x_partial = filter_pca @ past_matrix
        future_prediction = future_filter @ x_partial
        # print(future_prediction.shape)
        self.matrices = {
            **self.matrices,
            "U_pca": U_pca,
            "S_pca": S_pca,
            "V_pca": V_pca,
            "filter_pca": filter_pca,
        }

        return future_prediction, x_partial, future_filter, filter_pca, S_pca 

    def predict_future_cca(self):
        """Given past and future number of time steps,
        this functions performs a CCA
        to predict the future <horizon>.
        """
        past_matrix = deepcopy(self.past_matrix).T  # memory x delay
        future_matrix = deepcopy(self.future_matrix).T  # horizon x delay
        hankel_matrix = deepcopy(self.hankel_matrix)

        # Toeplitz matrix = PP^T
        toeplitz_past = deepcopy(self.toeplitz_past)
        toeplitz_future = deepcopy(self.toeplitz_future)

        # try:
        #     Rp_half = np.linalg.cholesky(toeplitz_past)
        # except:
        #     Rp_half = np.linalg.cholesky(toeplitz_past + 1e-10 * np.eye(toeplitz_past.shape[0]))

        # try:
        #     Rf_half = np.linalg.cholesky(toeplitz_future)
        # except:
        #     Rf_half = np.linalg.cholesky(toeplitz_future + 1e-10 * np.eye(toeplitz_future.shape[0]))

        # using sqrtm method
        try:
            Rp_half = sqrtm(toeplitz_past)
        except:
            Rp_half = sqrtm(toeplitz_past + 1e-10 * np.eye(toeplitz_past.shape[0]))
        try:
            Rf_half = sqrtm(toeplitz_future)
        except:
            Rf_half = sqrtm(toeplitz_future + 1e-10 * np.eye(toeplitz_future.shape[0]))

        cca_matrix, U, S, Vh, filter_cca= analysis.cca_method(
            hankel_matrix, Rf_half=Rf_half, Rp_half=Rp_half
        )

        future_filter = np.linalg.inv(Rf_half) @ U
        x_partial = filter_cca[:, :] @ past_matrix
        future_prediction = future_filter[:, :] @ x_partial

        self.matrices = {
            **self.matrices,
            "toeplitz_matrix_past": toeplitz_past,
            "toeplitz_matrix_future": toeplitz_future,
            "Rp_half": Rp_half,
            "Rf_half": Rf_half,
            "cca_matrix": cca_matrix,
            "U_cca": U,
            "S_cca": S,
            "Vh_cca": Vh,
            "filter_cca": filter_cca,
        }

        return future_prediction, x_partial, future_filter, filter_cca, S

    def predict_future_minproj(self):
        """Given past and future number of time steps,
        this functions performs a minimum projection
        to predict the future <horizon>.
        """
        past_matrix = deepcopy(self.past_matrix).T  # memory x delay
        future_matrix = deepcopy(self.future_matrix).T  # horizon x delay
        nb_of_samples = self.hankel_matrix.shape[1]
        hankel_matrix = deepcopy(self.hankel_matrix)

        # Toeplitz matrix = PP^T
        toeplitz_past = deepcopy(self.toeplitz_past)
        toeplitz_future = deepcopy(self.toeplitz_future)

        minproj_matrix, U_minproj, V_minproj, Vh_minproj, filter_minproj = (
            analysis.min_proj_method(
                hankel_matrix=hankel_matrix, Rp=toeplitz_past
            )
        )

        filter_minproj = filter_minproj[:self.rank, :] if self.rank is not None else filter_minproj[:,:]
        x_partial = filter_minproj @ past_matrix

        h_psi = hankel_matrix @ filter_minproj.T
        psi_r_psi = filter_minproj @ toeplitz_past @ filter_minproj.T

        future_filter = h_psi @ np.linalg.inv(psi_r_psi)
        future_filter = future_filter[:, :self.rank] if self.rank is not None else future_filter[:,:]
        # print(future_filter.shape, x_partial.shape, filter_minproj.shape)
        future_prediction = future_filter @ x_partial

        self.matrices = {
            **self.matrices,
            "toeplitz_matrix_past": toeplitz_past,
            "toeplitz_matrix_future": toeplitz_future,
            "minproj_matrix": minproj_matrix,
            "U_minproj": U_minproj,
            "V_minproj": V_minproj,
            "Vh_minproj": Vh_minproj,
            "filter_minproj": filter_minproj,
        }

        return future_prediction, x_partial, future_filter, filter_minproj,V_minproj

    def quantify_mse(self):
        """Normalize the prediction and compute the MSE
        between the prediction and the ground truth.
        """
        # normalize the trajectory and the prediction
        traj_short = self.trajectory[self.memory : self.memory + self.horizon]
        pred_short =  self.future_prediction[:, 0]

        self.mse = np.mean((traj_short - pred_short) ** 2)

    def quantify_correlation(self):

        traj_short = self.trajectory[self.memory : self.memory + self.horizon]
        pred_short =  self.future_prediction[:, 0]

        self.corr_coef = np.corrcoef(traj_short, pred_short)[0,1]

    def normalize_pred(self, ground_truth, prediction):
        """Finds a multiplier coefficient to make first time step
        of the prediction equal to the ground truth's first.
        """
        # align prediction with ground truth by finding the ratio and offset
        gt_min, gt_max = np.min(ground_truth), np.max(ground_truth)
        # normalize pred
        prediction = (prediction - np.min(prediction)) / (
            np.max(prediction) - np.min(prediction)
        )
        prediction = prediction * (gt_max - gt_min) + ground_truth[0]

        return prediction[:]

    def plot_prediction(self, export_path=None, normalize=False, title="",figsize=(2,1)):
        """Plot the prediction and the ground truth."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        memory_time = np.arange(self.memory+1)
        # future_time = np.arange(self.memory, self.memory + self.horizon)
        pred_time = np.arange(self.memory, self.memory + self.horizon)
        ax.plot(
            memory_time,
            self.trajectory[: self.memory+1],
            label="past",
            color="dimgray",
        )
        ax.plot(
            pred_time,
            self.trajectory[self.memory: self.memory + self.horizon],
            label="observed",
            color="black",
        )
        
        if not normalize:
            ax.plot(
                pred_time,
                self.future_prediction[:, 0],
                label="prediction",
                color="firebrick",
            )
        else:
            pred_normalized = self.normalize_pred(
                self.trajectory[self.memory : self.memory + self.horizon],
                self.future_prediction[:, 0],
            )
            ax.plot(
                pred_time,
                pred_normalized,
                label="prediction",
                ls=":",
                color="firebrick",
            )
        ax.axvline(self.memory, color="grey", lw=0.5, alpha=0.5, ls="--")
        ax.legend()
        # set x ticks
        ax.set_xticks([0, self.memory, self.memory + self.horizon])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(" Voltage (mV)")
        ax.set_title(title)

        if export_path is not None:
            plt.savefig(export_path, dpi=300)
        plt.show()

    def plot_filters(self, export_path=None, title="", which_filter="past"):
        """Plots filters."""
        if which_filter == "past":
            filters = self.past_filter
        else:
            filters = self.future_filter

        colors = plt.cm.coolwarm(np.linspace(0, 1, filters.shape[0]))

        fig, ax = plt.subplots(figsize=(2, 1))

        for x in range(filters.shape[0]):
            # normalize by the max values of the filters
            plt.plot(filters[x, :] / filters[x, :].max(), color=colors[x])

        plt.xlabel("Time (memory)")
        plt.ylabel("Value")
        plt.title(title)

        if export_path is not None:
            plt.savefig(export_path, dpi=300)

        plt.show()
