from typing import Dict, Tuple
import numpy as np
import scipy as sp
from scipy.linalg import sqrtm
import sklearn.utils.extmath as extmath
from scipy.fft import fft, ifft
# import jax.numpy as jnp
#from jax import jit



def SVD_analysis(traj, m = 10, h = 10):
    '''
    Compute the SVD of three different balanced truncatio
    traj:     trajectory of the observed data
    lag:      the length of the past
    h:      the length of the future, an integer
    '''
    # crete the Hankel matrix from trajectory
    lag = m + h
    if len(traj) < 10000 or lag < 500:
        H = sp.linalg.hankel(traj[:lag], traj[lag-1:])
        num_sample = H.shape[1]
        P = H[:m,:]
        P = np.flipud(P)
        F = H[m:,:]
        Q = F@P.T/num_sample

        # Compute SVD of partially whittened covariance
        Rp = P@P.T/num_sample
        Rf = F@F.T/num_sample
        # Rp_half = np.linalg.cholesky(Rp)
        # Rf_half = np.linalg.cholesky(Rf)
        Rp_half = sqrtm(Rp)
        Rf_half = sqrtm(Rf)

    else:
        # divide the trajectory into segments
        seg_len = 2*lag
        num_seg = np.floor(len(traj)/seg_len).astype(int)
        Q, Rp, Rf = 0, 0, 0
        for i in range(num_seg):
            sub_traj = traj[i*seg_len:(i+1)*seg_len]
            H = sp.linalg.hankel(sub_traj[:lag], sub_traj[lag-1:])
            P = np.flipud(H[:m,:])
            F = H[m:,:]
            Q += F@P.T
            Rp += P@P.T
            Rf += F@F.T
        num_sample = num_seg*seg_len
        P = P/num_sample
        F = F/num_sample
        Q = Q/num_sample
        Rp = Rp/num_sample
        Rf = Rf/num_sample

        # Rp_half = np.linalg.cholesky(Rp)
        # Rf_half = np.linalg.cholesky(Rf)
        Rp_half = sqrtm(Rp_half)
        Rf_half = sqrtm(Rf_half)

    # SVD of the past future covariance
    U1, S1, Vh1 = sp.linalg.svd(Q,full_matrices=False)
    filter1 = Vh1

    # SVD of oneside whitened covariance
    W2 = Q@np.linalg.inv(Rp)@Q.T
    U2, S2, Vh2 = sp.linalg.svd(W2,full_matrices=False)
    filter2 = U2.T@Q@np.linalg.inv(Rp)
    # filter2 = filter2.T

    # CCA
    W3 = np.linalg.inv(Rf_half)@Q@np.linalg.inv(Rp_half).T
    W3 = np.real(W3)
    U3, S3, Vh3 = sp.linalg.svd(W3,full_matrices=False)
    filter3 = Vh3@np.linalg.inv(Rp_half)

    # store the results in dictionary
    results = {'U1':U1,'S1':S1,'Vh1':Vh1,'filter1':filter1,
               'U2':U2,'S2':S2,'Vh2':Vh2,'filter2':filter2,
               'U3':U3,'S3':S3,'Vh3':Vh3,'filter3':filter3,
               'Rp_half':Rp_half,'Rf_half':Rf_half,'Q':Q,
               'W2':W2, 'W3':W3,'P':P,'F':F,'Rp':Rp,'Rf':Rf}
    return results

def SVD_layer2(Fs, Ps, regularize = False):
    '''
    Compute the SVD of three different balanced truncatio

    Fs:        future data lag matrix, a list for three method
    Ps:        past data lag matrix, a list for three method
    '''
    # crete the Hankel matrix from trajectory
    for ix, (F, P) in enumerate(zip(Fs, Ps)):
        lag = P.shape[0]
        num_sample = P.shape[1]
        sep = np.round(lag/2).astype(int)
        Q = F@P.T/num_sample

        # Compute SVD of partially whittened covariance
        Rp = P@P.T/num_sample
        Rf = F@F.T/num_sample

        if regularize:
            Rp = Rp + 1e-10*np.eye(Rp.shape[0])
            Rf = Rf + 1e-10*np.eye(Rf.shape[0])
        # Rp_half = np.linalg.cholesky(Rp)
        # Rf_half = np.linalg.cholesky(Rf)
        Rp_half = sqrtm(Rp)
        Rf_half = sqrtm(Rf)


        if ix == 0:
            # W = F@P.T/num_sample
            U1, S1, Vh1 = sp.linalg.svd(Q,full_matrices=False)
            filter1 = Vh1
        if ix == 1:
            # W = P@P.T/num_sample
            W2 = Q@np.linalg.inv(Rp)@Q.T
            U2, S2, Vh2 = sp.linalg.svd(W2,full_matrices=False)
            filter2 = U2.T@Q@np.linalg.inv(Rp)
        if ix == 2:
            W3 = np.linalg.inv(Rf_half)@Q@np.linalg.inv(Rp_half).T
            W3 = np.real(W3)
            U3, S3, Vh3 = sp.linalg.svd(W3,full_matrices=False)
            filter3 = Vh3@np.linalg.inv(Rp_half)

    # store the results in dictionary
    results = {'U1':U1,'S1':S1,'Vh1':Vh1,'filter1':filter1,
               'U2':U2,'S2':S2,'Vh2':Vh2,'filter2':filter2,
               'U3':U3,'S3':S3,'Vh3':Vh3,'filter3':filter3,
               'Rp_half':Rp_half,'Rf_half':Rf_half,'Q':Q,
               'W2':W2, 'W3':W3,'P':P,'F':F,'Rp':Rp,'Rf':Rf}
    return results

def low_rank_filter(Q,R):
    '''
    Compute the low rank filter using the SVD of the Hankel matrix
    Q:        past, future Hankel matrix
    R:        coviarance matrix
    '''
    # preare the matrices used
    # Rp_half = np.linalg.cholesky(R)
    Rp_half = sqrtm(R)

    # SVD of the past future covariance
    U1, S1, Vh1 = sp.linalg.svd(Q,full_matrices=False)
    filter1 = Vh1

    # SVD of oneside whitened covariance
    W2 = Q@np.linalg.inv(R)@Q.T
    U2, S2, Vh2 = sp.linalg.svd(W2,full_matrices=False)
    filter2 = U2.T@Q@np.linalg.inv(R)
    # filter2 = filter2.T

    # CCA
    W3 = np.linalg.inv(Rp_half)@Q@np.linalg.inv(Rp_half).T
    W3 = np.real(W3)
    U3, S3, Vh3 = sp.linalg.svd(W3,full_matrices=False)
    filter3 = Vh3@np.linalg.inv(Rp_half)

    # store the results in dictionary
    results = {'U1':U1,'S1':S1,'Vh1':Vh1,'filter1':filter1,
               'U2':U2,'S2':S2,'Vh2':Vh2,'filter2':filter2,
               'U3':U3,'S3':S3,'Vh3':Vh3,'filter3':filter3,
               'Rp_half':Rp_half,'Q':Q,
               'W2':W2, 'W3':W3,'Rp':R}
    return results

def compare_objects(svd_result):
    '''
    Compare three objective functions
    '''
    Rp_half = svd_result['Rp_half']
    
    obj1 = np.linalg.norm(svd_result['Q']@svd_result['filter1'],ord = 'fro')
    obj2 = np.linalg.norm(np.linalg.inv(Rp_half)@svd_result['Q']@svd_result['filter2'].T,ord = 'fro')
    # obj3 = 
    return None

# reverse correlation to get the filter
def reverse_corr_filter(input_signal, response, filter_len):
    '''
    Estimate the acausal filter using reverse correlation. Solve the linear equation in the Fourier domain
    input_signal: input signal
    y: response signal
    filter_len: length of the filter
    '''

    # Step 1: Transform to Frequency Domain
    input_fft = fft(input_signal)
    response_fft = fft(response)

    # Step 3: Estimate the Filter in Frequency Domain
    estimated_filter_fft = response_fft / input_fft

    # Step 4: Transform Back to Time Domain
    estimated_filter = ifft(estimated_filter_fft)
    estimated_filter = np.real(estimated_filter[:filter_len])  # Take the real part and truncate to the filter length

    return estimated_filter


def SVD_jit(traj,lag,reg=1e-6):
    '''
    SVD analysis using the jit package, which significantly accelerate the computation
    '''
    if len(traj.shape) == 1:
        num_trail = 1
        T = traj.shape[0]
    else:
        T, num_trail = traj.shape
    seg_len = T -lag
    sep = np.round(lag/2).astype(int)

    Q,Rp,Rf = 0,0,0
    for i in range(num_trail):
        H = [traj[j:lag+j] for j in range(len(traj)-lag)]
        H = np.vstack(H)
        H = H.T
        P = np.flipud(H[:sep,:])
        F = H[sep:,:]
        # F = H[sep:,:]
        Q += jnp.matmul(F, jnp.transpose(P))
        Rp += jnp.matmul(P, jnp.transpose(P))
        Rf += jnp.matmul(F, jnp.transpose(F))
    num_sample = num_trail*seg_len
    Q = Q/num_sample
    Rp = Rp/num_sample
    Rf = Rf/num_sample

    # regularize the covariance matrix
    # Rp_half = np.linalg.cholesky(Rp + reg*np.eye(Rp.shape[0]))
    # Rf_half = np.linalg.cholesky(Rf + reg*np.eye(Rf.shape[0]))
    Rp_half = sqrtm(Rp + reg*np.eye(Rp.shape[0]))
    Rf_half = sqrtm(Rf + reg*np.eye(Rf.shape[0]))

    # SVD of the past future covariance
    U1, S1, Vh1 = sp.linalg.svd(Q,full_matrices=False)
    filter1 = Vh1

    # SVD of oneside whitened covariance
    W2 = Q@np.linalg.inv(Rp)@Q.T
    U2, S2, Vh2 = sp.linalg.svd(W2,full_matrices=False)
    # filter2 = U2.T@Q@np.linalg.inv(Rp + reg*np.eye(Rp.shape[0]))
    filter2 = U2.T@Q@np.linalg.inv(Rp)

    # CCA
    W3 = np.linalg.inv(Rf_half)@Q@np.linalg.inv(Rp_half).T
    W3 = np.real(W3)
    U3, S3, Vh3 = sp.linalg.svd(W3,full_matrices=False)
    filter3 = Vh3@np.linalg.inv(Rp_half)

    # store the results in dictionary
    results = {'U1':U1,'S1':S1,'Vh1':Vh1,'filter1':filter1,
                'U2':U2,'S2':S2,'Vh2':Vh2,'filter2':filter2,
                'U3':U3,'S3':S3,'Vh3':Vh3,'filter3':filter3,
                'Rp_half':Rp_half,'Rf_half':Rf_half,'Q':Q,
                'W2':W2, 'W3':W3,'P':P,'F':F,'Rp':Rp,'Rf':Rf}
    return results

# for spatial temporal coviarance matrix
def reshape_cov_mat(C,patch_size = 5, time_lag = 7, reverse=False):
    '''
    Reshape the spatial-temporal covariance matrix for the nearest kronecker prodcut approximation
    '''
    # get the size of the covariance matrix
    n = patch_size**2
    m = time_lag
    assert C.shape[0] == n*m, "The size of the covariance matrix is not correct"

    # define the shape of the new covariance matrix
    if reverse:
        new_cov = np.zeros((n*n, m*m))
        for i in range(n):
            for j in range(n):
                block = C[j*m:(j+1)*m,i*m:(i+1)*m]
                # flatten the block column wise and set it as the row of the new covariance matrix
                new_cov[i*n+j,:] = block.flatten('F')
                # new_cov[i*m+j,:] = np.kron(C[i*n:(i+1)*n,j*n:(j+1)*n])
    else:
        new_cov = np.zeros((m*m, n*n))
        for i in range(m):
            for j in range(m):
                block = C[j*n:(j+1)*n,i*n:(i+1)*n]
                # flatten the block column wise and set it as the row of the new covariance matrix
                new_cov[i*m+j,:] = block.flatten()
            # new_cov[i*m+j,:] = np.kron(C[i*n:(i+1)*n,j*n:(j+1)*n])

    return new_cov


# Generate up-down stairs trajectory, with up 3 steps and down 3 steps
def test_traj_step(num_steps=4,transient_length=100,plot=True):
    num_steps = 4
    transient_length = 100
    test_traj = []
    # amplitude = 1
    slope = 1
    # direction = 1
    # slopes = 2*np.random.rand(num_transients) + 0.5
    xs = np.linspace(-10,10, transient_length)

    for i in range(num_steps):
        transient = np.tanh(xs/slope)/2 + 1 + (i+1)
        test_traj.append(transient)
    for i in range(num_steps):
        transient = -np.tanh(xs/slope)/2 + 1 + (num_steps-i)
        test_traj.append(transient)
    test_traj = np.concatenate(test_traj)

    # add noise to the trajectory
    noise_std = 0.0
    test_traj += noise_std*np.random.randn(len(test_traj))
    return test_traj


### This part is written by Gizem
""" Module containing functions for truncation methods. """

def perform_ridge_regression(
    input_data: np.ndarray, output_data: np.ndarray, lag: int = 10, **kwargs
):
    """Performs ridge regression on the input and output data."""
    from sklearn.linear_model import Ridge

    alpha = kwargs.get("alpha", 5)
    # construct the delay matrix
    delay_matrix = sp.linalg.hankel(
        input_data[lag : 2 * lag], input_data[2 * lag - 1 :]
    )
    clf = Ridge(alpha=alpha)
    output_resp = output_data[lag : -lag + 1]
    ridge_regression = clf.fit(delay_matrix.T, output_resp)

    return delay_matrix, ridge_regression


def pca_method(hankel_matrix: np.ndarray) -> Tuple:
    """
    Calculates the SVD and filter using the PCA method.

    Parameters
    ----------
    hankel_matrix : np.ndarray
        Hankel matrix = FP^{T}

    Returns
    -------
    Tuple
        Tuple of U, S, Vh, and filter_pca.
    """
    U, S, Vh = sp.linalg.svd(hankel_matrix, full_matrices=False)
    filter_pca = Vh[:, :]

    return U, S, Vh, filter_pca


def cca_method(
    hankel_matrix: np.ndarray, Rf_half: np.ndarray, Rp_half: np.ndarray
) -> Tuple:
    """
    Calculates the SVD and filter using the CCA method.

    Parameters
    ----------
    hankel_matrix : np.ndarray
        Hankel matrix = FP^{T}
    Rf_half : np.ndarray
        Squared root of the future covariance matrix.
    Rp_half : np.ndarray
        Squared root of the past covariance matrix.

    Returns
    -------
    Tuple
        Tuple of cca_matrix (Rf^{-1/2}HRp^{-1/2}),
        U, S, Vh, and filter_cca.
    """
    cca_matrix = np.linalg.inv(Rf_half) @ hankel_matrix @ np.linalg.inv(Rp_half).T
    cca_matrix = np.real(cca_matrix)
    U, S, Vh = sp.linalg.svd(cca_matrix, full_matrices=False)
    filter_cca = Vh @ np.linalg.inv(Rp_half)

    return cca_matrix, U, S, Vh, filter_cca


def min_proj_method(hankel_matrix: np.ndarray, Rp: np.ndarray) -> Tuple:
    """
    Calculates the SVD and filter using the min. projection method.

    Parameters
    ----------
    hankel_matrix : np.ndarray
        Hankel matrix = FP^{T}
    Rp : np.ndarray
        Past covariance matrix.

    Returns
    -------
    Tuple
        Tuple of min_proj_matrix (H Rp^{-1} H^{T}),
        U, S, Vh, and filter_min_proj.
    """
    min_proj_matrix = hankel_matrix @ np.linalg.inv(Rp) @ hankel_matrix.T
    U, S, Vh = sp.linalg.svd(min_proj_matrix, full_matrices=False)
    filter_min_proj = U.T @ hankel_matrix @ np.linalg.inv(Rp)

    return min_proj_matrix, U, S, Vh, filter_min_proj


def create_past_future_matrices(
    trajectory: np.ndarray, memory: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create past and future matrices.
    Past trajectory is the trajectory from 0 to memory.
    Future trajectory is the trajectory
    from memory + 1 to memory + 1 + horizon.

    This method builds block matrices of past and future trajectories.
    """
    n_samples = len(trajectory) - memory - horizon
    future_matrix = np.zeros((n_samples, horizon))
    past_matrix = np.zeros((n_samples, memory))

    for i in range(0, n_samples):
        past_matrix[i, :] = trajectory[i : i + memory]
        future_matrix[i, :] = trajectory[
            i + memory + 1 : i + memory + 1 + horizon
        ]

    # Flip past matrix so that the last row is the oldest
    past_matrix = np.flip(past_matrix[:,:], axis=1)

    return past_matrix, future_matrix


def SVD_layer2(Fs, Ps, regularize=False):
    """
    Compute the SVD of three different balanced truncatio

    Fs:        future data lag matrix, a list for three method
    Ps:        past data lag matrix, a list for three method
    """
    # crete the Hankel matrix from trajectory
    for ix, (F, P) in enumerate(zip(Fs, Ps)):
        lag = P.shape[0]
        num_sample = P.shape[1]
        sep = np.round(lag / 2).astype(int)
        Q = F @ P.T / num_sample

        # Compute SVD of partially whittened covariance
        Rp = P @ P.T / num_sample
        Rf = F @ F.T / num_sample

        if regularize:
            Rp = Rp + 1e-10 * np.eye(Rp.shape[0])
            Rf = Rf + 1e-10 * np.eye(Rf.shape[0])
        Rp_half = np.linalg.cholesky(Rp)
        Rf_half = np.linalg.cholesky(Rf)

        if ix == 0:
            # W = F@P.T/num_sample
            U1, S1, Vh1 = sp.linalg.svd(Q, full_matrices=False)
            filter1 = Vh1
        if ix == 1:
            # W = P@P.T/num_sample
            W2 = Q @ np.linalg.inv(Rp) @ Q.T
            U2, S2, Vh2 = sp.linalg.svd(W2, full_matrices=False)
            filter2 = U2.T @ Q @ np.linalg.inv(Rp)
        if ix == 2:
            W3 = np.linalg.inv(Rf_half) @ Q @ np.linalg.inv(Rp_half).T
            W3 = np.real(W3)
            U3, S3, Vh3 = sp.linalg.svd(W3, full_matrices=False)
            filter3 = Vh3 @ np.linalg.inv(Rp_half)

    # store the results in dictionary
    results = {
        "U1": U1,
        "S1": S1,
        "Vh1": Vh1,
        "filter1": filter1,
        "U2": U2,
        "S2": S2,
        "Vh2": Vh2,
        "filter2": filter2,
        "U3": U3,
        "S3": S3,
        "Vh3": Vh3,
        "filter3": filter3,
        "Rp_half": Rp_half,
        "Rf_half": Rf_half,
        "Q": Q,
        "W2": W2,
        "W3": W3,
        "P": P,
        "F": F,
        "Rp": Rp,
        "Rf": Rf,
    }
    return results


def compare_objects(svd_result):
    """
    Compare three objective functions
    """
    Rp_half = svd_result["Rp_half"]

    obj1 = np.linalg.norm(svd_result["Q"] @ svd_result["filter1"], ord="fro")
    obj2 = np.linalg.norm(
        np.linalg.inv(Rp_half) @ svd_result["Q"] @ svd_result["filter2"].T, ord="fro"
    )
    # obj3 =
    return None

