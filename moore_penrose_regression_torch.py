# import numpy as np
# from scipy.linalg import svd

import torch
from torch.linalg import svd

# eps64 = np.finfo("float").eps
# eps32 = np.finfo("float32").eps
# eps = 1e-6 

eps64 = torch.finfo(torch.float64).eps
eps32 = torch.finfo(torch.float32).eps
eps = 1e-6

def moore_penrose_regression_torch(X, Y, device="cuda", dtype=torch.double):
    """
    X (Pytorch tensor of floats): 2d column matrix 
    Y (Pytorch tensor of floats): 2d column matrix
    """
    assert X.dtype==dtype, f"Type of X {X.dtype} does not match chosen dtype {dtype}"
    assert Y.dtype==dtype, f"Type of Y {Y.dtype} does not match chosen dtype {dtype}"
    # assert X.device==device, f"Device of X {X.device} does not match chosen device {device}"
    # assert Y.device==device, f"Device of Y {Y.device} does not match chosen device {device}"
    # linear regression but more stable (singular value decomposition)
    k = eps32 #eps #1e-5 #0 #eps64 #1e-11 #1e-9 #
    Nsamp = X.shape[0]
    Xreg = torch.hstack(
        [
            torch.ones([Nsamp, 1], dtype=dtype, device=device), 
            X
        ]
    )
    if Xreg.dtype == torch.float and (device == "cuda" or device==torch.device("cuda")):
        print("Performing SVD on CPU for extra precision")
        Xreg = Xreg.double().to("cpu")
        U, Sig, Vt = svd(Xreg, full_matrices = True) #False) #
        Y = Y.double().to("cpu")
        convert_back = True
    else:
        U, Sig, Vt = svd(Xreg, full_matrices = True) #False) #
        convert_back=False
    V = Vt.T
    # numerically stabalize Sig
    # option 1: clamp
    sign = torch.sign(Sig)
    sign[sign==0] = 1
    idx = torch.abs(Sig) < k
    Sig[idx] = k * sign[idx]
    # # option 2: add everywhere
    # sign = torch.sign(Sig)
    # sign[sign==0] = 1
    # Sig = Sig + sign*k
    # end numerically stabalize Sig
    Sig_pseudoinverse = torch.zeros(
        Xreg.shape[::-1], dtype=torch.double, device="cpu" if convert_back else device
    )
    Sig_pseudoinverse.diagonal().copy_(1/Sig)
    # Sig_pseudoinverse = np.linalg.pinv(Sig, rcond = k)
    Beta = V @ Sig_pseudoinverse @ U.T @ Y

    if convert_back:
        Beta = Beta.float().to("cuda")
    return(Beta.to(device))

