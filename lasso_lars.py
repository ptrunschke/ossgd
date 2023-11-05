"""
An implementation of the LASSO-LARS algorithm that is mathematically more rigorous than the one implemented in scikit-learn.
"""
import warnings

import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from bisect import insort_left


def min_pos(X):
    """
    Find the minimum value of an array over positive values
    Returns a huge value if none of the values are positive
    """
    return np.min(X[X > 0], initial=np.inf)


def rotg(x):
    """
    Compute cos and sin entries for Givens plane rotation.

    Given the Cartesian coordinates x of a point in two dimensions, the function returns the parameters c and s
    associated with the Givens rotation and modifies x accordingly. The parameters c and s define a unitary matrix such that:

        ┌      ┐      ┌   ┐
        │ c  s │      │ r │
        │-s  c │ x == │ 0 │
        └      ┘      └   ┘

    x is modified to contain x[0] = r >= 0 and x[1] = 0.
    """
    if np.all(x==0):
        return np.array([1,0])  # c, s
    else:
        r = np.linalg.norm(x)
        cs = x/r
        x[:] = (r, 0)
        return cs


def rot(xy, cs):
    """
    Perform a Givens plane rotation for the cos and sin entries c and s.

    Given two vectors x and y, each vector element of these vectors is replaced as follows:
        xₖ = c*xₖ + s*yₖ
        yₖ = c*yₖ - s*xₖ
    """
    assert xy.ndim == 2 and xy.shape[0] == 2 and cs.shape == (2,)

    c,s = cs
    x,y = xy
    tmp = c*x + s*y
    y[:] = c*y - s*x
    x[:] = tmp


def cholesky_delete(L, go_out):
    """
    Remove a row and column from the cholesky factorization.
    """
    # Delete row go_out.
    L[go_out:-1] = L[go_out+1:]
    # The resulting matrix has non-zero elements in the off-diagonal entries L[i,i+1] for i>=go_out.

    n = L.shape[0]
    for i in range(go_out, n-1):
        # Rotate the two column vectors L[i:,i] and L[i:,i+1].
        #NOTE: rotg gives the rotation that provides a positive diagonal entry.
        cs = rotg(L[i,i:i+2])
        rot(L[i+1:,i:i+2].T, cs)


def est_cond_triangular(L):
    """
    Lower bound on the condition number of a triangular matrix w.r.t. the row-sum norm.

    References:
    -----------
    .. [1] https://en.wikipedia.org/wiki/Condition_number
    .. [2] https://github.com/PetterS/SuiteSparse/blob/master/CHOLMOD/Cholesky/cholmod_rcond.c
    """
    assert L.ndim == 2 and L.shape[0] == L.shape[1]
    if L.shape[0] == 0: return 0
    diagL = abs(np.diag(L))
    return np.max(diagL) / np.min(diagL)


class ConvergenceWarning(UserWarning):
    pass


class LarsState(object):
    """
    References
    ----------
    .. [1] "Least Angle Regression", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf
    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_
    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_

    """
    def __init__(self, XTX, XTy):
        self.tiny = np.finfo(np.float64).tiny  # Used to avoid division by 0 without perturbing other computations too much. (Note that x+self.tiny != x only if x is tiny itself.)
        self.condition_threshold = 1e16  # 1e8**2

        n_features, = np.shape(XTy)
        assert XTX.shape == (n_features, n_features)
        self.XTX = XTX

        self.coef = np.zeros((0,))
        self.active = []  #NOTE: COEF AND ACTIVE ALWAYS HAVE THE SAME LENGTH
        self.inactive = list(range(n_features))  #NOTE: ALWAYS SORTED

        self.Cov = np.asarray(XTy)          # Cov[n_active:] contains the covariances of the inactive covariates. Cov[:n_active] = 0.
        self.signCov = np.empty(n_features, dtype=np.int8)  # sign_active[:n_active] holds the sign of the covariance of active covariates.
        #TODO: Maybe the variable signCov can be eliminated.
        self.C = np.max(np.fabs(self.Cov))  # covariance of the active covariates
        self.CData = np.inf
        self.prevCData = np.inf

        initial_size = min(10, n_features)
        self.L = np.zeros((initial_size, initial_size), dtype=XTy.dtype)  # self.L holds the Cholesky factorization. Only the lower part is referenced.
        self.solve_cholesky, = get_lapack_funcs(('potrs',), (self.L,))
        self.G = np.zeros((initial_size, n_features), dtype=XTy.dtype)    # G[i] holds XTX[active[i]].

        self.drop = False

    def add_L(self):
        n_active = len(self.active)
        self.L[n_active, :n_active] = self.G[n_active,self.active]  # Y.T@y
        linalg.solve_triangular(self.L[:n_active, :n_active], self.L[n_active, :n_active], lower=1, overwrite_b=True, check_finite=False)  # Solve L@w = Y.T@y. `overwrite_b=True` implies that L[n_active, :n_active] will contain the solution.

    def check_condition_threshold(self):
        n_active = len(self.active)
        assert np.all(np.isfinite(self.G[:n_active]))
        assert np.all(np.isfinite(self.L[:n_active,:n_active]))
        assert est_cond_triangular(self.L[:n_active,:n_active])**2 <= self.condition_threshold

    def add_index(self):
        """
        Update the cholesky decomposition for the Gram matrix L@L.T = X.T@X = XTX.

        Let Y = X[:,:n_active] and L be the Cholesky factor of Y.T@Y.
        Compute the factor L_new of Y_new = X[:,:n_active+1] as
                    ┌      ┐
                    │ L  0 │
            L_new = │      │.
                    │ w  z │
                    └      ┘
        Note that Y_new = [ Y y ] with y = X[:,n_active].
        Writing out the condition L_new@L_new.T = Y.T@Y yields the conditions

            L@w = Y.T@y    and    z**2 = y.T@y - w.T@w.

        """
        self.check_condition_threshold()

        while True:
            C_idx_inactive = np.argmax(np.fabs(self.Cov[self.inactive]))
            C_idx = self.inactive[C_idx_inactive]
            if abs(self.Cov[C_idx]) == 0:
                #NOTE: This is probably an index that has been removed earlier.
                raise ConvergenceWarning("Early stopping the lars path, as every regressor with nonzero covariance would make the system unstable.")

            self.prevCData, self.CData = self.CData, abs(self.Cov[C_idx])
            #NOTE: lars.C can be updated in every step via self.C -= gamma * AA. This however is numerically unstable.
            #      The numerical values of the covariances (y.T @ X @ coef) should all be equal but tend to diverge.
            #      In itself this may not pose a problem. But when these covariances fall below those in self.Cov this means
            #      that the updates of Cov bring a numerical error that is greater than the remaining correlation with the regressors.
            # print(f"{self.C:.3e} == {self.CData:.3e} <= {self.prevCData:.3e}")
            if self.CData > self.prevCData:
                raise ConvergenceWarning("Early stopping the lars path, as the remaining covariances fall below the numerical error.")


            self.signCov[C_idx] = np.sign(self.Cov[C_idx])
            self.Cov[C_idx] = 0

            n_active = len(self.active)

            # Ensure that G and L have the capacity for a new row.
            assert self.G.shape[0] == self.L.shape[0]
            if self.G.shape[0] <= n_active:
                n_features = self.Cov.shape[0]
                assert self.G.shape == (n_active, n_features)
                assert self.L.shape == (n_active, n_active)
                old_capacity = n_active
                new_capacity = min(2*n_active, n_features)
                #NOTE: G and L are padded with zeros. Otherwise they could contain nan's or inf's that would result in a failure of check_condition_threshold().
                self.G = np.pad(self.G, ((0, new_capacity-old_capacity), (0, 0)))
                self.L = np.pad(self.L, ((0, new_capacity-old_capacity),))

            self.G[n_active] = self.XTX[C_idx]  # Y.T@y
            if n_active > 0:
                self.add_L()
            yTy = self.G[n_active, C_idx]
            wTw = np.linalg.norm(self.L[n_active,:n_active]) ** 2
            self.L[n_active, n_active] = np.sqrt(max(yTy - wTw, 0)) + self.tiny

            self.coef = np.append(self.coef, 0)
            self.active.append(C_idx)
            self.inactive.pop(C_idx_inactive)

            idcs_rem = self.remove_ill_conditioned_indices()
            if idcs_rem == [C_idx]:  # The last added regressor causes the system to become ill-conditioned.
                continue             # We can pretend like it never existed by starting over without increasing n_active.
            else:                    # In any other case we have added a new index (although `remove_ill_conditioned_indices` may have removed another).
                break

        self.check_condition_threshold()


    def remove_ill_conditioned_indices(self):
        """
        This function successively removes the regressors with smallest eigenvalue until the condition number of `self.G` deceeds `self.condition_threshold`.
        The corresponding indices have to be removed for good and can not be added back at a later time.
        This is because LARS adds regressors based on their covariance which is currently the largest among all active regressors.
        It returns a list of all indices that were removed.
        """
        idcs_rem = []
        eigs_rem = []
        n_active = len(self.active)
        while n_active > 1:
            #NOTE: The choice of the condition threshold is imporant and influences the quality of our solution. (It directly affects which regressors are added or dropped for good...)
            if est_cond_triangular(self.L[:n_active,:n_active])**2 <= self.condition_threshold:  # The square appears since we need to solve the system L@x=y twice.
                break
            # Select the index of the smallest eigenvalue. (The eigenvalues of a triangular matrix are the diagonal entries.)
            # We do not take the absolute value, because L should have only positive eigenvalues. (By Sylvesters law of inertia and since the Gramian is positive definite.)
            ii = np.argmin(np.diag(self.L[:n_active,:n_active]))

            cholesky_delete(self.L[:n_active, :n_active], ii)
            for i in range(ii, n_active-1):
                self.G[i] = self.G[i+1]

            self.coef = np.delete(self.coef, ii)
            C_idx = self.active.pop(ii)
            insort_left(self.inactive, C_idx)
            self.Cov[C_idx] = 0  # Drop the index for good.

            idcs_rem.append(C_idx)
            eigs_rem.append(self.L[n_active-1, n_active-1])

            n_active -= 1

        if len(idcs_rem) > 0:
            n_active = len(self.active)
            max_eig = np.max(np.diag(self.L[:n_active, :n_active]))
            rel_eigs_rem = "{" + ",".join(f"{idx}: {eig/max_eig:.2e}" for idx,eig in zip(idcs_rem, eigs_rem)) + "}"
            warnings.warn(f"Regressors in active set degenerate. Removing those with small relative eigenvalues: {rel_eigs_rem}.", ConvergenceWarning)

        return idcs_rem

    def remove_index(self):
        self.check_condition_threshold()

        n_active = len(self.active)
        idx = np.nonzero(np.fabs(self.coef) < 1e-12)[0][::-1]
        if len(idx) == 0:
            raise ConvergenceWarning(f"Early stopping the lars path, as a regressor that ought to be removed from the active set has nonzero coefficient.")
        assert np.all(idx[:-1] > idx[1:])

        for ii in idx:
            if n_active <= 1:
                break
            cholesky_delete(self.L[:n_active, :n_active], ii)
            for i in range(ii, n_active-1):
                self.G[i] = self.G[i+1]
            self.coef = np.delete(self.coef, ii)
            C_idx = self.active.pop(ii)
            insort_left(self.inactive, C_idx)
            self.Cov[C_idx] = self.C*self.signCov[C_idx]
            n_active -= 1

        self.remove_ill_conditioned_indices()
        self.check_condition_threshold()
        self.drop = False

    def step(self):
        self.check_condition_threshold()

        # least squares solution
        n_active = len(self.active)
        least_squares, _ = self.solve_cholesky(self.L[:n_active, :n_active], self.signCov[self.active], lower=True)
        AA = 1. / np.sqrt(np.sum(least_squares * self.signCov[self.active]))
        assert 0 < AA and np.isfinite(AA)
        least_squares *= AA  # w as defined in (2.6)

        # equiangular direction of variables in the active set
        corr_eq_dir = least_squares @ self.G[:n_active, self.inactive]  # a as defined in (2.12)

        gamma_hat = min(min_pos((self.C-self.Cov[self.inactive]) / (AA-corr_eq_dir+self.tiny)),
                        min_pos((self.C+self.Cov[self.inactive]) / (AA+corr_eq_dir+self.tiny)))  # \hat{\gamma} as defined in (2.13)
        gamma_hat = min(gamma_hat, self.C/AA)  # Stabilizes the algorithm since C/AA is the maximal step size (self.C - gamma_hat*AA == 0).

        bOd = - self.coef / (least_squares+self.tiny)  # - \hat{\beta} / \hat{d}
        gamma_tilde = min_pos(bOd)  # \tilde{\gamma} as defined in (3.5)
        gamma = min(gamma_tilde, gamma_hat)

        # update coefficient
        self.coef += gamma * least_squares
        # update correlations
        self.Cov[self.inactive] -= gamma * corr_eq_dir
        self.C -= gamma * AA

        self.drop = gamma_tilde < gamma_hat


def lars_index_path(X, y):
    n_samples, n_features = X.shape
    assert y.shape == (n_samples,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lars = LarsState(X.T@X, X.T@y)
        while len(lars.active) < n_features:
            try:
                if not lars.drop:
                    lars.add_index()
                lars.step()
                if lars.drop:
                    lars.remove_index()
            except ConvergenceWarning as w:
                break
            yield lars.active[:]
