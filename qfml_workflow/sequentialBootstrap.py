from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn
import threading

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import issparse
from scipy.sparse import hstack as sparse_hstack
from joblib import Parallel

from sklearn.base import is_classifier
from sklearn.base import ClassifierMixin, MultiOutputMixin, RegressorMixin, TransformerMixin

from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.utils.fixes import delayed
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import (
    check_is_fitted,
    _check_sample_weight,
    _check_feature_names_in,
)
from sklearn.utils.validation import _num_samples
#from sklearn.utils._param_validation import Interval, StrOptions

from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class SequentialRandomForest(RandomForestClassifier):
    def __init__(self, t1):
        self.t1 = t1
        super().__init__()

    def getIndMatrix(self,barIx,t1):
    # Get indicator matrix
        indM=pd.DataFrame(0,index=barIx,columns=range(t1.shape[0]))
        for i,(t0,t1) in enumerate(t1.iteritems()):indM.loc[t0:t1,i]=1.
        return indM

    def getAvgUniqueness(self,indM):
        # Average uniqueness from indicator matrix
        c=indM.sum(axis=1) # concurrency
        u=indM.div(c,axis=0) # uniqueness
        avgU=u[u>0].mean() # average uniqueness
        return avgU

    def seqBootstrap(self,indM,sLength=None):
        # Generate a sample via sequential bootstrap
        if sLength is None:sLength=indM.shape[1]
        phi=[]
        while len(phi)<sLength:
            avgU=pd.Series()
            for i in indM:
                indM_=indM[phi+[i]] # reduce indM
                avgU.loc[i]=self.getAvgUniqueness(indM_).iloc[-1]
            prob=avgU/avgU.sum() # draw prob
            phi+=[np.random.choice(indM.columns,p=prob)]
        return phi
    
    def _generate_sample_indicies(self,random_state, n_samples, n_samples_bootstrap):
        indM = self.getIndMatrix(range(n_samples), self.t1)
        return self.seqBootstrap(indM=indM, sLength=n_samples_bootstrap)
    

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        super()._validate_params()
        #self.

        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X, y = super()._validate_data(
            X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = super()._get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._validate_estimator()
        if isinstance(self, (super().RandomForestRegressor, super().ExtraTreesRegressor)):
            # TODO(1.3): Remove "auto"
            if self.max_features == "auto":
                warn(
                    "`max_features='auto'` has been deprecated in 1.1 "
                    "and will be removed in 1.3. To keep the past behaviour, "
                    "explicitly set `max_features=1.0` or remove this "
                    "parameter as it is also the default value for "
                    "RandomForestRegressors and ExtraTreesRegressors.",
                    FutureWarning,
                )
        elif isinstance(self, (RandomForestClassifier, super().ExtraTreesClassifier)):
            # TODO(1.3): Remove "auto"
            if self.max_features == "auto":
                warn(
                    "`max_features='auto'` has been deprecated in 1.1 "
                    "and will be removed in 1.3. To keep the past behaviour, "
                    "explicitly set `max_features='sqrt'` or remove this "
                    "parameter as it is also the default value for "
                    "RandomForestClassifiers and ExtraTreesClassifiers.",
                    FutureWarning,
                )

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(super().MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(super()._parallel_build_trees)(
                    t,
                    self.bootstrap,
                    X,
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            y_type = type_of_target(y)
            if y_type in ("multiclass-multioutput", "unknown"):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )
            self._set_oob_score_and_attributes(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

