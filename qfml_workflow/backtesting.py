import itertools as itt
import numbers
import numpy as np
import pandas as pd
from qfml_workflow import betSizing

from abc import abstractmethod
from typing import Iterable, Tuple, List

def embargoFromPercent(dates, percent):
    embIdx = int(dates.shape[0] * 0.01)
    tdel = dates[embIdx-1] - dates[0]
    
    return tdel


class BaseTimeSeriesCrossValidator:
    """
    Abstract class for time series cross-validation.

    Time series cross-validation requires each sample has a prediction time pred_time, at which the features are used to
    predict the response, and an evaluation time eval_time, at which the response is known and the error can be
    computed. Importantly, it means that unlike in standard sklearn cross-validation, the samples X, response y,
    pred_times and eval_times must all be pandas dataframe/series having the same index. It is also assumed that the
    samples are time-ordered with respect to the prediction time (i.e. pred_times is non-decreasing).

    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.

    """
    def __init__(self, n_splits=10):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(f"The number of folds must be of Integral type. {n_splits} of type {type(n_splits)}"
                             f" was passed.")
        n_splits = int(n_splits)
        if n_splits <= 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting n_splits = 2 "
                             f"or more, got n_splits = {n_splits}.")
        self.n_splits = n_splits
        self.pred_times = None
        self.eval_times = None
        self.indices = None

    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series = None,
              pred_times: pd.Series = None, eval_times: pd.Series = None):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series):
            raise ValueError('X should be a pandas DataFrame/Series.')
        if not isinstance(y, pd.Series) and y is not None:
            raise ValueError('y should be a pandas Series.')
        if not isinstance(pred_times, pd.Series):
            raise ValueError('pred_times should be a pandas Series.')
        if not isinstance(eval_times, pd.Series):
            raise ValueError('eval_times should be a pandas Series.')
        if y is not None and (X.index == y.index).sum() != len(y):
            raise ValueError('X and y must have the same index')
        if (X.index == pred_times.index).sum() != len(pred_times):
            raise ValueError('X and pred_times must have the same index')
        if (X.index == eval_times.index).sum() != len(eval_times):
            raise ValueError('X and eval_times must have the same index')

        '''
        # TODO: FIXED #################################################
        if not pred_times.equals(pred_times.sort_values()):
            raise ValueError('pred_times should be sorted')
        bool_series = eval_times.eq(eval_times.sort_values())
        if not len(bool_series[bool_series == False]) == 0:
            raise ValueError('eval_times should be sorted')
        '''
            
        self.pred_times = pred_times
        self.eval_times = eval_times
        self.indices = np.arange(X.shape[0])


class PurgedWalkForwardCV(BaseTimeSeriesCrossValidator):
    """
    Purged walk-forward cross-validation

    As described in Advances in financial machine learning, Marcos Lopez de Prado, 2018.

    The samples are decomposed into n_splits folds containing equal numbers of samples, without shuffling. In each cross
    validation round, n_test_splits contiguous folds are used as the test set, while the train set consists in between
    min_train_splits and max_train_splits immediately preceding folds.

    Each sample should be tagged with a prediction time pred_time and an evaluation time eval_time. The split is such
    that the intervals [pred_times, eval_times] associated to samples in the train and test set do not overlap. (The
    overlapping samples are dropped.)

    With split_by_times = True in the split method, it is also possible to split the samples in folds spanning equal
    time intervals (using the prediction time as a time tag), instead of folds containing equal numbers of samples.

    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.

    n_test_splits : int, default = 1
        Number of folds used in the test set. Must be at least 1.

    min_train_splits: int, default = 2
        Minimal number of folds to be used in the train set.

    max_train_splits: int, default = None
        Maximal number of folds to be used in the train set. If None, there is no upper limit.

    """
    def __init__(self, n_splits=10, n_test_splits=1, min_train_splits=2, max_train_splits=None):
        super().__init__(n_splits)
        if not isinstance(n_test_splits, numbers.Integral):
            raise ValueError(f"The number of test folds must be of Integral type. {n_test_splits} of type "
                             f"{type(n_test_splits)} was passed.")
        n_test_splits = int(n_test_splits)
        if n_test_splits <= 0 or n_test_splits >= self.n_splits - 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting "
                             f"n_test_splits between 1 and n_splits - 1, got n_test_splits = {n_test_splits}.")
        self.n_test_splits = n_test_splits

        if not isinstance(min_train_splits, numbers.Integral):
            raise ValueError(f"The minimal number of train folds must be of Integral type. {min_train_splits} of type "
                             f"{type(min_train_splits)} was passed.")
        min_train_splits = int(min_train_splits)
        if min_train_splits <= 0 or min_train_splits >= self.n_splits - self.n_test_splits:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting "
                             f"min_train_splits between 1 and n_splits - n_test_splits, got min_train_splits = "
                             f"{min_train_splits}.")
        self.min_train_splits = min_train_splits

        if max_train_splits is None:
            max_train_splits = self.n_splits - self.n_test_splits
        if not isinstance(max_train_splits, numbers.Integral):
            raise ValueError(f"The maximal number of train folds must be of Integral type. {max_train_splits} of type "
                             f"{type(max_train_splits)} was passed.")
        max_train_splits = int(max_train_splits)
        if max_train_splits <= 0 or max_train_splits > self.n_splits - self.n_test_splits:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting "
                             f"max_train_split between 1 and n_splits - n_test_splits, got max_train_split = "
                             f"{max_train_splits}.")
        self.max_train_splits = max_train_splits
        self.fold_bounds = []

    def split(self, X: pd.DataFrame, y: pd.Series = None, pred_times: pd.Series = None, eval_times: pd.Series = None,
              split_by_time: bool = False) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield the indices of the train and test sets.

        Although the samples are passed in the form of a pandas dataframe, the indices returned are position indices,
        not labels.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            Samples. Only used to extract n_samples.

        y : pd.Series, not used, inherited from _BaseKFold

        pred_times : pd.Series, shape (n_samples,), required
            Times at which predictions are made. pred_times.index has to coincide with X.index.

        eval_times : pd.Series, shape (n_samples,), required
            Times at which the response becomes available and the error can be computed. eval_times.index has to
            coincide with X.index.

        split_by_time: bool
            If False, the folds contain an (approximately) equal number of samples. If True, the folds span identical
            time intervals.

        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.

        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.

        """
        super().split(X, y, pred_times, eval_times)

        # Fold boundaries
        self.fold_bounds = compute_fold_bounds(self, split_by_time)

        count_folds = 0
        for fold_bound in self.fold_bounds:
            if count_folds < self.min_train_splits:
                count_folds = count_folds + 1
                continue
            if self.n_splits - count_folds < self.n_test_splits:
                break
            # Computes the bounds of the test set, and the corresponding indices
            test_indices = self.compute_test_set(fold_bound, count_folds)
            # Computes the train set indices
            train_indices = self.compute_train_set(fold_bound, count_folds)

            count_folds = count_folds + 1
            yield train_indices, test_indices

    def compute_train_set(self, fold_bound: int, count_folds: int) -> np.ndarray:
        """
        Compute the position indices of samples in the train set.

        Parameters
        ----------
        fold_bound : int
            Bound between the train set and the test set.

        count_folds : int
            The number (starting at 0) of the first fold in the test set.

        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.

        """
        if count_folds > self.max_train_splits:
            start_train = self.fold_bounds[count_folds - self.max_train_splits]
        else:
            start_train = 0
        train_indices = np.arange(start_train, fold_bound)
        # Purge
        train_indices = purge(self, train_indices, fold_bound, self.indices[-1])
        return train_indices

    def compute_test_set(self, fold_bound: int, count_folds: int) -> np.ndarray:
        """
        Compute the indices of the samples in the test set.

        Parameters
        ----------
        fold_bound : int
            Bound between the train set and the test set.

        count_folds : int
            The number (starting at 0) of the first fold in the test set.

        Returns
        -------
        test_indices: np.ndarray
            A numpy array containing the test indices.

        """
        if self.n_splits - count_folds > self.n_test_splits:
            end_test = self.fold_bounds[count_folds + self.n_test_splits]
        else:
            end_test = self.indices[-1] + 1
        return np.arange(fold_bound, end_test)


class CombPurgedKFoldCV(BaseTimeSeriesCrossValidator):
    """
    Purged and embargoed combinatorial cross-validation

    As described in Advances in financial machine learning, Marcos Lopez de Prado, 2018.

    The samples are decomposed into n_splits folds containing equal numbers of samples, without shuffling. In each cross
    validation round, n_test_splits folds are used as the test set, while the other folds are used as the train set.
    There are as many rounds as n_test_splits folds among the n_splits folds.

    Each sample should be tagged with a prediction time pred_time and an evaluation time eval_time. The split is such
    that the intervals [pred_times, eval_times] associated to samples in the train and test set do not overlap. (The
    overlapping samples are dropped.) In addition, an "embargo" period is defined, giving the minimal time between an
    evaluation time in the test set and a prediction time in the training set. This is to avoid, in the presence of
    temporal correlation, a contamination of the test set by the train set.

    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.

    n_test_splits : int, default=2
        Number of folds used in the test set. Must be at least 1.

    embargo_td : pd.Timedelta, default=0
        Embargo period (see explanations above).

    """
    def __init__(self, n_splits=10, n_test_splits=2, embargo_td=pd.Timedelta(minutes=0)):
        super().__init__(n_splits)
        if not isinstance(n_test_splits, numbers.Integral):
            raise ValueError(f"The number of test folds must be of Integral type. {n_test_splits} of type "
                             f"{type(n_test_splits)} was passed.")
        n_test_splits = int(n_test_splits)
        if n_test_splits <= 0 or n_test_splits > self.n_splits - 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting "
                             f"n_test_splits between 1 and n_splits - 1, got n_test_splits = {n_test_splits}.")
        self.n_test_splits = n_test_splits
        if not isinstance(embargo_td, pd.Timedelta):
            raise ValueError(f"The embargo time should be of type Pandas Timedelta. {embargo_td} of type "
                             f"{type(embargo_td)} was passed.")
        if embargo_td < pd.Timedelta(minutes=0):
            raise ValueError(f"The embargo time should be positive, got embargo = {embargo_td}.")
        self.embargo_td = embargo_td

    def split(self, X: pd.DataFrame, y: pd.Series = None,
              pred_times: pd.Series = None, eval_times: pd.Series = None) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield the indices of the train and test sets.

        Although the samples are passed in the form of a pandas dataframe, the indices returned are position indices,
        not labels.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            Samples. Only used to extract n_samples.

        y : pd.Series, not used, inherited from _BaseKFold

        pred_times : pd.Series, shape (n_samples,), required
            Times at which predictions are made. pred_times.index has to coincide with X.index.

        eval_times : pd.Series, shape (n_samples,), required
            Times at which the response becomes available and the error can be computed. eval_times.index has to
            coincide with X.index.

        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.

        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.

        """
        super().split(X, y, pred_times, eval_times)

        # Fold boundaries
        fold_bounds = [(fold[0], fold[-1] + 1) for fold in np.array_split(self.indices, self.n_splits)]
        # List of all combinations of n_test_splits folds selected to become test sets
        selected_fold_bounds = list(itt.combinations(fold_bounds, self.n_test_splits))
        # In order for the first round to have its whole test set at the end of the dataset
        selected_fold_bounds.reverse()

        for fold_bound_list in selected_fold_bounds:
            # Computes the bounds of the test set, and the corresponding indices
            test_fold_bounds, test_indices = self.compute_test_set(fold_bound_list)
            # Computes the train set indices
            train_indices = self.compute_train_set(test_fold_bounds, test_indices)

            yield train_indices, test_indices

    def compute_train_set(self, test_fold_bounds: List[Tuple[int, int]], test_indices: np.ndarray) -> np.ndarray:
        """
        Compute the position indices of samples in the train set.

        Parameters
        ----------
        test_fold_bounds : List of tuples of position indices
            Each tuple records the bounds of a block of indices in the test set.

        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.

        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.

        """
        # As a first approximation, the train set is the complement of the test set
        train_indices = np.setdiff1d(self.indices, test_indices)
        # But we now have to purge and embargo
        for test_fold_start, test_fold_end in test_fold_bounds:
            # Purge
            train_indices = purge(self, train_indices, test_fold_start, test_fold_end)
            # Embargo
            train_indices = embargo(self, train_indices, test_indices, test_fold_end)
        return train_indices

    def compute_test_set(self, fold_bound_list: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Compute the indices of the samples in the test set.

        Parameters
        ----------
        fold_bound_list: List of tuples of position indices
            Each tuple records the bounds of the folds belonging to the test set.

        Returns
        -------
        test_fold_bounds: List of tuples of position indices
            Like fold_bound_list, but with the neighboring folds in the test set merged.

        test_indices: np.ndarray
            A numpy array containing the test indices.

        """
        test_indices = np.empty(0)
        test_fold_bounds = []
        for fold_start, fold_end in fold_bound_list:
            # Records the boundaries of the current test split
            if not test_fold_bounds or fold_start != test_fold_bounds[-1][-1]:
                test_fold_bounds.append((fold_start, fold_end))
            # If the current test split is contiguous to the previous one, simply updates the endpoint
            elif fold_start == test_fold_bounds[-1][-1]:
                test_fold_bounds[-1] = (test_fold_bounds[-1][0], fold_end)
            test_indices = np.union1d(test_indices, self.indices[fold_start:fold_end]).astype(int)
        return test_fold_bounds, test_indices


def compute_fold_bounds(cv: BaseTimeSeriesCrossValidator, split_by_time: bool) -> List[int]:
    """
    Compute a list containing the fold (left) boundaries.

    Parameters
    ----------
    cv: BaseTimeSeriesCrossValidator
        Cross-validation object for which the bounds need to be computed.
    split_by_time: bool
        If False, the folds contain an (approximately) equal number of samples. If True, the folds span identical
        time intervals.
    """
    if split_by_time:
        full_time_span = cv.pred_times.max() - cv.pred_times.min()
        fold_time_span = full_time_span / cv.n_splits
        fold_bounds_times = [cv.pred_times.iloc[0] + fold_time_span * n for n in range(cv.n_splits)]
        return cv.pred_times.searchsorted(fold_bounds_times)
    else:
        return [fold[0] for fold in np.array_split(cv.indices, cv.n_splits)]


def embargo(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray,
            test_indices: np.ndarray, test_fold_end: int) -> np.ndarray:
    """
    Apply the embargo procedure to part of the train set.

    This amounts to dropping the train set samples whose prediction time occurs within self.embargo_dt of the test
    set sample evaluation times. This method applies the embargo only to the part of the training set immediately
    following the end of the test set determined by test_fold_end.

    Parameters
    ----------
    cv: Cross-validation class
        Needs to have the attributes cv.pred_times, cv.eval_times, cv.embargo_dt and cv.indices.

    train_indices: np.ndarray
        A numpy array containing all the indices of the samples currently included in the train set.

    test_indices : np.ndarray
        A numpy array containing all the indices of the samples in the test set.

    test_fold_end : int
        Index corresponding to the end of a test set block.

    Returns
    -------
    train_indices: np.ndarray
        The same array, with the indices subject to embargo removed.

    """
    if not hasattr(cv, 'embargo_td'):
        raise ValueError("The passed cross-validation object should have a member cv.embargo_td defining the embargo"
                         "time.")
    last_test_eval_time = cv.eval_times.iloc[test_indices[test_indices <= test_fold_end]].max()
    min_train_index = len(cv.pred_times[cv.pred_times <= last_test_eval_time + cv.embargo_td])
    if min_train_index < cv.indices.shape[0]:
        allowed_indices = np.concatenate((cv.indices[:test_fold_end], cv.indices[min_train_index:]))
        train_indices = np.intersect1d(train_indices, allowed_indices)
    return train_indices


def purge(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray,
          test_fold_start: int, test_fold_end: int) -> np.ndarray:
    """
    Purge part of the train set.

    Given a left boundary index test_fold_start of the test set, this method removes from the train set all the
    samples whose evaluation time is posterior to the prediction time of the first test sample after the boundary.

    Parameters
    ----------
    cv: Cross-validation class
        Needs to have the attributes cv.pred_times, cv.eval_times and cv.indices.

    train_indices: np.ndarray
        A numpy array containing all the indices of the samples currently included in the train set.

    test_fold_start : int
        Index corresponding to the start of a test set block.

    test_fold_end : int
        Index corresponding to the end of the same test set block.

    Returns
    -------
    train_indices: np.ndarray
        A numpy array containing the train indices purged at test_fold_start.

    """
    time_test_fold_start = cv.pred_times.iloc[test_fold_start]
    # The train indices before the start of the test fold, purged.
    train_indices_1 = np.intersect1d(train_indices, cv.indices[cv.eval_times < time_test_fold_start])
    # The train indices after the end of the test fold.
    train_indices_2 = np.intersect1d(train_indices, cv.indices[test_fold_end:])
    return np.concatenate((train_indices_1, train_indices_2))


def runCSCVBacktesting(cv, sm, classifierModel,  X, y,  n_groups, k, paths, prediction_times,evaluation_times, lw=5):
    # generate the combinations
    # n groups
    #N = n_groups + 1
    N = n_groups
    
    test_groups = np.array(list(itt.combinations(np.arange(N), k))).reshape(-1, k)
    n_splits = len(test_groups)
    n_paths = n_splits * k // n_groups 

    t_span = len(X.index)
    group_num = np.arange(t_span) // (t_span // N )
    group_num[group_num == N] = N -1
    #group_num[group_num == N] = f'G{N-1}' 

    group_strs = ['G'+str(n) for n in range(N)]
    mapping = dict(zip(np.unique(group_num), group_strs))


    # Map the numeric index to string labels
    string_index = [mapping[value] for value in group_num]

    testSets = np.zeros((N, n_splits))
    splitPredictions = pd.DataFrame({'groupIndex':string_index},index=X.index)

    is_test_group = np.full((N, n_splits), fill_value=False)
    for col, pair in enumerate(test_groups):
            i, j = pair
            is_test_group[[i, j], col] = True

    # map test sets to paths
    path_folds = np.full((N, n_paths), fill_value=np.nan)

    for i in range(n_paths):
            for j in range(N):
                s_idx = is_test_group[j, :].argmax().astype(int)
                path_folds[j, i] = s_idx
                is_test_group[j, s_idx] = False

    #sm = SimpleModel(X, y)
    #X, y, tbEvents = sm.createBenchmarkModel()
    # Generate the training/testing visualizations for each CV split
    print((X.index == y.index).sum() == (len(y)))
    for ii, (tr, tt) in enumerate(cv.split(X, y, pred_times=prediction_times, eval_times=evaluation_times)):
    #for ii, (tr, tt) in enumerate(cv.split(X.reindex(evaluation_times.index), y.reindex(evaluation_times.index), prediction_times.reindex(evaluation_times.index), eval_times)):

        # print('fold', ii, '\n')
        # print(tr, '\n')
        # print(tt, '\n')
       # splitPredictions['S'+str(ii)] = model.predict(X.iloc[tt, :])
        X_train = X.iloc[tr].dropna()
        y_train = y.iloc[tr].dropna()
        X_test = X.iloc[tt].dropna()

        splitIndex = n_splits - ii - 1


        splitLabels = sm.makePredictions(X_train, y_train, X_test, classifierModel)
        splitPredictions.loc[splitLabels.index, 'S' + str(splitIndex)] = splitLabels

    splitPredictions = splitPredictions[['groupIndex'] + [f'S{i}' for i in range(n_splits)]]
        
    return splitPredictions, path_folds

    # add groupIndex as multiindex
    # index by groupnumber, splitnumber


def runCSCVBacktestingBetSizing(cv, sm, classifierModel,  X, y,  n_groups, k, paths, prediction_times,evaluation_times, tripleBarrierEvents, stepSize=0.05, numClasses=2, lw=5):
    # generate the combinations
    # n groups
    #N = n_groups + 1
    N = n_groups
    
    test_groups = np.array(list(itt.combinations(np.arange(N), k))).reshape(-1, k)
    n_splits = len(test_groups)
    n_paths = n_splits * k // n_groups 

    t_span = len(X.index)
    group_num = np.arange(t_span) // (t_span // N )
    group_num[group_num == N] = N -1
    #group_num[group_num == N] = f'G{N-1}' 

    group_strs = ['G'+str(n) for n in range(N)]
    mapping = dict(zip(np.unique(group_num), group_strs))


    # Map the numeric index to string labels
    string_index = [mapping[value] for value in group_num]

    testSets = np.zeros((N, n_splits))
    splitPredictions = pd.DataFrame({'groupIndex':string_index},index=X.index)

    is_test_group = np.full((N, n_splits), fill_value=False)
    for col, pair in enumerate(test_groups):
            i, j = pair
            is_test_group[[i, j], col] = True

    # map test sets to paths
    path_folds = np.full((N, n_paths), fill_value=np.nan)

    for i in range(n_paths):
            for j in range(N):
                s_idx = is_test_group[j, :].argmax().astype(int)
                path_folds[j, i] = s_idx
                is_test_group[j, s_idx] = False

    #sm = SimpleModel(X, y)
    #X, y, tbEvents = sm.createBenchmarkModel()
    # Generate the training/testing visualizations for each CV split
    print((X.index == y.index).sum() == (len(y)))
    for ii, (tr, tt) in enumerate(cv.split(X, y, pred_times=prediction_times, eval_times=evaluation_times)):
    #for ii, (tr, tt) in enumerate(cv.split(X.reindex(evaluation_times.index), y.reindex(evaluation_times.index), prediction_times.reindex(evaluation_times.index), eval_times)):

        # print('fold', ii, '\n')
        # print(tr, '\n')
        # print(tt, '\n')
       # splitPredictions['S'+str(ii)] = model.predict(X.iloc[tt, :])
        X_train = X.iloc[tr].dropna()
        y_train = y.iloc[tr].dropna()
        X_test = X.iloc[tt].dropna()
        tbEvents = tripleBarrierEvents.loc[X_test.index]

        splitIndex = n_splits - ii - 1


        #splitLabels = sm.makePredictions(X_train, y_train, X_test, classifierModel)
        splitLabels = sm.makePredictionsBetSizing(X_train, y_train, X_test, classifierModel,tbEvents,stepSize,numClasses )
        splitPredictions.loc[splitLabels.index, 'S' + str(splitIndex)] = splitLabels

    splitPredictions = splitPredictions[['groupIndex'] + [f'S{i}' for i in range(n_splits)]]
        
    return splitPredictions, path_folds

    # add groupIndex as multiindex
    # index by groupnumber, splitnumber


def runCSCVBacktestingFixed(cv, sm, classifierModel,  X, y,  n_groups, 
                            k, paths, prediction_times,evaluation_times, tripleBarrierEvents, 
                            stepSize=0.05, numClasses=10, lw=5):
    # generate the combinations
    # n groups
    #N = n_groups + 1
    N = n_groups
    
    test_groups = np.array(list(itt.combinations(np.arange(N), k))).reshape(-1, k)
    n_splits = len(test_groups)
    n_paths = n_splits * k // n_groups 

    t_span = len(X.index)
    group_num = np.arange(t_span) // (t_span // N )
    group_num[group_num == N] = N -1
    #group_num[group_num == N] = f'G{N-1}' 

    group_strs = ['G'+str(n) for n in range(N)]
    mapping = dict(zip(np.unique(group_num), group_strs))


    # Map the numeric index to string labels
    string_index = [mapping[value] for value in group_num]

    testSets = np.zeros((N, n_splits))
    splitPredictions = pd.DataFrame({'groupIndex':string_index},index=X.index)

    is_test_group = np.full((N, n_splits), fill_value=False)
    for col, pair in enumerate(test_groups):
            i, j = pair
            is_test_group[[i, j], col] = True

    # map test sets to paths
    path_folds = np.full((N, n_paths), fill_value=np.nan)

    for i in range(n_paths):
            for j in range(N):
                s_idx = is_test_group[j, :].argmax().astype(int)
                path_folds[j, i] = s_idx
                is_test_group[j, s_idx] = False

    #sm = SimpleModel(X, y)
    #X, y, tbEvents = sm.createBenchmarkModel()
    # Generate the training/testing visualizations for each CV split
    print((X.index == y.index).sum() == (len(y)))
    for ii, (tr, tt) in enumerate(cv.split(X, y, pred_times=prediction_times, eval_times=evaluation_times)):
    #for ii, (tr, tt) in enumerate(cv.split(X.reindex(evaluation_times.index), y.reindex(evaluation_times.index), prediction_times.reindex(evaluation_times.index), eval_times)):

        # print('fold', ii, '\n')
        # print(tr, '\n')
        # print(tt, '\n')
       # splitPredictions['S'+str(ii)] = model.predict(X.iloc[tt, :])
        X_train = X.iloc[tr].dropna()
        y_train = y.iloc[tr].dropna()
        X_test = X.iloc[tt].dropna()
        tbEvents = tripleBarrierEvents.loc[X_test.index]

        splitIndex = n_splits - ii - 1


        #splitLabels = sm.makePredictions(X_train, y_train, X_test, classifierModel)
        #splitLabels = sm.makePredictionsBetSizing(X_train, y_train, X_test, classifierModel,tbEvents,stepSize,numClasses )
        classifierModel.fit(X_train, y_train)
        splitLabels = pd.Series(classifierModel.predict(X_test), index=X_test.index)
        splitProb = np.amax(classifierModel.predict_proba(X_test),axis=1)
        #stepSize = 0.05
        #numClasses = 10
        foldEvents = tripleBarrierEvents.loc[X_test.index]

        betSizes = betSizing.getSignalNoSide(foldEvents,stepSize,splitProb,splitLabels,numClasses,numThreads=1)
        #splitPredictions.loc[splitLabels.index, 'S' + str(splitIndex)] = splitLabels
        splitPredictions.loc[splitLabels.index, 'S' + str(splitIndex)] = betSizes

    splitPredictions = splitPredictions[['groupIndex'] + [f'S{i}' for i in range(n_splits)]]
        
    return splitPredictions, path_folds

def generatePathFolds(t_span, n, k, prediction_times, evaluation_times, verbose=True):
    # split data into N groups, with N << T
    # this will assign each index position to a group position
    group_num = np.arange(t_span) // (t_span // n)
    group_num[group_num == n] = n-1
    
    # generate the combinations 
    test_groups = np.array(list(itt.combinations(np.arange(n), k))).reshape(-1, k)
    C_nk = len(test_groups)
    n_paths = C_nk * k // n 
    
    
    if verbose:
        print('n_sim:', C_nk)
        print('n_paths:', n_paths)
    
    # is_test is a T x C(n, k) array where each column is a logical array 
    # indicating which observation in in the test set
    is_test_group = np.full((n, C_nk), fill_value=False)
    is_test = np.full((t_span, C_nk), fill_value=False)
    
    # assign test folds for each of the C(n, k) simulations
    for k, pair in enumerate(test_groups):
        i, j = pair
        is_test_group[[i, j], k] = True
        
        # assigning the test folds
        mask = (group_num == i) | (group_num == j)
        is_test[mask, k] = True
        
    # for each path, connect the folds from different simulations to form a backtest path
    # the fold coordinates are: the fold number, and the simulation index e.g. simulation 0, fold 0 etc
    path_folds = np.full((n, n_paths), fill_value=np.nan)
    
    for i in range(n_paths):
        for j in range(n):
            s_idx = is_test_group[j, :].argmax().astype(int)
            path_folds[j, i] = s_idx
            is_test_group[j, s_idx] = False
            #cv.split(X, y, pred_times=prediction_times, eval_times=evaluation_times)
    
    # finally, for each path we indicate which simulation we're building the path from and the time indices
    paths = np.full((t_span, n_paths), fill_value= np.nan)
    
    for p in range(n_paths):
        for i in range(n):
            mask = (group_num == i)
            paths[mask, p] = int(path_folds[i, p])
    # paths = paths_# .astype(int)

    return (is_test, paths, path_folds) 