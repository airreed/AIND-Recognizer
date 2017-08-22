import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
import sys


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        max_model = None
        min_BIC= sys.maxsize
        # TODO implement model selection based on BIC scores
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            hmm_model = self.base_model(num_states)
            if hmm_model == None:
                continue
            try:
                logL = hmm_model.score(self.X, self.lengths)
                d = hmm_model.n_features
                p = num_states ** 2 + 2 * num_states * d - 1
                logN = math.log(len(self.X))
                BIC = -2 * logL + p * logN
                # the smaller the BIC, the better the model
                if BIC < min_BIC:
                    min_BIC = BIC
                    max_model = hmm_model
            except:
                if self.verbose:
                    print("No BIC score!")
        if max_model == None:
            return self.base_model(self.n_constant)
        return max_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        max_model = None
        max_DIC= -sys.maxsize - 1
        # TODO implement model selection based on BIC scores
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            hmm_model = self.base_model(num_states)
            if hmm_model == None:
                continue
            try:
                logL = hmm_model.score(self.X, self.lengths)
                # get average log likelihood for other words
                logL_average_other = self.OtherWordsAverageScore(hmm_model)
                if logL_average_other == None:
                    continue
                DIC = logL - logL_average_other
                # the bigger the DIC, the better the model
                if DIC > max_DIC:
                    max_DIC = DIC
                    max_model = hmm_model
            except:
                if self.verbose:
                    print("No DIC score!")
        if max_model == None:
            return self.base_model(self.n_constant)
        return max_model

    def OtherWordsAverageScore(self, model):
        logL_sum = 0
        n = 0
        for word, (X, lengths) in self.hwords.items():
            if word == self.this_word:
                continue
            try:
                logL = model.score(X, lengths)
                logL_sum += logL
                n += 1
            except:
                if self.verbose:
                    print("No Log Likelihood for ", word)
        if n != 0:
            logL_average = logL_sum / n
            return logL_average
        return None

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        max_model = None
        max_logL = -sys.maxsize - 1
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            if len(self.lengths) < 3:
                hmm_model = self.base_model(num_states)
                try:
                    logL = hmm_model.score(self.X, self.lengths)
                    if logL > max_logL:
                        max_logL = logL
                        max_model = hmm_model                    
                except:
                    if self.verbose:
                        print("No log likelihood score!")
                continue
            split_method = KFold()
            logL_sum = 0
            n = 0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                try:
                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                    logL = hmm_model.score(X_test, lengths_test)
                    logL_sum += logL
                    n += 1
                    if self.verbose:
                        print("model created for {} with {} states".format(self.this_word, num_states))
                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, num_states))
            if n != 0:
                logL_average = logL_sum / n
                if logL_average > max_logL:
                    max_logL = logL_average
                    max_model = hmm_model
        if max_model == None:
            return self.base_model(self.n_constant)
        return max_model



