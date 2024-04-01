import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA', FutureWarning)


class TracePrediction:
    def __init__(self,
                 num_future_intervals,
                 method='baseline',
                 predict_config={
                     'p': 1, # model complexity, ARIMA
                     'd': 1, # order of differencing, ARIMA
                     'q': 0, # averaging, ARIMA
                     'damped_trend': True, # set True to apply damped trend, Exponential Smoothing
                     'trend': 'add', # 'add' or 'mul', Exponential Smoothing
                     'baseline_penalty': 0.1, # baseline
                 },
                 upper_bound=32,
                 lower_bound=8,
                 ):
        self.num_future_intervals = num_future_intervals
        self.predict_config = predict_config
        self.method = method
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.in_grow_bound = 3
        self.out_grow_bound = 3
        self.in_penalty = 1.0
        self.out_penalty = 1.0
        assert self.method in ["baseline", "ARIMA", "ES"] # ES stands for Exponential Smoothing

    def predict_next_intervals(self, past_history, add_history=None, remove_history=None, verbose=False):
        in_trace = out_trace = None
        past_history = self.remove_outlier_all(past_history, size=3) #FIX 1: remove outliers
        past_history = self.handle_turnpoint(past_history) #FIX 3: flatten contents before turning point
        if self.check_sorting(past_history, ascend=True):
            delta = past_history[-1] - min(past_history)
            if 5 <= delta < 10:
                self.in_penalty = 0.75
            elif delta >= 10:
                self.in_penalty = 0.5
        if self.check_sorting(past_history, ascend=False):
            delta = max(past_history) - past_history[-1]
            if 5 <= delta < 10:
                self.out_penalty = 0.75
            elif delta >= 10:
                self.out_penalty = 0.5

        if self.method == 'ARIMA':
            if np.std(past_history) == 0:
                estimated_trace = [round(past_history[-1]) for _ in range(self.num_future_intervals)]
            else:
                model = ARIMA(past_history,
                              order=(self.predict_config['p'],
                                     self.predict_config['d'],
                                     self.predict_config['q']))
                model_fit = model.fit(disp=5 if verbose else -1)
                estimated_trace = np.rint(model_fit.forecast(self.num_future_intervals)[0])
            in_prediction = np.zeros(self.num_future_intervals)
            out_prediction = np.zeros(self.num_future_intervals)
            # control range of change per prediction
            first_elem = estimated_trace[0]
            for k in range(1, self.num_future_intervals):
                if estimated_trace[k] > self.in_grow_bound + first_elem:
                    estimated_trace[k] = self.in_grow_bound + first_elem
                if estimated_trace[k] < first_elem - self.out_grow_bound:
                    estimated_trace[k] = first_elem - self.out_grow_bound
            # fix prediction curves by upper and lower bounds
            for k in range(self.num_future_intervals):
                if estimated_trace[k] > self.upper_bound:
                    estimated_trace[k] = self.upper_bound
                if estimated_trace[k] < self.lower_bound:
                    estimated_trace[k] = self.lower_bound
            pred_std = np.std(estimated_trace)
            if remove_history and pred_std == 0 and np.std(past_history) != 0:
                out_std, out_var, out_mean = np.std(remove_history), np.var(remove_history), np.mean(remove_history)
                opt_remove_map = np.random.poisson(min(out_std, out_var, out_mean), self.num_future_intervals)
                out_prediction += opt_remove_map

            new_estimated_trace = np.zeros(self.num_future_intervals)
            curr_N = past_history[-1]
            # in_prediction, out_prediction and estimated_trace must match the original formula
            for k in range(self.num_future_intervals):
                if estimated_trace[k] > curr_N:
                    in_prediction[k] += estimated_trace[k] - curr_N
                elif estimated_trace[k] < curr_N:
                    out_prediction[k] += curr_N - estimated_trace[k]
                curr_N = estimated_trace[k]
            # update estimated_trace with results of additional prediction rules
            curr_N = past_history[-1]
            for k in range(self.num_future_intervals):
                new_estimated_trace[k] = round(curr_N + in_prediction[k] * self.in_penalty - out_prediction[k] * self.out_penalty)
                curr_N = new_estimated_trace[k]
            # align in and out trace given updated estimated trace
            new_in_prediction = np.zeros(self.num_future_intervals)
            new_out_prediction = np.zeros(self.num_future_intervals)
            curr_N = past_history[-1]
            for k in range(self.num_future_intervals):
                if new_estimated_trace[k] > curr_N:
                    new_in_prediction[k] += new_estimated_trace[k] - curr_N
                elif new_estimated_trace[k] < curr_N:
                    new_out_prediction[k] += curr_N - new_estimated_trace[k]
                curr_N = new_estimated_trace[k]
            estimated_trace = [int(x) for x in new_estimated_trace] #TODO: change this line to pass unit test!
            in_trace = [int(x) for x in new_in_prediction] #TODO: logic not implemented yet. So far it is all zero
            out_trace = [int(x) for x in new_out_prediction] #TODO: will implement add_history logic for augmenting in prediction curves later.
            #TODO: expect more rules added to optimize naive ARIMA results

        elif self.method == 'ES':
            model = ExponentialSmoothing(past_history,
                                         damped_trend=self.predict_config['damped_trend'],
                                         trend=self.predict_config['trend'])
            model_fit = model.fit()
            estimated_trace = [round(x) for x in model_fit.forecast(self.num_future_intervals)]
        else: # baseline predicts future window (ideally, 3-5 minutes) of same size as past_history, using linear drifting from most recent value
            last_value = past_history[-1]
            past_length = len(past_history)
            addition_times = sum([past_history[i] - past_history[i - 1] > 0 for i in range(1, past_length)])
            failure_times = sum([past_history[i] - past_history[i - 1] < 0 for i in range(1, past_length)])
            change_rate = (addition_times - failure_times) / past_length
            estimated_trace = [round(last_value + (j + 1) * change_rate * self.predict_config['baseline_penalty']) \
                    for j in range(self.num_future_intervals)]
            out_trace = [0] + [int(estimated_trace[i] < estimated_trace[i - 1]) for i in range(1, len(estimated_trace))]
        return {
            'N': estimated_trace,
            'in': in_trace,
            'out': out_trace,
        }
    
    def check_sorting(self, array, ascend=True):
        i = 0
        while i < len(array) - 1:
            if ascend and array[i] > array[i + 1]:
                return False
            if ascend == False and array[i] < array[i + 1]:
                return False
            i += 1 
        return True

    def remove_outlier(self, array, size=1):
        assert len(array) > 2
        for i in range(len(array) - 1 - size):
            if array[i] == array[i + 1 + size] and all([array[i] != array[i + j] for j in range(1, size + 1)]):
                for j in range(1, size + 1):
                    array[i + j] = array[i]
        return array

    def remove_outlier_all(self, array, size=1):
        for x in range(size, 0, -1):
            array = self.remove_outlier(array, size=x)
        return array

    def handle_turnpoint(self, array, ascend=True):
        l = array[0]
        r = array[-1]
        if np.min(array) < min(l, r):
            target = np.min(array)
            pos = np.where(array == target)[0][-1]
            for i in range(pos):
                array[i] = target
        elif np.max(array) > max(l, r):
            target = np.max(array)
            pos = np.where(array == target)[0][-1]
            for i in range(pos):
                array[i] = target
        return array
