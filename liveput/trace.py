import csv
import bisect
import math

from trace_prediction import TracePrediction
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA', FutureWarning)

class SpotTrace:
    def __init__(self, trace_file_path, start_hour=0, end_hour=None,
                 interval_length=60, num_future_intervals=7, predictor='ARIMA',
                 pred_gap=1):
        self.trace_file_path = trace_file_path
        self.start_hour = start_hour
        self.end_hour = end_hour
        # FIXME(replay): when we use segment, can we see the real past intervals at the beginning
        self.see_real_past_at_beginning = True #False

        self.timestamps = []
        # history of node info of each interval, each item is a list of [#add, #remove, #available]
        self.node_history = []

        self.available_nodes = []
        self.add_history = []
        self.remove_history = []

        self.N_0 = 0
        self.pred_gap = pred_gap

        # seconds of an interval
        self.interval_length = interval_length

        # number of future intervals to look ahead
        self.num_future_intervals = num_future_intervals
        # number of past intervals to use for prediction (recommended)
        if self.num_future_intervals == 14:
            self.num_past_intervals = 17
        elif self.num_future_intervals == 12:
            self.num_past_intervals = 12
        elif self.num_future_intervals == 10:
            self.num_past_intervals = 10
        elif self.num_future_intervals == 9:
            self.num_past_intervals = 9
        elif self.num_future_intervals == 8:
            self.num_past_intervals = 8
        elif self.num_future_intervals == 7:
            self.num_past_intervals = 8
        elif self.num_future_intervals == 5:
            self.num_past_intervals = 10
        else:
            if self.num_future_intervals > 15:
                print("Warning: Be careful about using larger past intervals, as prediction model may be unresponsive to intense change")
            self.num_past_intervals = self.num_future_intervals
            assert self.num_past_intervals >= 7 and self.num_future_intervals >= 5, "Too few data points for time-series prediction"
        self.predictor = TracePrediction(self.num_future_intervals, method=predictor)

        # read and pre-process trace
        self.read_trace()
        self.pad_trace()

        self.total_intervals = len(self.node_history)
        self.start_interval_id = math.floor(self.start_hour * 3600 / self.interval_length)
        if self.end_hour is None:
            self.end_interval_id = self.total_intervals
        else:
            self.end_interval_id = math.ceil(self.end_hour * 3600 / self.interval_length)

    @property
    def num_intervals(self):
        """Number of intervals in the trace.
        """
        return min(self.total_intervals, self.end_interval_id - self.start_interval_id)

    @property
    def interval_duration(self):
        """The duration of each interval in seconds.
        """
        return self.interval_length

    def get_node_number(self, cur_interval_id):
        """Return the number of available nodes at the given interval.
        """
        cur_interval_id = cur_interval_id + self.start_interval_id
        if not self.see_real_past_at_beginning and cur_interval_id < self.start_interval_id:
            return self.N_0
        if cur_interval_id < 0:
            return self.N_0
        return self.node_history[cur_interval_id][2]

    def get_prev_intervals(self, current_interval_id, num_intervals=1, timestamps=False):
        """
        Find the previous interval up until now.
        """
        current_interval_id = current_interval_id + self.start_interval_id
        if self.see_real_past_at_beginning:
            start_idx = max(0, current_interval_id - num_intervals + 1)
        else:
            start_idx = max(self.start_interval_id, current_interval_id - num_intervals + 1)
        end_idx = current_interval_id + 1

        prev_intervals = []
        for i, info in enumerate(self.node_history[start_idx:end_idx]):
            if not self.see_real_past_at_beginning and start_idx == self.start_interval_id and i == 0:
                prev_intervals.append([info[2], info[2], 0])
            else:
                prev_intervals.append([info[2], info[0], info[1]])

        if timestamps:
            return prev_intervals, self.timestamps[start_idx:end_idx]
        return prev_intervals

    def get_next_intervals(self, start_interval_id, num_intervals=1, predict=False, timestamps=False):
        """Get a sequence of future intervals.
        Args:
            start_interval_id (int): the id of the interval to start from.
            num_intervals (int): the number of intervals to get.
            predict (bool): whether to use prediction.
        Returns:
            A list of intervals, each interval is a list of [#available, #add, #remove].
            The interval id of the first interval is `start_interval_id`.
        """
        future_intervals = []
        if predict:
            return self.predict_next_multi_intervals(start_interval_id - 1)

        start_interval_id = start_interval_id + self.start_interval_id

        end_interval_id = min(start_interval_id + num_intervals, self.end_interval_id)
        future_node_history = self.node_history[start_interval_id:end_interval_id]

        for i, info in enumerate(future_node_history):
            if not self.see_real_past_at_beginning and start_interval_id == self.start_interval_id and i == 0:
                future_intervals.append([info[2], info[2], 0])
            else:
                future_intervals.append([info[2], info[0], info[1]])

        if timestamps:
            return future_intervals, self.timestamps[start_interval_id:end_interval_id]
        return future_intervals

    def read_trace(self):
        total_nodes = 0
        with open(self.trace_file_path, "r") as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                t, event = int(row[0]) // 1000, str(row[1])
                if len(self.timestamps) > 0 and t == self.timestamps[-1]:
                    info = self.node_history[-1]
                    if event == "add":
                        info[0] += 1
                        total_nodes += 1
                        info[2] = total_nodes
                    else:
                        info[1] += 1
                        total_nodes -= 1
                        info[2] = total_nodes
                    self.node_history[-1] = info
                else:
                    self.timestamps.append(t)
                    info = [0] * 3
                    if event == "add":
                        info[0] += 1
                        total_nodes += 1
                        info[2] = total_nodes
                    else:
                        info[1] += 1
                        total_nodes -= 1
                        info[2] = total_nodes
                    self.node_history.append(info)
            assert len(self.timestamps) == len(self.node_history)
            self.add_history = [item[0] for item in self.node_history]
            self.remove_history = [item[1] for item in self.node_history]
            self.available_nodes = [item[2] for item in self.node_history]

    def pad_trace(self):
        '''
        Pad traces with timestamps and node counts by interval length.
        '''
        new_timestamps = []
        new_node_history = []

        trace_length = len(self.timestamps)
        for x in range(trace_length):
            new_timestamps.append(self.timestamps[x])
            new_node_history.append(self.node_history[x])
            while x + 1 < trace_length and new_timestamps[-1] + self.interval_length < self.timestamps[x + 1]:
                new_timestamps.append(new_timestamps[-1] + self.interval_length)
                # padded items should maintain the same available nodes as last documented in the trace
                info_item = [0, 0, self.node_history[x][-1]]
                new_node_history.append(info_item)
        self.node_history = new_node_history
        self.available_nodes = [item[2] for item in new_node_history]
        self.add_history = [item[0] for item in new_node_history]
        self.remove_history = [item[1] for item in new_node_history]
        self.timestamps = new_timestamps

    def get_next_multi_intervals_by_time(self, t_stamp, intervals=None):
        '''
        Given current time stamp, find the next interval of that contains intervals elements.
        t_stamp: current time stamp
        '''
        if intervals is None:
            intervals = self.num_future_intervals
        start_idx = bisect.bisect_left(self.timestamps, t_stamp)
        assert start_idx < len(self.timestamps)
        # if current timestamp is reached, look for the next id as starting point of next interval
        if self.timestamps[start_idx] <= t_stamp:
            start_idx += 1
        end_idx = start_idx + intervals
        return self.node_history[start_idx : end_idx], self.timestamps[start_idx : end_idx]

    def get_prev_multi_intervals_by_time(self, t_stamp, intervals=None):
        '''
        Find the previous interval up until now.

        t_stamp: current time stamp
        '''
        if intervals is None:
            intervals = self.num_past_intervals
        end_idx = bisect.bisect_left(self.timestamps, t_stamp)
        start_idx, end_idx = max(0, end_idx - intervals + 1), end_idx + 1
        return self.node_history[start_idx : end_idx], self.timestamps[start_idx : end_idx]

    def predict_next_multi_intervals(self, cur_interval_id, evaluate=False):
        """Predict the next intervals.
        Predict next `self.num_future_intervals` intervals.
        Returns a sequence of predicted future intervals.
        """

        prev_intervals = self.get_prev_intervals(cur_interval_id, num_intervals=self.num_past_intervals)

        node_hist = [interval[0] for interval in prev_intervals]
        add_hist = [interval[1] for interval in prev_intervals]
        remove_hist = [interval[2] for interval in prev_intervals]
        pred_next_intervals = []
        if cur_interval_id + self.start_interval_id < self.num_past_intervals or (not self.see_real_past_at_beginning and cur_interval_id < self.num_past_intervals):
            if cur_interval_id == -1:
                pred_next_intervals = []
                init_nnodes = self.available_nodes[self.start_interval_id]
                for i in range(self.num_future_intervals):
                    if i == 0:
                        pred_next_intervals.append([init_nnodes, init_nnodes - self.N_0, 0])
                    else:
                        pred_next_intervals.append([init_nnodes, 0, 0])
            else:
                pred_next_intervals = [[node_hist[-1], 0, 0] for _ in range(self.num_future_intervals)]
        else:
            pred_trace = self.predictor.predict_next_intervals(
                node_hist, add_history=add_hist, remove_history=remove_hist
            )

            for i in range(self.num_future_intervals):
                interval = [pred_trace['N'][i], pred_trace['in'][i], pred_trace['out'][i]]
                pred_next_intervals.append(interval)

        if evaluate:
            # compare predictions with ground truths
            next_intervals = self.get_next_intervals(cur_interval_id + 1, num_intervals=self.num_future_intervals)
            L1_res = self.get_avg_L1(pred_next_intervals, next_intervals)
            return pred_next_intervals, L1_res

        return pred_next_intervals

    def get_avg_L1(self, pred_intervals, true_intervals):
        L1_n, L1_a, L1_r = 0, 0, 0
        for pred_interval, true_interval in zip(pred_intervals, true_intervals):
            L1_n += abs(true_interval[0] - pred_interval[0])
            L1_a += abs(true_interval[1] - pred_interval[1])
            L1_r += abs(true_interval[2] - pred_interval[2])

        return {
            'N': L1_n / float(self.num_future_intervals),
            'in': L1_a / float(self.num_future_intervals),
            'out': L1_r / float(self.num_future_intervals),
        }

    def prediction_unit_test(self, look_ahead):
        diffs = dict()
        i = 0
        gaps = 10
        ft_gap = 4
        notice_period = 2 # minutes
        while i < self.num_intervals:
            future_intervals_truth = self.get_next_intervals(i, num_intervals=look_ahead, predict=False)
            future_intervals = self.get_next_intervals(i, num_intervals=look_ahead, predict=True)
            prev_N = self.get_node_number(i - 1)
            for fid, interval in enumerate(future_intervals):
                nnodes, n_in, n_out = interval
                assert nnodes >= 0
                assert n_in >= 0
                assert n_out >= 0

                assert n_out <= prev_N
                prev_N = prev_N + n_in - n_out
                assert prev_N == nnodes
            prediction = [interval[0] for interval in future_intervals]
            ground_truth = [interval[0] for interval in future_intervals_truth]
            predicted_diff = []
            if len(ground_truth) < len(prediction):
                break
            assert len(prediction) == len(ground_truth)
            for j in range(len(ground_truth)):
                predicted_diff.append(abs(prediction[j] - ground_truth[j]))
            diffs[i] = sum(predicted_diff)

            i += gaps

        distributions = [0] * 5
        avg_pred_diff = 0.0
        for score in diffs.values():
            avg_pred_diff += score / self.num_future_intervals
            if score <= look_ahead:
                distributions[0] += 1
            elif look_ahead < score <= 2 * look_ahead:
                distributions[1] += 1
            elif 2 * look_ahead < score <= 3 * look_ahead:
                distributions[2] += 1
            elif 3 * look_ahead < score <= 4 * look_ahead:
                distributions[3] += 1
            else:
                distributions[4] += 1
        print(f"Avg. Pred. Diff. (look_ahead={look_ahead}) / {len(diffs)} counts: {avg_pred_diff / len(diffs)}")
        print(f"Dist. of Pred. Diff. (look_ahead={look_ahead}): {distributions}")

if __name__ == '__main__':
    look_ahead = 8
    t = SpotTrace('traces/p3trace.csv', num_future_intervals=look_ahead)
    t.prediction_unit_test(look_ahead=look_ahead)

    # look_ahead = 12
    # t = SpotTrace('traces/p3trace_12preemption.csv', num_future_intervals=look_ahead)
    # t.prediction_unit_test(look_ahead=look_ahead)

    # t = SpotTrace('traces/p3trace.csv', start_hour=5.5, end_hour=6.5, num_future_intervals=look_ahead)
    # t.prediction_unit_test(look_ahead=look_ahead)
