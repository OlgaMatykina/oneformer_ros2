import rclpy
import time
import numpy as np


class InferenceSpeedMeter:

    def __init__(self, rate=1):
        self.rate = rate

        self.T = None

        self.N = None
        self.T_sum = 0
        self.T2_sum = 0
        self.T_min = np.inf
        self.T_max = 0

    def start(self):
        self.T = time.perf_counter()

    def stop(self):
        T = time.perf_counter() - self.T
        if self.N is None:
            self.N = 0
        else:
            self.N += 1
            self.T_sum += T
            self.T2_sum += T**2
            self.T_min = min(T, self.T_min)
            self.T_max = max(T, self.T_max)

            if self.N % self.rate == 0:
                avg = self.T_sum / self.N
                std = np.sqrt(self.T2_sum / self.N - avg**2)
                rclpy.logging.get_logger('semseg_inference_speed').debug(
                    'Inference time: CUR={}s AVG={}s STD={}s MIN={}s MAX={}s'.format(T, avg, std, self.T_min, self.T_max))
                # print('Inference time: CUR={}s AVG={}s STD={}s MIN={}s MAX={}s'.format(T, avg, std, self.T_min, self.T_max))
