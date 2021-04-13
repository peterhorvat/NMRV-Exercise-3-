from Utilities.ex2_utils import *
from Utilities.ex3_utils import *
from utils.tracker import Tracker
import time
class MOSSETracker(Tracker):

    def name(self):
        return 'MOSSE'

    def initialize(self, image, region):

        # Parameters
        self.alpha = 0.09
        self.lamda = 0.00001
        self.sigma = 3.3  # 0.5 - 5

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (int(region[2]) - 1 if region[2] % 2 == 0 else int(region[2]),
                     int(region[3]) - 1 if region[3] % 2 == 0 else int(region[3]))

        self.G = np.fft.fft2(create_gauss_peak(self.size, self.sigma))
        self.F = np.fft.fft2(cv2.cvtColor(get_patch(image, self.position, self.G.shape[::-1])[0], cv2.COLOR_BGR2GRAY))
        self.H = np.divide(np.multiply(self.G, np.conjugate(self.F)),
                           np.multiply(self.F, np.conjugate(self.F)) + self.lamda)
        self.cos_window = create_cosine_window((self.F.shape[1], self.F.shape[0]))

    def track(self, image):
        G = np.fft.fft2(create_gauss_peak(self.size, self.sigma))
        F = np.fft.fft2(cv2.cvtColor(get_patch(image, self.position, G.shape[::-1])[0], cv2.COLOR_BGR2GRAY))
        F = np.multiply(F, self.cos_window)
        Ht = np.divide(np.multiply(G, np.conjugate(F)),
                       np.multiply(F, np.conjugate(F)) + self.lamda)
        R = np.fft.ifft2(np.multiply(Ht, F))
        self.H = (1 - self.alpha) * self.H + self.alpha * Ht

        x, y = np.unravel_index(R.argmax(), R.shape[::-1])
        if x > self.size[0] / 2:
            x = x - self.size[0]
        if y > self.size[1] / 2:
            y = y - self.size[1]
        new_x = self.position[0] + x - self.size[0] / 2
        new_y = self.position[1] + y - self.size[1] / 2
        return new_x, new_y, self.size[0], self.size[1]


