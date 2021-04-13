from ms_mode_seeking import *
from Utilities.ex2_utils import *
from Utilities.ex3_utils import *
from utils.tracker import Tracker

class MeanShiftTracker(Tracker):

    def name(self):
        return 'MeanShift_tracker'

    def initialize(self, image, region):
        self.eps = 0.5
        self.bins = 16
        self.sigma = 2
        self.thresh = 0.03
        self.enlarge_factor = 1

        self.window = max(region[2], region[3]) * self.enlarge_factor
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        kernel = create_epanechnik_kernel(region[2], region[3], self.sigma)
        patch = get_patch(image, self.position, np.shape(kernel))[0]
        self.q_histogram = extract_histogram(patch=patch, nbins=self.bins, weights=kernel)

    def track(self, image):

        patch = get_patch(image, self.position, self.size)[0]
        p_histogram = extract_histogram(patch=patch, nbins=self.bins)

        # Replace Zero values with near zero values epsilon
        self.q_histogram[self.q_histogram == 0] = sys.float_info.epsilon
        v = np.sqrt(np.divide(self.q_histogram, (p_histogram + self.eps)))
        wi = backproject_histogram(patch=patch, histogram=v, nbins=self.bins)
        wi_center = (wi.shape[0]//2, wi.shape[1]//2)

        new_cords = MS_modeSeeking(img=wi, center=wi_center, k_sz=self.size, n_iters=1, thresh=self.thresh)

        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        return left+(new_cords[0]-wi_center[0]), top+(new_cords[1]-wi_center[1]), self.size[0], self.size[1]


