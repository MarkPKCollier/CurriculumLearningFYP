import bisect
import numpy as np

class Exp3S:
    def __init__(self, num_tasks, eta, beta, eps):
        self.N = num_tasks
        self.w = np.zeros(num_tasks)
        self.eta = eta
        self.beta = beta
        self.eps = eps
        self.rewards = []
        self.t = 1
        self.w = np.zeros(self.N)
        self.max_rewards = 50000

    def draw_task(self):
        self.pi = (1 - self.eps) * self._softmax(self.w) + (self.eps / self.N)
        self.k = np.random.choice(self.N, p=self.pi)
        return self.k

    def update_w(self, v, t):
        '''v is learning progress, t is time to process batch that provided v'''
        r_ = v/t
        self._reservoir_sample(r_)
        q_lo, q_hi = self._quantiles()
        r = self._r(q_lo, q_hi, r_)

        alpha_t = 1/float(self.t)
        r_b_t = np.asarray([((r if i == self.k else 0) + self.beta)/self.pi[i] for i in range(self.N)])
        tmp = np.exp(self.w + self.eta * r_b_t)
        for i in range(self.N):
            s = 0
            for j in range(self.N):
                if i != j:
                    s += tmp[j]
            self.w[i] = np.log((1 - alpha_t) * tmp[i] + (alpha_t/(self.N - 1)) * s)

        self.t += 1

    def _quantiles(self):
        dist = max(self.rewards) - min(self.rewards)
        q_lo_pos = int(0.2 * len(self.rewards))
        q_hi_pos = int(0.8 * len(self.rewards)) - 1
        return self.rewards[q_lo_pos], self.rewards[q_hi_pos]

    def _reservoir_sample(self, r_):
        insert = False
        if len(self.rewards) >= self.max_rewards and np.random.random_sample() < 10.0/float(self.t):
            pos = np.random.randint(0, high=len(self.rewards))
            del a[pos]
            insert = True
        if insert or len(self.rewards) < self.max_rewards:
            pos = bisect.bisect_left(self.rewards, r_)
            self.rewards.insert(pos, r_)

    def _r(self, q_lo, q_hi, r_):
        if r_ < q_lo:
            return -1.0
        elif r_ >= q_hi:
            return 1.0
        else:
            return (2.0*(r_ - q_lo)/float(q_hi - q_lo)) - 1.0

    def _softmax(self, w):
        e_w = np.exp(w)
        return e_w / np.sum(e_w)

