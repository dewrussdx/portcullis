from portcullis.agent import Agent
from portcullis.env import Env
import matplotlib.pyplot as plt


class Sim():
    RNG_SEED = 2170596287

    def __init__(self, agent: Agent):
        self.agent = agent

    def plot(self):
        plt.show()

    def run(self, num_episodes: int = 1_000, mem_samples: int = 64, training: bool = True, seed: int = RNG_SEED) -> float:
        avg_score = 0.0
        hi_score = 0.0
        env: Env = self.agent.env
        state, _ = env.reset(seed=seed)
        self.agent.load()
        for i in range(num_episodes):
            done = False
            score = 0.0
            samples = 0
            state, _ = env.reset()
            while not done:
                self.agent.env.render()
                action = self.agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                if training:
                    self.agent.remember(
                        state, action, reward, next_state, done)
                    if training and True:
                        self.agent.learn(mem_samples)
                        self.agent.adj_eps()
                    samples += 1
                state = next_state
                score += reward
            if training and False:
                self.agent.learn(mem_samples, soft_update=False)
                self.agent.adj_eps()
            if score > hi_score:
                hi_score = score
                print('New high-score:', hi_score)
                if training and True:
                    self.agent.save()
            avg_score += score
            print('#', i + 1, '[TRAIN]' if training else '[EVAL]', 'Score:', score, 'Eps:',
                  self.agent.eps, 'Samples:', samples, 'Mem:', self.agent.mem.size())
        env.close()
        return avg_score / float(num_episodes)
