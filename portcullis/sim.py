
from portcullis.agent import Agent
from portcullis.env import Env
import random


class Sim():
    DEFAULT_RNG_SEED = 2170596287

    def __init__(self, agent: Agent):
        self.agent = agent

    def run(self, num_episodes: int, mem_samples: int) -> float:
        self.agent.load()
        avg_score = 0.0
        hiscore = 0.0
        env: Env = self.agent.env
        for i in range(num_episodes):
            done = False
            state = env.reset()
            score = 0.0
            while not done:
                action = self.agent.act(state)
                next_state, reward, done = env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                score += reward
            if score > hiscore:
                hiscore = score
                self.agent.save()
            self.agent.learn(mem_samples=mem_samples)
            avg_score += score
            eps = self.agent.adj_eps()
            print('#', i + 1, 'Score:', score, 'Eps', eps)
        return avg_score / float(num_episodes)

    @staticmethod
    def seed_rng(seed: int = DEFAULT_RNG_SEED):
        random.seed(seed)
