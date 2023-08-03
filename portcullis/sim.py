from portcullis.agent import Agent
from portcullis.env import Env
import matplotlib.pyplot as plt
from time import time


class Sim:
    RNG_SEED = 2170596287

    def __init__(self, agent: Agent):
        self.agent = agent

    def plot(self):
        plt.show()

    def run(
        self,
        num_episodes: int = 1_000,
        mem_samples: int = 64,
        train_after_load: bool = True,
        seed: int = RNG_SEED,
        checkpoint_timer=5 * 60,
    ):
        env: Env = self.agent.env
        state, _ = env.reset(seed=seed)
        self.agent.load(training=train_after_load)
        for i in range(num_episodes):
            score = 0.0
            samples = 0
            done = False
            state, _ = env.reset()
            timer = time()
            while not done:
                samples += 1
                self.agent.env.render()
                action = self.agent.act(state)
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                if self.agent.training:
                    self.agent.remember(
                        state, action, reward, next_state, done)
                    self.agent.learn(mem_samples)
                    self.agent.adj_eps()
                    if (time() - timer) >= checkpoint_timer:
                        print('Episode timed out after',
                              checkpoint_timer, 'seconds')
                        done = True
                state = next_state
                score += reward
            if self.agent.training:
                if self.agent.is_highscore(score):
                    self.agent.save()
            print(
                "#",
                i + 1,
                "[TRAIN]" if self.agent.training else "[EVAL]",
                "Score:",
                score,
                "Eps:",
                self.agent.eps,
                "Samples:",
                samples,
                "Mem:",
                self.agent.mem.size(),
            )
        env.close()
