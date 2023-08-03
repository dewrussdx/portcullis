from portcullis.agent import Agent
from portcullis.env import Env
import matplotlib.pyplot as plt
from time import time
import sys


class Sim:
    def __init__(self, agent: Agent):
        self.agent = agent

    def plot(self) -> None:
        plt.show()

    def run(self, args) -> None:
        total_time = 0.0
        env: Env = self.agent.env
        print('Seeding simulation:', args.seed)
        state, _ = env.reset(seed=args.seed)
        training = args.mode.lower() == 'train'
        self.agent.set_mode(training)
        if args.load:
            self.agent.load(path=args.path)
        self.agent.set_mode(training)
        for i in range(args.num_episodes):
            score = 0.0
            samples = 0
            done = False
            state, _ = env.reset()
            timer = time()
            try:
                while not done:
                    samples += 1
                    self.agent.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, trunc, _ = env.step(action)
                    if not args.ignore_trunc:
                        done = done or trunc
                    if self.agent.training:
                        self.agent.remember(
                            state, action, reward, next_state, done)
                        self.agent.train(args.batch_size)
                        self.agent.dec_eps()
                    state = next_state
                    score += reward
            except KeyboardInterrupt:
                if args.interactive:
                    while True:
                        print('Simulation interrupted.')
                        print('(1) Continue')
                        print('(2) Exit')
                        choice = input()
                        if choice == '1':
                            break
                        elif choice == '2':
                            sys.exit(0)
            if self.agent.training:
                if self.agent.is_highscore(score):
                    if args.save:
                        self.agent.save(path=args.path)
            elapsed = time() - timer
            total_time += elapsed
            # Status line
            print(
                f'#{i+1}:',
                f'[TRAIN:{args.algo}]' if self.agent.training else f'[EVAL:{args.algo}]',
                f'Score: {score:.2f} / {self.agent.high_score:.2f}',
                f'Eps: {self.agent.eps:.2f}',
                f'Mem: {samples} / {self.agent.mem.size()}',
                f'Time: {elapsed:.2f} / {total_time:.2f}',
            )
        env.close()
