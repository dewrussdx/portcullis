from portcullis.agent import Agent
from portcullis.env import Env
import matplotlib.pyplot as plt
import random
from time import time
import sys
import torch
import numpy as np
from portcullis.agent import TD3, DDPGplus, DDPG
from portcullis.mem import ReplayBuffer


class Sim:
    def __init__(self, agent: Agent):
        self.agent = agent

    def plot(self) -> None:
        plt.show()

    def run(self, args) -> None:
        total_time = 0.0
        env: Env = self.agent.env
        print('Seeding simulation:', args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

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

    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment

    def eval_policy(policy, env_name, seed, eval_episodes=10):
        eval_env = gym.make(env_name)
        eval_env.seed(seed + 100)

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, _ = eval_env.reset()
            done = False
            while not done:
                action = policy.select_action(np.array(state))
                state, reward, done, trunc, _ = eval_env.step(action)
                done = done or trunc
                avg_reward += reward

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    def run_continuous(self, env, args):
        # Set seeds
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": args.gamma,
            "tau": args.tau,
        }

        # Initialize policy
        policy = None
        if args.algo == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = args.policy_noise * max_action
            kwargs["noise_clip"] = args.noise_clip * max_action
            kwargs["policy_freq"] = args.policy_freq
            policy = TD3(**kwargs)
        elif args.algo == "DDPGplus":
            policy = DDPGplus(**kwargs)
        elif args.algo == "DDPG":
            policy = DDPG(**kwargs)
        assert policy is not None

        if args.load:
            policy.load(args.path)

        replay_buffer = ReplayBuffer(state_dim, action_dim)

        # Evaluate untrained policy
        # evaluations = [self.eval_policy(policy, args.env, args.seed)]

        state, _ = env.reset(seed=args.seed)
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(int(args.max_timesteps)):

            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action *
                                       args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done, trunc, _ = env.step(action)
            done_bool = float(
                done or trunc) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, _ = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % args.eval_freq == 0:
                # evaluations.append(self.eval_policy(policy, args.env, args.seed))
                # np.save(f"./results/{file_name}", evaluations)
                if args.save:
                    policy.save(args.path)
