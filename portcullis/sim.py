from portcullis.agent import Agent
from portcullis.env import Env
import matplotlib.pyplot as plt
import random
from time import time
import sys
import torch
import numpy as np
from portcullis.agent import TD3, LAP_DDQN
from portcullis.replay import ReplayBuffer, PriotizedReplayBuffer
import copy


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
                f'Eps: {self.agent.eps:.4f}',
                f'Mem: {samples} / {self.agent.mem.size()}',
                f'Time: {elapsed:.2f} / {total_time:.2f}',
            )
        env.close()

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
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3(**kwargs)

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

    def run_discrete_lap(self, env, args):
        # Set seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # Initialize and load policy
        parameters = {
            # LAP/PAL
            "alpha": 0.4,
            "min_priority": 1,
            # Exploration
            "start_timesteps": 1e4,
            "initial_eps": 0.1,
            "end_eps": 0.1,
            "eps_decay_period": 25e4,
            # Evaluation
            "eval_freq": 1e3,
            "eval_eps": 0,
            # Learning
            "discount": 0.99,
            "buffer_size": 1e6,
            "batch_size": 64,
            "optimizer": "Adam",
            "optimizer_parameters": {
                "lr": 3e-4
            },
            "train_freq": 1,
            "polyak_target_update": True,
            "target_update_freq": 1,
            "tau": 0.005
        }
       

        is_gym, env_type, num_actions, state_dim, _ = Env.get_env_spec(env)
        assert env_type == Env.DISCRETE

        kwargs = {
            "num_actions": num_actions,
            "state_dim": state_dim,
            "discount": parameters["discount"],
            "optimizer": parameters["optimizer"],
            "optimizer_parameters": parameters["optimizer_parameters"],
            "polyak_target_update": parameters["polyak_target_update"],
            "target_update_frequency": parameters["target_update_freq"],
            "tau": parameters["tau"],
            "initial_eps": parameters["initial_eps"],
            "end_eps": parameters["end_eps"],
            "eps_decay_period": parameters["eps_decay_period"],
            "eval_eps": parameters["eval_eps"],
            "alpha": parameters["alpha"],
            "min_priority": parameters["min_priority"],
        }

        policy = LAP_DDQN(**kwargs)

        prioritized = True
        replay_buffer = PriotizedReplayBuffer(
            state_dim,
            prioritized,
            parameters["batch_size"],
            parameters["buffer_size"],
        )

        state, _ = env.reset(seed=args.seed)
        done = False
        episode_start = True
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        # Interact with the environment for max_timesteps
        for t in range(int(args.max_timesteps)):

            episode_timesteps += 1

            # if args.train_behavioral:
            if t < parameters["start_timesteps"]:
                action = env.action_space.sample() if is_gym else env.random_action()
            else:
                action = policy.select_action(np.array(state))

            # Perform action and log results
            next_state, reward, done, trunc, _ = env.step(action)
            episode_reward += reward

            # Only consider "done" or "trunc" if episode terminates due to failure condition
            done_float = float(
                done or trunc) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state,
                              reward, done_float)  # add (done, episode_start)
            state = copy.copy(next_state)
            episode_start = False

            # Train agent after collecting sufficient data
            if t >= parameters["start_timesteps"] and (t + 1) % parameters["train_freq"] == 0:
                policy.train(replay_buffer)

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, _ = env.reset()
                done = False
                episode_start = True
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % parameters["eval_freq"] == 0:
                print('EPSILON =>',policy.eps)
       