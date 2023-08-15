from portcullis.agent import Agent
from portcullis.env import Env
import matplotlib.pyplot as plt
import random
from time import time
import torch
import numpy as np
from portcullis.agent import TD3
from portcullis.replay import ReplayBuffer


class Sim:
    def __init__(self, agent: Agent):
        self.agent = agent

    def plot(self) -> None:
        plt.plot(y=self.agent.scores, title=self.agent.name)
        plt.show()

    def run(self, args) -> None:
        env: Env = self.agent.env
        is_gym, env_type, _, _, _ = Env.get_env_spec(env)
        assert env_type == Env.DISCRETE

        print('Seeding simulation:', args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        state, _ = env.reset(seed=args.seed)

        # Set and load agent mode
        training = args.mode.lower() == 'train'
        self.agent.set_mode(training)
        if args.load:
            self.agent.load(path=args.path)
        self.agent.set_mode(training)

        # Prefill replay buffer
        if training:
            print('Filling up replay buffer before training starts...')
            while self.agent.mem.size < args.mem_prefill:
                action = env.action_space.sample() if is_gym else env.random_action()
                next_state, reward, done, trunc, _ = env.step(action)
                done = done or trunc
                self.agent.mem.add(state, action, next_state, reward, done)
                if done:
                    state, _ = env.reset()

        # Set simulation to initial state
        print('Simulation is starting...')
        state, _ = env.reset()
        score = 0.
        episode_ticks = 0
        episode_count = 0
        total_time = 0.
        timer = time()

        while episode_count < args.num_episodes:
            episode_ticks += 1

            # Select action
            action = self.agent.act(state)

            # Perform action
            next_state, reward, done, trunc, _ = env.step(action)
            # If you want to continue training even when env solved...
            # done = done or (trunc if episode_ticks <
            #                env._max_episode_steps else False)
            done = done or trunc

            # Store data in replay buffer
            if training:
                self.agent.remember(state, action, next_state, reward, done)

            state = next_state
            score += reward

            if training:
                self.agent.train()

            if done:
                # checkpoint
                episode_count += 1
                if self.agent.is_highscore(score):
                    if args.save:
                        self.agent.save()

                elapsed = time() - timer
                total_time += elapsed
                print(
                    f'#{episode_count}:',
                    f'[TRAIN:{args.algo}]' if self.agent.training else f'[EVAL:{args.algo}]',
                    f'Score:{score:.2f}/{self.agent.high_score:.2f}',
                    f'Eps:{self.agent.eps:.2f}',
                    f'Mem:{self.agent.mem.usage():.2f}%',
                    f'Time:{elapsed:.2f}/{total_time:.2f}',
                )
                # Reset environment
                timer = time()
                state, _ = env.reset()
                score = 0.
                episode_ticks = 0
        env.close()
        print('Simulation has ended.')

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
