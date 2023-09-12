#import tkinter as tk
#from tkinter import messagebox
import numpy as np
from portcullis.env import Env, State, Action


class TicTacToe(Env):
    X = 0
    O = 1

    def __init__(self, interactive: bool = False, num_episodes: int = 100):
        super().__init__(name='TicTacToe', env_type=Env.DISCRETE,
                         action_space=np.arange(9), observation_space=np.zeros(pow(3, 9), dtype=np.int32))
        self.window = tk.Tk()
        self.window.title('TicTacToe Environment')
        self.board = self.observation_space
        self.player = 0
        self.interactive = interactive
        self.num_episodes = num_episodes
        self.create_gui()

    def reset(self, seed: int = None) -> tuple[State, any]:
        self.reset_game()
        return super().reset(seed)

    def step(self, action: Action) -> tuple[State, float, bool, bool, any]:
        """Perform a action in this environment and compute the reward, for the MDP. This function 
        returns the next state, the reward, if simulation is done or truncated. Note that to stay compatible
        with the gym environments we also return 'info' as last parameter and with None value in the tuple,
        like so: next_state, reward, done, truncated, info = env.step(action)
        """
        reward = 0.
        done = False
        # Player-1 move according to action
        if self.try_move(action):
            val = self.check_for_winner(self.get_reward)
            if type(val) == int:
                reward = val
            else:
                assert val == False
        else:
            reward = -1  # Invalid move

        # Make a random move for player-2
        if not done:
            self.random_move()
            val = self.check_for_winner(self.get_reward)
            if type(val) == int:
                reward = -val
            else:
                assert val == False

        next_state = self.board
        done = self.index >= self.max_steps
        return next_state, reward, done, False, None

    def run(self) -> None:
        if self.interactive:
            self.window.mainloop(self.num_episodes)
        else:
            for _ in range(self.num_episodes):
                while self.ai_move():
                    pass

    def onclick(self, index: int) -> None:
        if self.interactive and self.board[index] == 0:
            self.move(index)
            self.check_for_winner()

    def create_gui(self) -> None:
        for i in range(3):
            for j in range(3):
                button = tk.Button(self.window, text="", font=(
                    "Arial", 50), height=2, width=6, bg="lightblue",
                    command=lambda index=i*3 + j: self.onclick(index) if self.interactive else None)
                button.grid(row=i, column=j, sticky="nsew")

    def update_gui(self, index: int) -> None:
        button = self.window.grid_slaves(
            row=int(index/3), column=int(index % 3))[0]
        button.config(text=['X', 'O'][self.player])
        button.update()

    def human_winner(self, token: int) -> bool:
        if token == 0:
            message = 'Cat\'s game!'
        else:
            message = f"Player {token} wins!"

        answer = messagebox.askyesno(
            "Game Over", message + " Do you want to restart the game?")

        if answer:
            self.reset_game()
        else:
            self.window.quit()
        return True

    def ai_winner(self, token: int) -> bool:
        if token == 0:
            message = 'Cat\'s game!'
        else:
            message = f"Player {token} wins!"
        print(message)
        self.reset_game()
        return True

    def get_reward(self, token: int) -> int:
        if token == 0:
            print('Cat\'s Game')
            return 0.
        print('Player', token, 'won the game')
        return 1.

    def check_for_winner(self, fn=None) -> bool | int:
        finalize = fn or (
            self.human_winner if self.interactive else self.ai_winner)
        token = self.board[0]
        if token != 0:
            if self.board[1] == token and self.board[2] == token:
                return finalize(token)
            elif self.board[3] == token and self.board[6] == token:
                return finalize(token)
            elif self.board[4] == token and self.board[8] == token:
                return finalize(token)
        token = self.board[1]
        if token != 0:
            if self.board[4] == token and self.board[7] == token:
                return finalize(token)
        token = self.board[2]
        if token != 0:
            if self.board[4] == token and self.board[6] == token:
                return finalize(token)
        token = self.board[3]
        if token != 0:
            if self.board[4] == token and self.board[5] == token:
                return finalize(token)
        if np.count_nonzero(self.board != 0) == 9:
            return finalize(0)
        return False

    def reset_gui(self) -> None:
        for i in range(3):
            for j in range(3):
                button = self.window.grid_slaves(row=i, column=j)[0]
                button.config(text='')

    def available_moves(self) -> np.ndarray[int]:
        indices = np.flatnonzero(self.board == 0)
        return self.action_space[indices]

    def random_move(self) -> None:
        indices = np.flatnonzero(self.board == 0)
        size = len(indices)
        assert size > 0
        pick = np.random.choice(size)
        self.move(indices[pick])

    def ai_move(self) -> None:
        self.random_move()
        return not self.check_for_winner()

    def try_move(self, index: int) -> bool:
        if self.board[index] == 0:
            self.move(index)
            return True
        return False

    def move(self, index: int) -> None:
        assert self.board[index] == 0
        self.board[index] = self.player + 1
        self.update_gui(index)
        self.player ^= 1

    def reset_game(self) -> None:
        self.board.fill(0)
        self.player = 0
        if self.interactive:
            self.reset_gui()


def main():
    ttt = TicTacToe(interactive=False)
    ttt.run()


if __name__ == "__main__":
    main()
