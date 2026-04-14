import random
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class TicTacToeEnv(gym.Env):
    def __init__(self):
        super().__init__() #konstrukor gym.Env
        self.board = [' ' for _ in range(9)]
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low = -1, high=1, shape=(9,), dtype=np.int32) #shape to wektor, ktory ma 9 wartości

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = [' ' for _ in range(9)]
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(9, dtype=np.int32) # tablica o dlugosci 9 wypełniona zerami
        for i, tile in enumerate(self.board):
            if tile == 'X':
                obs[i] = 1
            elif tile == 'O':
                obs[i] = -1
        return obs

    def step(self, action):
        """
                Wykonuje pojedynczą turę w grze Kółko i Krzyżyk.

                Args:
                    action (int): Indeks pola (0-8), na którym AI chce postawić znak 'X'.

                Returns:
                    tuple zawierający 5 elementów:
                    - obs (np.ndarray): Matematyczny stan planszy po ruchu (1 dla X, -1 dla O, 0 dla pustych).
                    - reward (float): Nagroda dla AI (10 za wygraną, -10 za błąd/przegraną, 0 za grę dalej).
                    - terminated (bool): True, jeśli gra się zakończyła (wygrana/remis/błąd).
                    - truncated (bool): Zawsze False (nie używamy limitu czasu w tej grze).
                    - info (dict): Dodatkowe informacje diagnostyczne, np. {"msg": "Remis"}.
                """

        #ruch ai
        if self.board[action] != ' ':
            return self._get_obs(), -10, True, False, {"msg": "Nielegalny ruch"}

        self.board[action] = 'X'

        if self.checkWinner('X'):
            return self._get_obs(), 10, True, False, {"msg": "AI Wygrało!"}

        if self.isDraw():
            return self._get_obs(),0,True, False, {"msg": "Remis!"}

        print("\nPlansza po ruchu AI:")
        self.displayBoard()

        #ruch bota(do trenigu)
        #self.makeMove(random.choice(self.availableMoves()), "O")

        # ruch człowieka
        while True:
            inpucik = int(input("podaj numer (1-9)")) - 1
            if inpucik in self.availableMoves():
                self.makeMove(inpucik, "O")
                break



        if self.checkWinner("O"):
            return self._get_obs(),-10,True, False, {"msg": "Ai przegralo"}

        if self.isDraw():
            return self._get_obs(),0,True, False, {"msg": "Remis!"}

        return self._get_obs(),0,False,False,{"msg": "Gramy dalej"}


    def availableMoves(self):
        availableTiles = []
        for i in range(len(self.board)):
            if self.board[i] == ' ':
                availableTiles.append(i)
        return availableTiles

    def displayBoard(self):
        for i in range(3):
            row = self.board[i * 3: (i + 1) * 3] #slice start:end
            print(f"| {row[0]} | {row[1]} | {row[2]} |")
            if i < 2:
                print("-" * 13)

    def makeMove(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            return True
        else:
            return False


    def isDraw(self):
        if not self.availableMoves() and self.checkWinner('O') == self.checkWinner('X') == False:
            return True
        else:
            return False

    def checkWinner(self, letter):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Wiersze
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Kolumny
            [0, 4, 8], [2, 4, 6]  # Przekątne
        ]
        for combo in winning_combinations:
            if self.board[combo[0]] == letter and self.board[combo[1]] == letter and self.board[combo[2]] == letter:
                return True
        return False

def training():
    env = TicTacToeEnv()
    # opcjonalny check czy wszytsko jest git
    check_env(env, warn=True)
    print("Środowisko przeszło testy! Zaczynamy trening...")

    # 3. Inicjalizujemy PPO(Proximal Policy Optimization)
    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=100000)

    model.save("tic_tac_toe_ai")
    print("Gotowe! Mózg AI zapisany jako tic_tac_toe_ai.zip")

def game():
    print("Wczytuję mózg AI...")
    model = PPO.load("tic_tac_toe_ai")

    # game init
    env = TicTacToeEnv()
    obs, info = env.reset()
    env.displayBoard()

    done = False

    print("\n--- ROZPOCZYNAMY GRĘ! AI to X, Ty to O ---")

    #  Pętla gry
    while not done:
        # Metoda predict zawsze zwraca dwie rzeczy: akcję oraz stan ukryty (u nas nieużywany, oznaczamy go jako '_').
        action, _ = model.predict(obs)

        print(f"\nAI wybiera pole: {action}")

        # wynik z ai
        obs, reward, terminated, truncated, info = env.step(action)
        env.displayBoard()

        if terminated:
            print(info["msg"])
            break

game()