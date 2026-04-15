import numpy as np
import gymnasium as gym
import random
import pickle



class TicTacToeEnv(gym.Env):
    def __init__(self):
        super().__init__() #konstrukor gym.Env
        self.board = [' ' for _ in range(9)]
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low = -1, high=1, shape=(9,), dtype=np.int32) #shape to wektor, ktory ma 9 wartości
        self.isGameActive = True

    def reset(self, seed=None, options=None):
        self.isGameActive = True
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
            self.isGameActive = False
            return self._get_obs(), 10, True, False, {"msg": "AI Wygrało!"}

        if self.isDraw():
            self.isGameActive = False
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
            self.isGameActive = False
            return self._get_obs(),-10,True, False, {"msg": "Ai przegralo"}

        if self.isDraw():
            self.isGameActive = False
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

class QBot:
    def __init__(self, epsilon=0.01):
        self.q_table = {}
        self.letter = 'X'
        self.epsilon = epsilon
        self.alpha = 0.1
        self.gamma = 0.9

    def get_state_key(self, board):
            """
            Takes a list (np. ['X', ' ', 'O', ...])
            and returns a string
            """
            return ''.join(board)

    def choose_action(self, board, available_moves):
        key = self.get_state_key(board)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(9)

        if random.random() < self.epsilon:
            return random.choice(available_moves)
        else:
            max = -1000
            grades = self.q_table[key]
            best_move =  -1
            for i in available_moves:
                if grades[i] > max:
                    max = grades[i]
                    best_move = i

            return best_move

    def learn(self, old_board, action, reward, new_board, done):
        old_key = self.get_state_key(old_board)
        new_key =self.get_state_key(new_board)

        if old_key not in self.q_table:
            self.q_table[old_key] = np.zeros(9)
        if new_key not in self.q_table:
            self.q_table[new_key] = np.zeros(9)

        old_q = self.q_table[old_key][action]
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[new_key])

        self.q_table[old_key][action] = old_q + self.alpha* (reward + self.gamma * max_future_q - old_q) # Bellmans Equation: https://www.deltami.edu.pl/media/articles/2008/04/delta-2008-04-rownanie-bellmana.pdf
        print(f"Update: {old_key} action {action} -> reward: {reward}")

    def save(self, filename="bot_brain.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Mózg bota zapisany do {filename}!")

    def load(self, filename="bot_brain.pkl"):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print("Mózg bota wczytany!")
        except FileNotFoundError:
            print("Nie znaleziono zapisanego mózgu, zaczynam od zera.")

def test():
    bot = QBot()
    bot.epsilon = 0.0  # Ustawiamy na 0, żeby wymusić logiczne myślenie (zawsze "else")
    testowa_plansza = ['X', ' ', 'O', ' ', ' ', ' ', ' ', ' ', ' ']
    dostepne_ruchy = [1, 3, 4, 5, 6, 7, 8]

    # Wstrzykujemy botowi wiedzę - ruch na środek (4) daje miliard punktów!
    key = bot.get_state_key(testowa_plansza)
    bot.q_table[key] = np.array([0, 0, 0, 0, 1000000, 0, 0, 0, 0])

    wybrany_ruch = bot.choose_action(testowa_plansza, dostepne_ruchy)

    print(f"Bot wybrał ruch nr: {wybrany_ruch}")
    # Jeśli Twój kod zadziała, konsola wypluje 4.

#test()

def play():
    env = TicTacToeEnv()
    bot = QBot()

    game_count = 100

    bot.load()

    for _ in range(game_count):
        env.reset()
        print("\n--- NOWA GRA ---")
        while env.isGameActive:
            state_before = list(env.board)
            action = bot.choose_action(state_before, env.availableMoves())
            _, reward, done, _, info = env.step(action)
            env.displayBoard()
            state_new = list(env.board)
            bot.learn(state_before, action, reward, state_new, done)
            if done:
                print(info["msg"])
                bot.save()
                break
play()
