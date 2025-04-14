## Second Attempt
## This is a trained agent that uses a regression model to evaluate the state of the game.
## This agent uses a negamax algorithm with alpha-beta pruning to find the best action to take.

from utils import State, load_data, Action
import numpy as np
import time
import random

class StudentAgent:
    def __init__(self, searchDepth: int = 4):
        """Instantiates your agent.
        """
        self.searchDepth = searchDepth
        self.timeLimit = 2.9
        self.startTime = None
        self.transpositionTable = {}

        ## From training (hardcoded for now)
        self.weights = np.array([
            [
                -0.08775905,  0.08653392, -0.02941646,  0.07780639,  0.0155079,
                 0.35724166,  0.01775325,  0.20830321, -0.21002494, -0.01946871,
                 0.01950825, -0.00667738,  0.0089836,   0.08506323,  0.08718471,
                 0.07453567,  0.0868069,   0.07439226,  0.08699849,  0.06961332,
                 0.07739924,  0.07496097,  0.00623519, -0.00333887,  0.05383141,
                -0.25388166,  0.06787859,  0.000479,   -0.02139454
            ]
        ], dtype=np.float32)

        self.bias = np.array([0.14248353], dtype=np.float32)


    def negamax(self, state: State, depth: int, alpha: float, beta: float, color: int) -> float:

        ## Timeout check
        if time.time() - self.startTime >= self.timeLimit:
            raise TimeoutError("Time limit exceeded in negamax")

        ## Check cached first
        stateHash = self.getStateHash(state)
        if (stateHash, depth) in self.transpositionTable:
            return self.transpositionTable[(stateHash, depth)]

        ## Terminal or depth limit
        if depth == 0 or state.is_terminal():
            ## if game is over use terminal_utility but if not evaluate using my regression model
            if state.is_terminal():
                value = state.terminal_utility()
            else:
                value = self.evaluateState(state)

            value *= color
            self.transpositionTable[(stateHash, depth)] = value
            return value

        max_value = -float('inf')
        valid_actions = state.get_all_valid_actions()

        for action in valid_actions:
            next_state = state.change_state(action)

            score = -self.negamax(next_state, depth - 1, -beta, -alpha, -color)
            max_value = max(max_value, score)
            alpha = max(alpha, score)

            if alpha >= beta:
                break

        self.transpositionTable[(stateHash, depth)] = max_value

        return max_value


    def getStateHash(self, state: State) -> tuple:
        board_flat = tuple(state.board.flatten())
        meta_flat = tuple(state.local_board_status.flatten())
        return (board_flat, meta_flat, state.fill_num)

    def evaluateState(self, state: State) -> float:
        """
        Evaluates the state using the formula
        """
        features = extract_features(state)
        value = np.dot(self.weights, features) + self.bias
        return value.item()


    # def minimax(self, state: State, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
    #     """
    #     Minimax algorithm to find the best action to take.
    #     """

    #     if depth == 0 or state.is_terminal():
    #         return StudentAgent.evaluateGame(state, self.player)

    #     if maximizing: ## Maximizing player
    #         v = float('-inf')
    #         for action in state.get_all_valid_actions():
    #             v = max(v, self.minimax(state.change_state(action), depth - 1, alpha, beta, False))
    #             alpha = max(alpha, v)
    #             if v >= beta:
    #                 return v
    #         return v
    #     else: ## Minimizing player
    #         v = float('inf')
    #         for action in state.get_all_valid_actions():
    #             v = min(v, self.minimax(state.change_state(action), depth - 1, alpha, beta, True))
    #             beta = min(beta, v)
    #             if v <= alpha:
    #                 return v
    #         return v

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        self.startTime = time.time()
        self.transpositionTable.clear()
        bestAction = state.get_random_valid_action()
        validActions = state.get_all_valid_actions()
        topLevelColor = +1 if state.fill_num == 1 else -1

        depth = 1
        while depth <= self.searchDepth:

            if time.time() - self.startTime >= self.timeLimit:
                break  ## Safety check before starting new depth

            # self.transpositionTable.clear()
            bestValue = -float('inf')
            currentBestAction = None

            try:
                for action in validActions:

                    if time.time() - self.startTime >= self.timeLimit:
                        raise TimeoutError("Time limit exceeded in choose_action inner loop")

                    nextState = state.change_state(action)
                    value = self.negamax(nextState, depth - 1, -float('inf'), float('inf'), topLevelColor)

                    if value > bestValue:
                        bestValue = value
                        currentBestAction = action


                if currentBestAction is not None:
                    bestAction = currentBestAction

                depth += 1

            except TimeoutError:
                break

        return bestAction



###### Feature Extraction Functions ######

def extract_features(state: State) -> np.ndarray:
    """
    Extract features from the state for training.

    Total Feature list:
    1. Player token count
    2. Opponent token count
    3. Player local wins
    4. Opponent local wins
    5. Draw local board count
    6. Difference in local wins
    7. Unclaimed boards
    8. Meta board center control to player
    9. Meta board center control to opponent
    10. Meta line potential for player
    11. Meta line potential for opponent
    12. Center board control in local board to player
    13. Center board control in local board to opponent
    14. One-hot encoding of forced board
    23. Count total local threats for player
    24. Count total local threats for opponent
    25. Difference in local threats
    26. Player turn
    27. Whether the last move was forced
    28. Number of valid moves
    29. Board fill ratio
    """

    features = []

    ## Constant definitions
    board = state.board
    localBoardStatus = state.local_board_status
    player = state.fill_num
    opponent = 3 - player
    lastAction = state.prev_local_action

    ## Token Counts
    playerCount = np.sum(board == 1)
    opponentCount = np.sum(board == 2)
    features.append(playerCount)
    features.append(opponentCount)

    ## p1, p2 local wins and draw count
    p1LocalWins = np.sum(localBoardStatus == 1)
    p2LocalWins = np.sum(localBoardStatus == 2)
    drawLbsCount = np.sum(localBoardStatus == 3)
    features.extend([p1LocalWins, p2LocalWins, drawLbsCount])

    ## Difference in local wins
    diffLocalWins = p1LocalWins - p2LocalWins
    features.append(diffLocalWins)

    ## Unclaimed boards
    emptyBoards = np.sum(localBoardStatus == 0)
    features.append(emptyBoards)

    ## meta board center control
    features.append(1 if localBoardStatus[1][1] == 1 else 0)
    features.append(1 if localBoardStatus[1][1] == 2 else 0)

    ## Meta Line potential
    metaThreatsP1 = metaLinePotential(localBoardStatus, 1)
    metaThreatsP2 = metaLinePotential(localBoardStatus, 2)
    features.extend([metaThreatsP1, metaThreatsP2])

    ## Center board control in local board
    centerCountP1 = countCenterCells(board, localBoardStatus, 1)
    centerCountP2 = countCenterCells(board, localBoardStatus, 2)
    features.extend([centerCountP1, centerCountP2])


    ## One-hot encoding of forced board if there is
    forced_board_9d = forcedBoardOneHot(lastAction)
    features.extend(forced_board_9d)

    ## Count total local threats across all open local boards
    totalLocalThreatsP1 = countTotalLocalThreats(board, localBoardStatus, 1)
    totalLocalThreatsP2 = countTotalLocalThreats(board, localBoardStatus, 2)
    features.extend([totalLocalThreatsP1, totalLocalThreatsP2])

    # Difference in local threats
    diffTotalLocalThreats = totalLocalThreatsP1 - totalLocalThreatsP2
    features.append(diffTotalLocalThreats)

    ## Player turn
    features.append(player)

    ## whether the last move was forced
    freedomMove = hasFreedomMove(localBoardStatus, lastAction)
    features.append(freedomMove)

    ## Number of valid moves
    valid_moves = state.get_all_valid_actions()
    features.append(len(valid_moves))

    ## Board fill ratio
    boardFillRatio = np.count_nonzero(board) / 81.0
    features.append(boardFillRatio)

    return np.array(features, dtype=np.float32)


##### Helper functions #####

def countLine(line: np.ndarray, player: int) -> int:
    """
    Counts the number of player tokens in a line (row, column, or diagonal).
    """
    return int(np.count_nonzero(line == player) == 2 and np.count_nonzero(line == 0) == 1)

def countTwoInARow(board: np.ndarray, player: int) -> int:
    """
    Checks for how many two in a row pattern each player has on still-active local boards
    Account for diagonals too
    """
    counter = 0
    for i in range(3):
        counter += countLine(board[i, :], player)
        counter += countLine(board[:, i], player)

    firstDiagonal = np.zeros(3, dtype=int)
    secDiagonal = np.zeros(3, dtype=int)
    for i in range(3):
        firstDiagonal[i] = board[i, i]
        secDiagonal[i] = board[i, 2 - i]

    counter += countLine(firstDiagonal, player)
    counter += countLine(secDiagonal, player)

    return counter

def metaLinePotential(meta: np.ndarray, player: int) -> int:
    counter = 0

    ## Check rows and columns
    for i in range(3):
        row = meta[i, :]
        col = meta[:, i]
        if np.count_nonzero(row == player) == 1 and np.count_nonzero(row == 0) == 2:
            counter += 1
        if np.count_nonzero(col == player) == 1 and np.count_nonzero(col == 0) == 2:
            counter += 1

    ## Check diagonals
    firstDiagonal = [0, 0, 0]
    secDiagonal = [0, 0, 0]
    for i in range(3):
        firstDiagonal[i] = meta[i, i]
        secDiagonal[i] = meta[i, 2 - i]

    if firstDiagonal.count(player) == 1 and firstDiagonal.count(0) == 2:
        counter += 1
    if secDiagonal.count(player) == 1 and secDiagonal.count(0) == 2:
        counter += 1

    return counter

def emptyCellCount(board: np.ndarray) -> int:
    return np.count_nonzero(board == 0)


def hasFreedomMove(localStatus: np.ndarray, lastMove: tuple) -> int:
    if lastMove is None:
        return 1  # first move
    lr, lc = lastMove
    return int(localStatus[lr][lc] != 0)

def forcedBoardOneHot(lastAction: tuple) -> list:
    """
    Returns a one-hot vector of length 9 indicating which local board
    the next move is forced to (if not closed).
    If lastAction is None (start of game), returns all zeros.
    """
    arr = [0] * 9
    if lastAction is not None:
        r, c = lastAction
        index = r * 3 + c
        arr[index] = 1
    return arr

def countCenterCells(board: np.ndarray, localBoardStatus: np.ndarray, player: int) -> int:
    """
    Counts how many open local boards have the center cell belonging to 'player_id'.
    """
    counter = 0
    for i in range(3):
        for j in range(3):
            if localBoardStatus[i][j] == 0:  # board must be open
                localBoardCenter = board[i][j][1, 1]
                if localBoardCenter == player:
                    counter += 1
    return counter


def countTotalLocalThreats(board: np.ndarray, localBoardStatus: np.ndarray, player: int) -> int:
    """
    Sums the two-in-a-row threats for 'player_id' across all open local boards.
    Uses 'countTwoInARow' on each open local board
    """
    totalThreats = 0
    for i in range(3):
        for j in range(3):
            if localBoardStatus[i][j] == 0:  # open local board
                localBoard = board[i][j]
                totalThreats += countTwoInARow(localBoard, player)
    return totalThreats


# state = State(
#     board=np.array([
#         [
#             [[1, 0, 2], [0, 1, 0], [0, 0, 1]],
#             [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
#             [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
#         ],
#         [
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#             [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#         ],
#         [
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
#             [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
#         ],
#     ]),
#     fill_num=1,
#     prev_action=(2, 2, 0, 1),
# )
# start_time = time.time()
# student_agent = StudentAgent()
# constructor_time = time.time()
# action = student_agent.choose_action(state)
# end_time = time.time()
# assert state.is_valid_action(action)
# print(f"Constructor time: {constructor_time - start_time}")
# print(f"Action time: {end_time - constructor_time}")
# assert constructor_time - start_time < 1
# assert end_time - constructor_time < 3

state = State(
    board=np.array([
        [
            [[1, 0, 2], [0, 1, 0], [0, 0, 1]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        ],
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
    ]),
    fill_num=1,
    prev_action=(2, 2, 0, 0)
)
start_time = time.time()
student_agent = StudentAgent()
constructor_time = time.time()
action = student_agent.choose_action(state)
end_time = time.time()
assert state.is_valid_action(action)
print(f"Constructor time: {constructor_time - start_time}")
print(f"Action time: {end_time - constructor_time}")
assert constructor_time - start_time < 1
assert end_time - constructor_time < 3