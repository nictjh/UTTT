### This code is responsible for pitting two agents against each other in a game of UTTT ###

from utils import State, load_data, Action
import numpy as np
import time
import random
from typing import Optional, Tuple

class StudentAgent:
    def __init__(self, searchDepth: int = 6):
        """Instantiates your agent.
        """
        self.searchDepth = searchDepth
        self.timeLimit = 2.99
        self.startTime = None
        self.transpositionTable = {}

        self.weights = np.array([
            [
                7.3400602e-02, -3.6237743e-02,  6.7538403e-02, -1.4367065e-01,
                -4.7295257e-02,  2.2409678e-01, -4.2857643e-02,  2.1471971e-01,
                -2.1295819e-01, -6.4664148e-02,  3.4142572e-02, -1.8101247e-02,
                2.2210512e-02,  3.2006705e-03,  2.7680721e-03, -2.8073471e-03,
                6.8107322e-03, -4.2146156e-03,  5.7563246e-03, -8.7335669e-03,
                2.0816782e-03,  8.7817700e-04, -4.7714268e-03,  1.7378660e-03,
                1.2291840e-01,  2.4189222e-01, -1.8258268e-02,  3.1877883e-02,
                -2.3297241e-02, -1.2025372e-02,  3.7034450e-03, -8.3784765e-04,
                7.0671607e-03,  5.6169205e-03, -9.8372167e-03, -6.3413982e-03,
                8.3924402e-03, -1.4578652e-02, -1.8276494e-04,  5.3997287e-03,
                2.7512924e-03, -2.2009462e-02, -1.0033873e+00, -2.4738135e-02,
                9.7053275e-02, -1.0414126e-01
            ]
        ], dtype=np.float32)

        self.bias = np.array([0.04403995], dtype=np.float32)



    def negamax(self, state: 'State', depth: int, alpha: float = 0.0, beta: float = 1.0) -> Tuple[float, Optional['Action']]:
        """
        Negamax-like search in [0..1] range.
        Returns (value, bestAction) from the perspective of the current player.
        """

        ## Timeout check
        if time.time() - self.startTime >= self.timeLimit:
            raise TimeoutError("Time limit exceeded in negamax_ab")

        ## Check cached values first
        stateKey = self.getStateHash(state)
        if stateKey in self.transpositionTable:
            cachedValue, cachedDepth, cachedBestMove = self.transpositionTable[stateKey]
            ## If we have a cached value at >= depth, we can use it
            if cachedDepth >= depth:
                return (cachedValue, None)
        else:
            cachedBestMove = None

        ## Check Terminal / depth limit
        if state.is_terminal():
            ## Return the 0..1 terminal utility from the CURRENT player's perspective
            terminalValue = state.terminal_utility()  # must be in [0..1]
            self.transpositionTable[stateKey] = (terminalValue, depth, None)
            return (terminalValue, None)

        if depth == 0:
            ## Just evaluate and return
            val = self.evaluateState(state)
            self.transpositionTable[stateKey] = (val, depth, None)
            return (val, None)

        bestValue = -float('inf')
        bestAction = None
        actions = state.get_all_valid_actions()

        if cachedBestMove and cachedBestMove in actions:
            val = self._evaluate_single_move(state, cachedBestMove, depth, alpha, beta)
            if val > bestValue:
                bestValue = val
                bestAction = cachedBestMove

            alpha = max(alpha, bestValue)
            if alpha >= beta:
                ## We got a cutoff using the TT best move alone
                self.transpositionTable[stateKey] = (bestValue, depth, bestAction)
                return (bestValue, bestAction)

            actions.remove(cachedBestMove) ## Dont evaluate again

        np.random.shuffle(actions)

        ## This evaluates the remaining moves
        for action in actions:
            val = self._evaluate_single_move(state, action, depth, alpha, beta)

            if val > bestValue:
                bestValue = val
                bestAction = action

            alpha = max(alpha, bestValue)

            if alpha >= beta:
                break

        self.transpositionTable[stateKey] = (bestValue, depth, bestAction)

        return (bestValue, bestAction)

    def _evaluate_single_move(self, state: 'State', action: 'Action', depth: int, alpha: float, beta: float) -> float:
        """Helper to evaluate a single move in negamax."""
        childState = state.change_state(action)
        childValue, _ = self.negamax(childState.invert(), depth - 1, 1.0 - beta, 1.0 - alpha)
        val = 1.0 - childValue
        return val

    def getStateHash(self, state: State) -> tuple:
        board_flat = tuple(state.board.flatten())
        meta_flat = tuple(state.local_board_status.flatten())
        return (board_flat, meta_flat, state.fill_num)

    def evaluateState(self, state: State) -> float:
        """
        Evaluates the state using the formula
        """
        features = extract_features(state)
        evaluatedValue = np.dot(self.weights, features) + self.bias
        ## Sigmoid Function
        value = 1 / (1 + np.exp(-evaluatedValue))
        value = np.clip(value, 0.001, 0.999)
        return value.item()

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        self.startTime = time.time()

        # if np.count_nonzero(state.board) == 0 and state.fill_num == 1:
        #     return (1, 1, 1, 1)

        bestAction = state.get_random_valid_action()

        for d in range(1, self.searchDepth + 1):
            # self.transpositionTable.clear()
            try:
                val, action = self.negamax(state, d, alpha=0.0, beta=1.0)
                if action is not None:
                    bestAction = action
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

    ## Expanded local features
    features.append(p1LocalWins * diffTotalLocalThreats)
    features.append(boardFillRatio * len(valid_moves))
    features.append((metaThreatsP1 - metaThreatsP2) * (totalLocalThreatsP1 - totalLocalThreatsP2))
    features.append(metaThreatsP1 * totalLocalThreatsP1)
    # features.append(boardFillRatio * p1LocalWins) ## Got worse...
    features.append(centerCountP1 * metaThreatsP1)
    features.append(centerCountP2 * metaThreatsP2) ## no difference at all ##Add back later
    features.append(p1LocalWins * metaThreatsP1)
    features.append(p2LocalWins * metaThreatsP2) ##Add back later


    ## Above this is already 0.2526 best
    features.append(emptyBoards * diffTotalLocalThreats)
    features.append(len(valid_moves) * diffTotalLocalThreats)

    ## Above best is 0.2524
    features.append(diffLocalWins * diffTotalLocalThreats)
    features.append(diffLocalWins * (metaThreatsP1 - metaThreatsP2))

    features.append(player * len(valid_moves)) ## Strong feature
    features.append(player * boardFillRatio)

    is_late_game = 1.0 if boardFillRatio > 0.7 else 0.0
    features.append(is_late_game)

    oneMoveAwayBoardsP1 = countLocalBoardsOneMoveFromWin(board, localBoardStatus, 1)
    oneMoveAwayBoardsP2 = countLocalBoardsOneMoveFromWin(board, localBoardStatus, 2)
    features.append(oneMoveAwayBoardsP1)
    features.append(oneMoveAwayBoardsP2)

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

def countLocalBoardsOneMoveFromWin(board, localBoardStatus, player):
    count = 0
    for big_r in range(3):
        for big_c in range(3):
            ## skip closed boards
            if localBoardStatus[big_r][big_c] != 0:
                continue
            sub_board = board[big_r][big_c]
            if is_board_one_move_away(sub_board, player):
                count += 1
    return count


def is_board_one_move_away(sub_board: np.ndarray, player: int) -> bool:
    """
    Returns True if there's at least one row/column/diagonal in the 3x3 sub_board
    that contains exactly 2 'player' tokens and 1 empty cell (0).
    Otherwise, returns False.
    """
    ## Check rows and columns
    for r in range(3):
        row = sub_board[r, :]
        if np.count_nonzero(row == player) == 2 and np.count_nonzero(row == 0) == 1:
            return True

    for c in range(3):
        col = sub_board[:, c]
        if np.count_nonzero(col == player) == 2 and np.count_nonzero(col == 0) == 1:
            return True

    ## Check diagonals
    firstDiag = sub_board.diagonal()
    secDiag = np.array([sub_board[0, 2], sub_board[1, 1], sub_board[2, 0]])

    if (np.count_nonzero(firstDiag == player) == 2 and np.count_nonzero(firstDiag == 0) == 1):
        return True
    if (np.count_nonzero(secDiag == player) == 2 and np.count_nonzero(secDiag == 0) == 1):
        return True

    return False













class RandomAgentC3:
    def __init__(self, searchDepth: int = 5):
        """Instantiates your agent.
        """
        self.searchDepth = searchDepth
        self.timeLimit = 2.97
        self.startTime = None
        self.transpositionTable = {}
        ## From training (hardcoded for now)
        self.weights = np.array([
            [
                2.63846181e-02, -4.84127726e-04,  1.72024578e-01, -2.03120902e-01,
                -2.78002173e-02,  1.39335498e-01, -2.30715387e-02,  2.11378157e-01,
                -2.22294822e-01, -7.20849708e-02,  3.45746689e-02, -1.23919314e-02,
                2.26355158e-02, -4.54940367e-03, -3.48353549e-03, -1.13967806e-02,
                -3.11757158e-03, -1.52657665e-02, -2.34105391e-03, -1.70872193e-02,
                -9.27404314e-03, -8.40950012e-03, -9.45400074e-02,  8.66745561e-02,
                -3.67267057e-02,  1.83778778e-01, -3.65473703e-02,  2.97409035e-02,
                1.05117917e-01,  4.97692823e-02,  5.86513989e-03, -1.24060281e-03,
                9.45399422e-03,  3.98962945e-03, -1.04771694e-02, -5.71270986e-03,
                7.00664893e-03,  1.80023238e-02, -1.69641193e-04, -2.57642940e-02,
                3.23661952e-03, -2.09874772e-02, -8.20159018e-01
            ]
        ], dtype=np.float32)
        # Define the bias as a NumPy array with dtype float32
        self.bias = np.array([-0.02859862], dtype=np.float32)
    def negamax(self, state: 'State', depth: int, alpha: float = 0.0, beta: float = 1.0) -> Tuple[float, Optional['Action']]:
        """
        Negamax-like search in [0..1] range.
        Returns (value, bestAction) from the perspective of the current player.
        """
        ## Timeout check
        if time.time() - self.startTime >= self.timeLimit:
            raise TimeoutError("Time limit exceeded in negamax_ab")
        ## Check cached values first
        stateKey = self.getStateHash(state)
        if stateKey in self.transpositionTable:
            cachedValue, cachedDepth, cachedBestMove = self.transpositionTable[stateKey]
            ## If we have a cached value at >= depth, we can use it
            if cachedDepth >= depth:
                return (cachedValue, None)
        else:
            cachedBestMove = None
        ## Check Terminal / depth limit
        if state.is_terminal():
            ## Return the 0..1 terminal utility from the CURRENT player's perspective
            terminalValue = state.terminal_utility()  # must be in [0..1]
            self.transpositionTable[stateKey] = (terminalValue, depth, None)
            return (terminalValue, None)
        if depth == 0:
            ## Just evaluate and return
            val = self.evaluateState(state)
            self.transpositionTable[stateKey] = (val, depth, None)
            return (val, None)
        bestValue = -float('inf')
        bestAction = None
        actions = state.get_all_valid_actions()
        ## Change from shallow sort to reduce overhead
        # np.random.shuffle(actions)
        # for action in actions:
        #     childState = state.change_state(action)
        #     ## 'invert()' flips perspective so child's "current player" is the other side
        #     childValue, _ = self.negamax(childState.invert(), depth - 1, 1.0 - beta, 1.0 - alpha)
        #     val = 1.0 - childValue
        #     if val > bestValue:
        #         bestValue = val
        #         bestAction = action
        #     alpha = max(alpha, val)
        #     if alpha >= beta:
        #         break
        # self.transpositionTable[stateKey] = (bestValue, depth)
        if cachedBestMove and cachedBestMove in actions:
            val = self._evaluate_single_move(state, cachedBestMove, depth, alpha, beta)
            if val > bestValue:
                bestValue = val
                bestAction = cachedBestMove
            alpha = max(alpha, bestValue)
            if alpha >= beta:
                # We got a cutoff using the TT best move alone
                self.transpositionTable[stateKey] = (bestValue, depth, bestAction)
                return (bestValue, bestAction)
            actions.remove(cachedBestMove) ## Dont evaluate again
        np.random.shuffle(actions)
        for action in actions:
            val = self._evaluate_single_move(state, action, depth, alpha, beta)
            if val > bestValue:
                bestValue = val
                bestAction = action
            alpha = max(alpha, bestValue)
            if alpha >= beta:
                break
        self.transpositionTable[stateKey] = (bestValue, depth, bestAction)
        return (bestValue, bestAction)
    def _evaluate_single_move(self, state: 'State', action: 'Action', depth: int, alpha: float, beta: float) -> float:
        """Helper to evaluate a single move in negamax."""
        childState = state.change_state(action)
        childValue, _ = self.negamax(childState.invert(), depth - 1, 1.0 - beta, 1.0 - alpha)
        val = 1.0 - childValue
        return val
    def getStateHash(self, state: State) -> tuple:
        board_flat = tuple(state.board.flatten())
        meta_flat = tuple(state.local_board_status.flatten())
        return (board_flat, meta_flat, state.fill_num)
    def evaluateState(self, state: State) -> float:
        """
        Evaluates the state using the formula
        """
        features = extract_features3(state)
        evaluatedValue = np.dot(self.weights, features) + self.bias
        ## Sigmoid Function
        value = 1 / (1 + np.exp(-evaluatedValue))
        value = np.clip(value, 0.001, 0.999)
        return value.item()
    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.
        Parameters
        ---------------
        state: The board to make a move on.
        """
        self.startTime = time.time()
        # if np.count_nonzero(state.board) == 0 and state.fill_num == 1:
        #     return Action(1, 1, 1, 1)
        bestAction = state.get_random_valid_action()
        for d in range(1, self.searchDepth + 1):
            # self.transpositionTable.clear()
            try:
                val, action = self.negamax(state, d, alpha=0.0, beta=1.0)
                if action is not None:
                    bestAction = action
            except TimeoutError:
                break
        return bestAction
###### Feature Extraction Functions ######
def extract_features3(state: State) -> np.ndarray:
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
    ## Expanded local features
    features.append(p1LocalWins * diffTotalLocalThreats)
    features.append(boardFillRatio * len(valid_moves))
    features.append((metaThreatsP1 - metaThreatsP2) * (totalLocalThreatsP1 - totalLocalThreatsP2))
    features.append(metaThreatsP1 * totalLocalThreatsP1)
    # features.append(boardFillRatio * p1LocalWins) ## Got worse...
    features.append(centerCountP1 * metaThreatsP1)
    features.append(centerCountP2 * metaThreatsP2) ## no difference at all ##Add back later
    features.append(p1LocalWins * metaThreatsP1)
    features.append(p2LocalWins * metaThreatsP2) ##Add back later
    ## Above this is already 0.2526 best
    features.append(emptyBoards * diffTotalLocalThreats)
    features.append(len(valid_moves) * diffTotalLocalThreats)
    ## Above best is 0.2524
    features.append(diffLocalWins * diffTotalLocalThreats)
    features.append(diffLocalWins * (metaThreatsP1 - metaThreatsP2))
    features.append(player * len(valid_moves)) ## Strong feature
    features.append(player * boardFillRatio)
    return np.array(features, dtype=np.float32)












class FlipHeadsAgent:
    def __init__(self, searchDepth: int = 5):
        """Instantiates your agent.
        """
        self.searchDepth = searchDepth
        self.timeLimit = 2.97
        self.startTime = None
        self.transpositionTable = {}

        self.weights = np.array([
            [
                7.2659180e-02, -3.8044654e-02,  6.4946517e-02, -1.2955402e-01,
                -3.9456803e-02,  2.3085727e-01, -3.7610121e-02,  2.1632835e-01,
                -2.0999669e-01, -6.2585562e-02,  5.9532642e-02, -1.9105729e-02,
                1.5340303e-02,  2.5824830e-03,  2.2374811e-03, -4.0264279e-03,
                6.0867430e-03, -4.2396574e-03,  4.9112993e-03, -9.7423540e-03,
                1.1421884e-03, -4.5417435e-04,  1.1976173e-01, -1.1147457e-01,
                1.1422707e-01,  2.4233215e-01, -9.7669996e-03,  3.1778868e-02,
                4.5579016e-02, -1.5556336e-02, -2.2332771e-02,  2.8126400e-03,
                5.9841294e-04,  6.3325535e-03, -7.4769869e-03,  6.0152751e-03,
                -4.0913047e-03, -5.7163942e-03,  7.6326067e-03, -2.7055874e-02,
                -2.2681704e-04, -3.7831811e-03,  2.4426510e-03, -2.1976752e-02,
                -1.0060562e+00, -1.8229434e-02, -9.3061494e-04,  9.3725502e-02,
                -1.0455957e-01
            ]
        ], dtype=np.float32)

        self.bias = np.array([-0.00686483], dtype=np.float32)



    def negamax(self, state: 'State', depth: int, alpha: float = 0.0, beta: float = 1.0) -> Tuple[float, Optional['Action']]:
        """
        Negamax-like search in [0..1] range.
        Returns (value, bestAction) from the perspective of the current player.
        """

        ## Timeout check
        if time.time() - self.startTime >= self.timeLimit:
            raise TimeoutError("Time limit exceeded in negamax")

        ## Check cached values first
        stateKey = self.getStateHash(state)
        if stateKey in self.transpositionTable:
            cachedValue, cachedDepth, cachedBestMove = self.transpositionTable[stateKey]
            ## If we have a cached value at >= depth, we can use it
            if cachedDepth >= depth:
                return (cachedValue, None)
        else:
            cachedBestMove = None

        ## Check Terminal / depth limit
        if state.is_terminal():
            ## Return the 0..1 terminal utility from the CURRENT player's perspective
            terminalValue = state.terminal_utility()  # must be in [0..1]
            self.transpositionTable[stateKey] = (terminalValue, depth, None)
            return (terminalValue, None)

        if depth == 0:
            ## Just evaluate and return
            val = self.evaluateState(state)
            self.transpositionTable[stateKey] = (val, depth, None)
            return (val, None)

        bestValue = -float('inf')
        bestAction = None
        actions = state.get_all_valid_actions()

        if cachedBestMove and cachedBestMove in actions:
            val = self._evaluate_single_move(state, cachedBestMove, depth, alpha, beta)
            if val > bestValue:
                bestValue = val
                bestAction = cachedBestMove

            alpha = max(alpha, bestValue)
            if alpha >= beta:
                ## We got a cutoff using the TT best move alone
                self.transpositionTable[stateKey] = (bestValue, depth, bestAction)
                return (bestValue, bestAction)

            actions.remove(cachedBestMove) ## Dont evaluate again

        np.random.shuffle(actions)

        ## This evaluates the remaining moves
        for action in actions:
            val = self._evaluate_single_move(state, action, depth, alpha, beta)

            if val > bestValue:
                bestValue = val
                bestAction = action

            alpha = max(alpha, bestValue)

            if alpha >= beta:
                break

        self.transpositionTable[stateKey] = (bestValue, depth, bestAction)

        return (bestValue, bestAction)

    def _evaluate_single_move(self, state: 'State', action: 'Action', depth: int, alpha: float, beta: float) -> float:
        """Helper to evaluate a single move in negamax."""
        childState = state.change_state(action)
        childValue, _ = self.negamax(childState.invert(), depth - 1, 1.0 - beta, 1.0 - alpha)
        val = 1.0 - childValue
        return val

    def getStateHash(self, state: State) -> tuple:
        board_flat = tuple(state.board.flatten())
        meta_flat = tuple(state.local_board_status.flatten())
        return (board_flat, meta_flat, state.fill_num)

    def evaluateState(self, state: State) -> float:
        """
        Evaluates the state using the formula
        """
        features = extract_features4(state)
        evaluatedValue = np.dot(self.weights, features) + self.bias
        ## Sigmoid Function
        value = 1 / (1 + np.exp(-evaluatedValue))
        value = np.clip(value, 0.001, 0.999)
        return value.item()

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        self.startTime = time.time()
        # if np.count_nonzero(state.board) == 0 and state.fill_num == 1:
        #     return (1, 1, 1, 1)

        bestAction = state.get_random_valid_action()

        for d in range(1, self.searchDepth + 1):
            # self.transpositionTable.clear()
            try:
                val, action = self.negamax(state, d, alpha=0.0, beta=1.0)
                if action is not None:
                    bestAction = action
            except TimeoutError:
                break

        return bestAction


###### Feature Extraction Functions ######

def extract_features4(state: State) -> np.ndarray:
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

    ## Expanded local features
    features.append(p1LocalWins * diffTotalLocalThreats)
    features.append(p2LocalWins * diffTotalLocalThreats)
    features.append(boardFillRatio * len(valid_moves))
    features.append((metaThreatsP1 - metaThreatsP2) * (totalLocalThreatsP1 - totalLocalThreatsP2))
    features.append(metaThreatsP1 * totalLocalThreatsP1)
    features.append(metaThreatsP2 * totalLocalThreatsP2)
    # features.append(boardFillRatio * p1LocalWins) ## Got worse...
    features.append(centerCountP1 * metaThreatsP1)
    features.append(centerCountP2 * metaThreatsP2) ## no difference at all ##Add back later
    features.append(p1LocalWins * metaThreatsP1)
    features.append(p2LocalWins * metaThreatsP2) ##Add back later


    ## Above this is already 0.2526 best
    features.append(emptyBoards * diffTotalLocalThreats)
    features.append(len(valid_moves) * diffTotalLocalThreats)

    ## Above best is 0.2524
    features.append(diffLocalWins * diffTotalLocalThreats)
    features.append(diffLocalWins * (metaThreatsP1 - metaThreatsP2))

    features.append(player * len(valid_moves)) ## Strong feature
    features.append(player * boardFillRatio) ## Strong one too

    is_late_game = 1.0 if boardFillRatio > 0.7 else 0.0
    features.append(is_late_game)
    features.append(is_late_game * diffTotalLocalThreats) ## Originally removed

    oneMoveAwayBoardsP1 = countLocalBoardsOneMoveFromWin(board, localBoardStatus, 1)
    oneMoveAwayBoardsP2 = countLocalBoardsOneMoveFromWin(board, localBoardStatus, 2)
    features.append(oneMoveAwayBoardsP1)
    features.append(oneMoveAwayBoardsP2)

    return np.array(features, dtype=np.float32)














class RandomAgentD3:
    def __init__(self, searchDepth: int = 6):
        """Instantiates your agent.
        """
        self.searchDepth = searchDepth
        self.timeLimit = 2.99
        self.startTime = None
        self.transpositionTable = {}

        ## From training (hardcoded for now)
        self.weights = np.array([
            [
                2.63846181e-02, -4.84127726e-04,  1.72024578e-01, -2.03120902e-01,
                -2.78002173e-02,  1.39335498e-01, -2.30715387e-02,  2.11378157e-01,
                -2.22294822e-01, -7.20849708e-02,  3.45746689e-02, -1.23919314e-02,
                2.26355158e-02, -4.54940367e-03, -3.48353549e-03, -1.13967806e-02,
                -3.11757158e-03, -1.52657665e-02, -2.34105391e-03, -1.70872193e-02,
                -9.27404314e-03, -8.40950012e-03, -9.45400074e-02,  8.66745561e-02,
                -3.67267057e-02,  1.83778778e-01, -3.65473703e-02,  2.97409035e-02,
                1.05117917e-01,  4.97692823e-02,  5.86513989e-03, -1.24060281e-03,
                9.45399422e-03,  3.98962945e-03, -1.04771694e-02, -5.71270986e-03,
                7.00664893e-03,  1.80023238e-02, -1.69641193e-04, -2.57642940e-02,
                3.23661952e-03, -2.09874772e-02, -8.20159018e-01
            ]
        ], dtype=np.float32)

        # Define the bias as a NumPy array with dtype float32
        self.bias = np.array([-0.02859862], dtype=np.float32)


    def negamax(self, state: 'State', depth: int, alpha: float = 0.0, beta: float = 1.0) -> Tuple[float, Optional['Action']]:
        """
        Negamax-like search in [0..1] range.
        Returns (value, bestAction) from the perspective of the current player.
        """

        ## Timeout check
        if time.time() - self.startTime >= self.timeLimit:
            raise TimeoutError("Time limit exceeded in negamax_ab")

        ## Check cached values first
        stateKey = self.getStateHash(state)
        if stateKey in self.transpositionTable:
            cachedValue, cachedDepth, cachedBestMove = self.transpositionTable[stateKey]
            ## If we have a cached value at >= depth, we can use it
            if cachedDepth >= depth:
                return (cachedValue, None)
        else:
            cachedBestMove = None

        ## Check Terminal / depth limit
        if state.is_terminal():
            ## Return the 0..1 terminal utility from the CURRENT player's perspective
            terminalValue = state.terminal_utility()  # must be in [0..1]
            self.transpositionTable[stateKey] = (terminalValue, depth, None)
            return (terminalValue, None)

        if depth == 0:
            ## Just evaluate and return
            val = self.evaluateState(state)
            self.transpositionTable[stateKey] = (val, depth, None)
            return (val, None)

        bestValue = -float('inf')
        bestAction = None
        actions = state.get_all_valid_actions()

        ## Change from shallow sort to reduce overhead
        # np.random.shuffle(actions)

        # for action in actions:

        #     childState = state.change_state(action)
        #     ## 'invert()' flips perspective so child's "current player" is the other side
        #     childValue, _ = self.negamax(childState.invert(), depth - 1, 1.0 - beta, 1.0 - alpha)
        #     val = 1.0 - childValue

        #     if val > bestValue:
        #         bestValue = val
        #         bestAction = action

        #     alpha = max(alpha, val)

        #     if alpha >= beta:
        #         break

        # self.transpositionTable[stateKey] = (bestValue, depth)

        if cachedBestMove and cachedBestMove in actions:
            val = self._evaluate_single_move(state, cachedBestMove, depth, alpha, beta)
            if val > bestValue:
                bestValue = val
                bestAction = cachedBestMove

            alpha = max(alpha, bestValue)
            if alpha >= beta:
                # We got a cutoff using the TT best move alone
                self.transpositionTable[stateKey] = (bestValue, depth, bestAction)
                return (bestValue, bestAction)

            actions.remove(cachedBestMove) ## Dont evaluate again

        np.random.shuffle(actions)

        for action in actions:
            val = self._evaluate_single_move(state, action, depth, alpha, beta)

            if val > bestValue:
                bestValue = val
                bestAction = action

            alpha = max(alpha, bestValue)

            if alpha >= beta:
                break

        self.transpositionTable[stateKey] = (bestValue, depth, bestAction)

        return (bestValue, bestAction)

    def _evaluate_single_move(self, state: 'State', action: 'Action', depth: int, alpha: float, beta: float) -> float:
        """Helper to evaluate a single move in negamax."""
        childState = state.change_state(action)
        childValue, _ = self.negamax(childState.invert(), depth - 1, 1.0 - beta, 1.0 - alpha)
        val = 1.0 - childValue
        return val

    def getStateHash(self, state: State) -> tuple:
        board_flat = tuple(state.board.flatten())
        meta_flat = tuple(state.local_board_status.flatten())
        return (board_flat, meta_flat, state.fill_num)

    def evaluateState(self, state: State) -> float:
        """
        Evaluates the state using the formula
        """
        features = extract_features2(state)
        evaluatedValue = np.dot(self.weights, features) + self.bias
        ## Sigmoid Function
        value = 1 / (1 + np.exp(-evaluatedValue))
        value = np.clip(value, 0.001, 0.999)
        return value.item()

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        self.startTime = time.time()
        bestAction = state.get_random_valid_action()

        for d in range(1, self.searchDepth + 1):
            # self.transpositionTable.clear()
            try:
                val, action = self.negamax(state, d, alpha=0.0, beta=1.0)
                if action is not None:
                    bestAction = action
            except TimeoutError:
                break

        return bestAction



###### Feature Extraction Functions ######

def extract_features2(state: State) -> np.ndarray:
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

    ## Expanded local features
    features.append(p1LocalWins * diffTotalLocalThreats)
    features.append(boardFillRatio * len(valid_moves))
    features.append((metaThreatsP1 - metaThreatsP2) * (totalLocalThreatsP1 - totalLocalThreatsP2))
    features.append(metaThreatsP1 * totalLocalThreatsP1)
    # features.append(boardFillRatio * p1LocalWins) ## Got worse...
    features.append(centerCountP1 * metaThreatsP1)
    features.append(centerCountP2 * metaThreatsP2) ## no difference at all ##Add back later
    features.append(p1LocalWins * metaThreatsP1)
    features.append(p2LocalWins * metaThreatsP2) ##Add back later


    ## Above this is already 0.2526 best
    features.append(emptyBoards * diffTotalLocalThreats)
    features.append(len(valid_moves) * diffTotalLocalThreats)

    ## Above best is 0.2524
    features.append(diffLocalWins * diffTotalLocalThreats)
    features.append(diffLocalWins * (metaThreatsP1 - metaThreatsP2))

    features.append(player * len(valid_moves)) ## Strong feature
    features.append(player * boardFillRatio)

    return np.array(features, dtype=np.float32)





class SimplestAgent:
    def __init__(self, searchDepth: int = 6):
        """Instantiates your agent.
        """
        self.searchDepth = searchDepth
        self.timeLimit = 2.95
        self.startTime = None
        self.transpositionTable = {}

        self.weights = np.array([
            [
                6.5757811e-02, -6.8906374e-02,  6.9117315e-02, -6.1823167e-02,
                2.5279328e-01, -2.4969099e-01, -9.8872995e-03,  6.0097682e-03,
                1.4365486e-04, -4.7026817e-03,  5.5049933e-02, -5.4368228e-02,
                6.2644958e-02, -6.0675163e-02
            ]
        ], dtype=np.float32)

        self.bias = np.array([0.00096601], dtype=np.float32)



    def negamax(self, state: 'State', depth: int, alpha: float = 0.0, beta: float = 1.0) -> Tuple[float, Optional['Action']]:
        """
        Negamax-like search in [0..1] range.
        Returns (value, bestAction) from the perspective of the current player.
        """
        # print(depth)
        ## Timeout check
        if time.time() - self.startTime >= self.timeLimit:
            raise TimeoutError("Time limit exceeded in negamax")

        ## Check cached values first
        stateKey = self.getStateHash(state)
        if stateKey in self.transpositionTable:
            cachedValue, cachedDepth = self.transpositionTable[stateKey]
            ## If we have a cached value at >= depth, we can use it
            if cachedDepth >= depth:
                return (cachedValue, None)

        ## Check Terminal / depth limit
        if state.is_terminal():
            ## Return the 0..1 terminal utility from the CURRENT player's perspective
            terminalValue = state.terminal_utility()  # must be in [0..1]
            self.transpositionTable[stateKey] = (terminalValue, depth)
            return (terminalValue, None)

        if depth == 0:
            ## Just evaluate and return
            val = self.evaluateState(state)
            self.transpositionTable[stateKey] = (val, depth)
            return (val, None)

        bestValue = -float('inf')
        bestAction = None
        actions = state.get_all_valid_actions()

        np.random.shuffle(actions)

        ## This evaluates the remaining moves
        for action in actions:
            val = self._evaluate_single_move(state, action, depth, alpha, beta)

            if val > bestValue:
                bestValue = val
                bestAction = action

            alpha = max(alpha, bestValue)

            if alpha >= beta:
                break

        self.transpositionTable[stateKey] = (bestValue, depth)

        return (bestValue, bestAction)

    def _evaluate_single_move(self, state: 'State', action: 'Action', depth: int, alpha: float, beta: float) -> float:
        """Helper to evaluate a single move in negamax."""
        childState = state.change_state(action)
        childValue, _ = self.negamax(childState.invert(), depth - 1, 1.0 - beta, 1.0 - alpha)
        val = 1.0 - childValue
        return val

    def getStateHash(self, state: State) -> tuple:
        board_flat = tuple(state.board.flatten())
        meta_flat = tuple(state.local_board_status.flatten())
        return (board_flat, meta_flat, state.fill_num)

    def evaluateState(self, state: State) -> float:
        """
        Evaluates the state using the formula
        """
        features = extract_features5(state)
        evaluatedValue = np.dot(self.weights, features) + self.bias
        ## Sigmoid Function
        value = 1 / (1 + np.exp(-evaluatedValue))
        value = np.clip(value, 0.001, 0.999)
        return value.item()

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        self.startTime = time.time()

        # if np.count_nonzero(state.board) == 0 and state.fill_num == 1:
        #     return (1, 1, 1, 1)

        bestAction = state.get_random_valid_action()

        for d in range(1, self.searchDepth + 1):
            # self.transpositionTable.clear()
            try:
                val, action = self.negamax(state, d, alpha=0.0, beta=1.0)
                if action is not None:
                    bestAction = action
            except TimeoutError:
                break

        return bestAction





def extract_features5(state: State) -> np.ndarray:
    local_status = state.local_board_status
    board = state.board
    features = []

    # Previous 8 features
    features.append(np.sum(local_status == 1))  # P1 local wins
    features.append(np.sum(local_status == 2))  # P2 local wins
    features.append(1 if local_status[1][1] == 1 else 0)
    features.append(1 if local_status[1][1] == 2 else 0)
    features.append(countTwoInARow(local_status, 1))
    features.append(countTwoInARow(local_status, 2))

    p1_centers = 0
    p2_centers = 0
    for i in range(3):
        for j in range(3):
            if local_status[i][j] == 0:
                center = board[i][j][1][1]
                if center == 1:
                    p1_centers += 1
                elif center == 2:
                    p2_centers += 1
    features.append(p1_centers)
    features.append(p2_centers)

    # New features start here

    # 8. Empty local boards
    features.append(np.sum(local_status == 0))

    # 9. Forced board open (if you're forced into a finished board, you can move freely)
    features.append(hasFreedomMove(state.local_board_status, state.prev_local_action))

    # 10-11. Threats in local boards (2-in-a-rows)
    p1_threats = 0
    p2_threats = 0
    for i in range(3):
        for j in range(3):
            if local_status[i][j] == 0:
                sub_board = board[i][j]
                p1_threats += countTwoInARow(sub_board, 1)
                p2_threats += countTwoInARow(sub_board, 2)
    features.append(p1_threats)
    features.append(p2_threats)

    # 12-13. Meta board lines with exactly 1 win by each player
    features.append(metaLinePotential(local_status, player=1))
    features.append(metaLinePotential(local_status, player=2))

    return np.array(features, dtype=np.float32)


class LousiestStudentAgent(StudentAgent):
    def choose_action(self, state: State) -> Action:
        # If you're using an existing Player 1 agent, you may need to invert the state
        # to have it play as Player 2. Uncomment the next line to invert the state.
        # state = state.invert()

        # Choose a random valid action from the current game state
        return state.get_random_valid_action()




class HuberLossAgent:
    def __init__(self, searchDepth: int = 6):
        """Instantiates your agent.
        """
        self.searchDepth = searchDepth
        self.timeLimit = 2.95
        self.startTime = None
        self.transpositionTable = {}

        self.weights = np.array([
            [
                7.45331571e-02,  6.63284659e-02, -4.30941060e-02,  9.44342539e-02,
                -9.35389921e-02,  2.43891716e-01, -2.33127862e-01, -1.22456532e-02,
                1.81713142e-02, -1.28765637e-02,  1.37973860e-01,  2.39836182e-02,
                -1.66359842e-02,  5.39807901e-02, -3.73200923e-02,  5.50543845e-01,
                1.73693746e-02,  4.05496284e-02,  5.15195318e-02,  9.81016755e-02,
                -1.02771714e-01,  5.53233549e-03, -1.69599766e-03,  9.63509083e-03,
                -5.59633225e-03, -8.06251287e-01, -1.43164000e-03,  1.01557858e-02,
                3.00204698e-02, -4.22811843e-02,  1.05814740e-01, -3.53328767e-03,
                9.38229356e-03,  9.35406052e-03,  1.98401641e-02,  1.09952542e-14,
                5.42764934e-41, -4.57274739e-11,  2.66423832e-39, -1.34671654e-17,
                -4.56991455e-40,  5.62269608e-40,  7.46223662e-40, -3.91734578e-03
            ]
        ], dtype=np.float32)

        self.bias = np.array([0.09446607], dtype=np.float32)

    def negamax(self, state: 'State', depth: int, alpha: float = 0.0, beta: float = 1.0) -> Tuple[float, Optional['Action']]:
        """
        Negamax-like search in [0..1] range.
        Returns (value, bestAction) from the perspective of the current player.
        """
        # print(depth)
        ## Timeout check
        if time.time() - self.startTime >= self.timeLimit:
            raise TimeoutError("Time limit exceeded in negamax")

        ## Check cached values first
        stateKey = self.getStateHash(state)
        if stateKey in self.transpositionTable:
            cachedValue, cachedDepth = self.transpositionTable[stateKey]
            ## If we have a cached value at >= depth, we can use it
            if cachedDepth >= depth:
                return (cachedValue, None)

        ## Check Terminal / depth limit
        if state.is_terminal():
            ## Return the 0..1 terminal utility from the CURRENT player's perspective
            terminalValue = state.terminal_utility()  # must be in [0..1]
            self.transpositionTable[stateKey] = (terminalValue, depth)
            return (terminalValue, None)

        if depth == 0:
            ## Just evaluate and return
            val = self.evaluateState(state)
            self.transpositionTable[stateKey] = (val, depth)
            return (val, None)

        bestValue = -float('inf')
        bestAction = None
        actions = state.get_all_valid_actions()

        np.random.shuffle(actions)

        ## This evaluates the remaining moves
        for action in actions:
            val = self._evaluate_single_move(state, action, depth, alpha, beta)

            if val > bestValue:
                bestValue = val
                bestAction = action

            alpha = max(alpha, bestValue)

            if alpha >= beta:
                break

        self.transpositionTable[stateKey] = (bestValue, depth)

        return (bestValue, bestAction)

    def _evaluate_single_move(self, state: 'State', action: 'Action', depth: int, alpha: float, beta: float) -> float:
        """Helper to evaluate a single move in negamax."""
        childState = state.change_state(action)
        childValue, _ = self.negamax(childState.invert(), depth - 1, 1.0 - beta, 1.0 - alpha)
        val = 1.0 - childValue
        return val

    def getStateHash(self, state: State) -> tuple:
        board_flat = tuple(state.board.flatten())
        meta_flat = tuple(state.local_board_status.flatten())
        return (board_flat, meta_flat, state.fill_num)

    def evaluateState(self, state: State) -> float:
        """
        Evaluates the state using the formula
        """
        features = extract_features6(state)
        evaluatedValue = np.dot(self.weights, features) + self.bias
        ## Sigmoid Function
        value = 1 / (1 + np.exp(-evaluatedValue))
        value = np.clip(value, 0.001, 0.999)
        return value.item()

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        self.startTime = time.time()

        # if np.count_nonzero(state.board) == 0 and state.fill_num == 1:
        #     return (1, 1, 1, 1)

        bestAction = state.get_random_valid_action()

        for d in range(1, self.searchDepth + 1):
            # self.transpositionTable.clear()
            try:
                val, action = self.negamax(state, d, alpha=0.0, beta=1.0)
                if action is not None:
                    bestAction = action
            except TimeoutError:
                break

        return bestAction


def extract_features6(state: State) -> np.ndarray:
    local_status = state.local_board_status
    board = state.board
    features = []
    features.append(state.fill_num)

    validMoves = state.get_all_valid_actions()
    # Previous 8 features
    p1Wins = np.sum(local_status == 1)
    p2Wins = np.sum(local_status == 2)
    diffLocalWins = p1Wins - p2Wins
    features.append(np.sum(local_status == 1))  # P1 local wins
    features.append(np.sum(local_status == 2))  # P2 local wins
    features.append(1 if local_status[1][1] == 1 else 0)
    features.append(1 if local_status[1][1] == 2 else 0)
    features.append(countTwoInARow(local_status, 1))
    features.append(countTwoInARow(local_status, 2))

    p1_centers = 0
    p2_centers = 0
    for i in range(3):
        for j in range(3):
            if local_status[i][j] == 0:
                center = board[i][j][1][1]
                if center == 1:
                    p1_centers += 1
                elif center == 2:
                    p2_centers += 1
    features.append(p1_centers)
    features.append(p2_centers)

    # New features start here

    # 8. Empty local boards
    features.append(np.sum(local_status == 0))

    # 9. Forced board open (if you're forced into a finished board, you can move freely)
    features.append(hasFreedomMove(state.local_board_status, state.prev_local_action))

    # 10-11. Threats in local boards (2-in-a-rows)
    p1_threats = 0
    p2_threats = 0
    for i in range(3):
        for j in range(3):
            if local_status[i][j] == 0:
                sub_board = board[i][j]
                p1_threats += countTwoInARow(sub_board, 1)
                p2_threats += countTwoInARow(sub_board, 2)
    features.append(p1_threats)
    features.append(p2_threats)

    # 12-13. Meta board lines with exactly 1 win by each player
    features.append(metaLinePotential(local_status, player=1))
    features.append(metaLinePotential(local_status, player=2))

    boardFillRatio = np.count_nonzero(board) / 81.0
    features.append(boardFillRatio)

    # New: Game phase indicators
    is_early_game = 1.0 if boardFillRatio < 0.3 else 0.0
    is_mid_game = 1.0 if 0.3 <= boardFillRatio <= 0.7 else 0.0
    is_end_game = 1.0 if boardFillRatio > 0.7 else 0.0
    features.extend([is_early_game, is_mid_game, is_end_game])

    # One-move-away counts for both players
    oneMoveAwayP1 = countLocalBoardsOneMoveFromWin(board, state.local_board_status, player=1)
    oneMoveAwayP2 = countLocalBoardsOneMoveFromWin(board, state.local_board_status, player=2)
    features.append(oneMoveAwayP1)
    features.append(oneMoveAwayP2)

    # Nearly full local boards overall
    nearlyFull = countNearlyFullLocalBoards(board, state.local_board_status)
    features.append(nearlyFull)

    forced_empty, forced_threat_diff = analyzeForcedBoard(state)
    features.append(forced_empty)
    features.append(forced_threat_diff)

    features.append(state.fill_num * len(validMoves)) ## Strong feature
    features.append(state.fill_num * boardFillRatio)

    cornerCountP1 = countCornerTokens(board, state.local_board_status, 1)
    cornerCountP2 = countCornerTokens(board, state.local_board_status, 2)
    features.append(cornerCountP1)
    features.append(cornerCountP2)

    features.append(is_end_game * (p1_threats - p2_threats))
    features.append(is_mid_game * diffLocalWins)
    features.append(diffLocalWins)


    ##.2256 above
    totalMoves = np.count_nonzero(board)  # All tokens on board
    openingFlag = 1.0 if totalMoves < 5 else 0.0
    features.append(openingFlag)

    local_counts = []
    for i in range(3):
        for j in range(3):
            local_counts.append(np.count_nonzero(board[i][j]))
    local_balance = np.std(local_counts)
    features.append(local_balance)

    contested = 0
    for i in range(3):
        for j in range(3):
            if local_status[i][j] == 0:
                sub_board = board[i][j]
                if (np.count_nonzero(sub_board == 1) > 0 and np.count_nonzero(sub_board == 2) > 0):
                    contested += 1
    features.append(contested)

    volatility = 0.0
    count_open = 0
    for i in range(3):
        for j in range(3):
            if local_status[i][j] == 0:
                sub_board = board[i][j]
                diff = abs(np.count_nonzero(sub_board == 1) - np.count_nonzero(sub_board == 2))
                volatility += diff
                count_open += 1
    if count_open > 0:
        volatility /= count_open
    features.append(volatility)

    meta_line_crit = metaLineCriticalIndicators(board)  # local_status is the meta board
    features.extend(meta_line_crit)

    forced_quality = forcedBoardQuality(state)
    features.append(forced_quality)

    return np.array(features, dtype=np.float32)


def countNearlyFullLocalBoards(board: np.ndarray, localBoardStatus: np.ndarray) -> int:
    count = 0
    for big_r in range(3):
        for big_c in range(3):
            if localBoardStatus[big_r][big_c] != 0:
                continue
            sub_board = board[big_r][big_c]
            empties = np.count_nonzero(sub_board == 0)
            if empties <= 3:  # Board nearly full: 6 or more tokens placed
                count += 1
    return count

def analyzeForcedBoard(state: State) -> Tuple[int, int]:
    lastAction = state.prev_local_action
    if lastAction is None:
        return (0, 0)
    forced_r = lastAction[0] % 3
    forced_c = lastAction[1] % 3
    if state.local_board_status[forced_r][forced_c] != 0:
        return (0, 0)
    forced_board = state.board[forced_r][forced_c]
    empty_in_forced = np.count_nonzero(forced_board == 0)

    # Count two-in-a-row threats in forced board (simple version)
    threatP1 = countTwoInARow(forced_board, 1)
    threatP2 = countTwoInARow(forced_board, 2)
    threat_diff = threatP1 - threatP2
    return (empty_in_forced, threat_diff)


def countCornerTokens(board: np.ndarray, localBoardStatus: np.ndarray, player: int) -> int:
    count = 0
    corners = [(0,0), (0,2), (2,0), (2,2)]
    for big_r in range(3):
        for big_c in range(3):
            if localBoardStatus[big_r][big_c] != 0:
                continue
            sub_board = board[big_r][big_c]
            for (i, j) in corners:
                if sub_board[i, j] == player:
                    count += 1
    return count


def metaLineCriticalIndicators(meta: np.ndarray) -> list[int]:
    """
    For each meta-line (row, column, or diagonal) in the 3x3 meta board (local_board_status),
    returns 1 if the line is "critical" (i.e., either player has exactly 2 wins and 1 open),
    otherwise 0. Returns a list of 8 binary indicators (3 rows, 3 columns, 2 diagonals).
    """
    crit = []
    # Rows
    for i in range(3):
        row = meta[i, :]
        if ((np.count_nonzero(row == 1) == 2 and np.count_nonzero(row == 0) == 1) or
            (np.count_nonzero(row == 2) == 2 and np.count_nonzero(row == 0) == 1)):
            crit.append(1)
        else:
            crit.append(0)
    # Columns
    for j in range(3):
        col = meta[:, j]
        if ((np.count_nonzero(col == 1) == 2 and np.count_nonzero(col == 0) == 1) or
            (np.count_nonzero(col == 2) == 2 and np.count_nonzero(col == 0) == 1)):
            crit.append(1)
        else:
            crit.append(0)
    # Diagonals
    diag1 = np.array([meta[0,0], meta[1,1], meta[2,2]])
    if ((np.count_nonzero(diag1 == 1) == 2 and np.count_nonzero(diag1 == 0) == 1) or
        (np.count_nonzero(diag1 == 2) == 2 and np.count_nonzero(diag1 == 0) == 1)):
        crit.append(1)
    else:
        crit.append(0)
    diag2 = np.array([meta[0,2], meta[1,1], meta[2,0]])
    if ((np.count_nonzero(diag2 == 1) == 2 and np.count_nonzero(diag2 == 0) == 1) or
        (np.count_nonzero(diag2 == 2) == 2 and np.count_nonzero(diag2 == 0) == 1)):
        crit.append(1)
    else:
        crit.append(0)
    return crit


def forcedBoardQuality(state: State) -> int:
    """
    Returns 1 if the forced board (determined by state.prev_local_action) is open and
    has 3 or fewer empty cells (i.e. likely to be decided quickly), otherwise 0.
    """
    if state.prev_local_action is None:
        return 0
    forced_r = state.prev_local_action[0] % 3
    forced_c = state.prev_local_action[1] % 3
    # Check if the forced board is open
    if state.local_board_status[forced_r][forced_c] != 0:
        return 0
    forced_board = state.board[forced_r][forced_c]
    empties = np.count_nonzero(forced_board == 0)
    return 1 if empties <= 3 else 0



def run(your_agent: StudentAgent, opponent_agent: StudentAgent, start_num: int):
    your_agent_stats = {"timeout_count": 0, "invalid_count": 0}
    opponent_agent_stats = {"timeout_count": 0, "invalid_count": 0}
    turn_count = 0

    state = State(fill_num=start_num)

    while not state.is_terminal():
        turn_count += 1

        agent_name = "your_agent" if state.fill_num == 1 else "opponent_agent"
        agent = your_agent if state.fill_num == 1 else opponent_agent
        stats = your_agent_stats if state.fill_num == 1 else opponent_agent_stats

        start_time = time.time()
        action = agent.choose_action(state.clone())
        end_time = time.time()

        random_action = state.get_random_valid_action()
        if end_time - start_time > 3:
            print(f"{agent_name} timed out!")
            stats["timeout_count"] += 1
            action = random_action
        if not state.is_valid_action(action):
            print(f"{agent_name} made an invalid action!")
            stats["invalid_count"] += 1
            action = random_action

        state = state.change_state(action)

    print(f"== {your_agent.__class__.__name__} (1) vs {opponent_agent.__class__.__name__} (2) - First Player: {start_num} ==")

    if state.terminal_utility() == 1:
        print("You win!")
    elif state.terminal_utility() == 0:
        print("You lose!")
    else:
        print("Draw")

    for agent_name, stats in [("your_agent", your_agent_stats), ("opponent_agent", opponent_agent_stats)]:
        print(f"{agent_name} statistics:")
        print(f"Timeout count: {stats['timeout_count']}")
        print(f"Invalid count: {stats['invalid_count']}")

    print(f"Turn count: {turn_count}\n")

your_agent = lambda: HuberLossAgent()
opponent_agent = lambda: LousiestStudentAgent()

run(your_agent(), opponent_agent(), 1)
run(opponent_agent(), your_agent(), 1)