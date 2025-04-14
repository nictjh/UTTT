import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from utils import load_data, State

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
    # features.extend(forced_board_9d)

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
    # features.append(is_late_game * diffTotalLocalThreats) ## Originally removed

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
    Counts how many open local boards have the center cell belonging to player
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
    Sums the two-in-a-row threats for player across all open local boards.
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
    that contains exactly 2 'player' tokens and 1 empty cell
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


###### Regression Model ######
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1) ## Output 1 scalar value

    def forward(self, x):
        return self.linear(x)

def trainModel():
    """
    Train the linear regression model using the 19 features I have
    """
    ## Load the data
    data = load_data()
    print(f"Loaded {len(data)} states")

    X = np.array([extract_features(state) for state, _ in data], dtype=np.float32)
    y = np.array([value for _, value in data], dtype=np.float32).reshape(-1, 1)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # continuous_cols = [0, 1, 2, 4, 5, 9, 10, 11, 12, 14, 15]
    # binary_cols = [3, 6, 7, 8, 13]

    # Define a column transformer that scales only continuous columns.
    # column_transformer = ColumnTransformer(
    #     transformers=[
    #         ("cont_scaler", StandardScaler(), continuous_cols),
    #         ("pass_through", "passthrough", binary_cols)
    #     ]
    # )

    # X_transformed = column_transformer.fit_transform(X)

    Xtensor = torch.tensor(X) ## Creates a tensor from the data, accepts lists, numpy array, etc...
    ytensor = torch.tensor(y)

    model = LinearRegressionModel(input_dim=X.shape[1]) ## takes in the input dimenaion size
    criterion = nn.MSELoss() ## This is my loss function, refer to notes
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=0.0001) ## Stochastic Gradient Descent
    num_epochs = 4000

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad() ## This steps clears any old gradients from the previous iterations
        outputs = model(Xtensor) ##  Predicts output y-hat using current weights
        loss = criterion(outputs, ytensor) ## Calculate loss
        loss.backward() ## computes the gradients of the loss w.r.t all params
        optimizer.step() ## Updates the weights and biases using the gradients

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item():.4f}')

    print("###### Final Weights and Bias #######")
    weights = model.linear.weight.data.numpy()
    bias = model.linear.bias.data.numpy()
    print("Weights:", weights)
    print("Bias:", bias)

if __name__ == "__main__":
    # data = load_data()
    # for state, value in data[:1]:
    #     print(state)
    #     features = extract_features(state)
    #     print("Features:", features)
    #     print("Value:", value)
    trainModel()