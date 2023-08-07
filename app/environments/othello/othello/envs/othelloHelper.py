neighbors = {}
neighborsUp = {}
neighborsUpRight = {}
neighborsUpLeft = {}
neighborsDown = {}
neighborsDownRight = {}
neighborsDownLeft = {}
neighborsLeft = {}
neighborsRight = {}
quickDirection = {
    0: neighborsUp,
    1: neighborsUpRight,
    2: neighborsRight,
    3: neighborsDownRight,
    4: neighborsDown,
    5: neighborsDownLeft,
    6: neighborsLeft,
    7: neighborsUpLeft}

def createSets(board):
    x = set()
    o = set()
    for count, value in enumerate(board):
        if value == 'o':
            o.add(count)
        elif value == 'x':
            x.add(count)
    return x, o

def inBounds(x, y):
    if x < 0 or x >= 8 or y < 0 or y >= 8:
        return None
    return x + (y * 8)

def createNeighbors():
    for value in range(64):
        x, y = value % 8, value // 8
        neighborsUp[value] = inBounds(x, (y - 1))
        neighborsDown[value] = inBounds(x, (y + 1))
        neighborsRight[value] = inBounds((x + 1), y)
        neighborsLeft[value] = inBounds((x - 1), y)
        neighborsUpLeft[value] = inBounds((x - 1), (y - 1))
        neighborsUpRight[value] = inBounds((x + 1), (y - 1))
        neighborsDownLeft[value] = inBounds((x - 1), (y + 1))
        neighborsDownRight[value] = inBounds((x + 1), (y + 1))
        neighbors[value] = {
            neighborsUp[value],
            neighborsDown[value],
            neighborsRight[value],
            neighborsLeft[value],
            neighborsUpLeft[value],
            neighborsUpRight[value],
            neighborsDownLeft[value],
            neighborsDownRight[value]}
        
def isClear(board, value):
    if value not in board[1] and value not in board[2]:
        return True
    return False

def whichGroup(spots, value):
    if spots == neighborsUp[value]:
        return 0
    if spots == neighborsUpRight[value]:
        return 1
    if spots == neighborsRight[value]:
        return 2
    if spots == neighborsDownRight[value]:
        return 3
    if spots == neighborsDown[value]:
        return 4
    if spots == neighborsDownLeft[value]:
        return 5
    if spots == neighborsLeft[value]:
        return 6
    if spots == neighborsUpLeft[value]:
        return 7

def canFlip(board, value, token):
    if token == 'o':
        opp = 'x'
    else:
        opp = 'o'
    # Up, UpRight, Right, DownRight, Down, DownLeft, Left, UpLeft
    directions = [False for x in range(0, 8)]
    contPossible = set()
    for spots in neighbors[value]:
        if spots is not None and board[0][spots] == opp:
            contPossible.add(spots)
            directions[whichGroup(spots, value)] = True
    for count, spots in enumerate(directions):
        if spots:
            direct = quickDirection[count]
            dup = value
            while direct[dup] is not None and board[0][direct[dup]] == opp:
                dup = direct[dup]
            if direct[dup] is None and board[0][dup] != opp:
                continue
            if direct[dup] is not None and board[0][direct[dup]] == token:
                return True
    return False
        
def possibleMoves(board, token):
    possibleSpots = set()
    for value in (board[1].union(board[2])):
        if isClear(board, neighborsUp[value]):
            possibleSpots.add(neighborsUp[value])
        if isClear(board, neighborsDown[value]):
            possibleSpots.add(neighborsDown[value])
        if isClear(board, neighborsRight[value]):
            possibleSpots.add(neighborsRight[value])
        if isClear(board, neighborsLeft[value]):
            possibleSpots.add(neighborsLeft[value])
        if isClear(board, neighborsUpLeft[value]):
            possibleSpots.add(neighborsUpLeft[value])
        if isClear(board, neighborsUpRight[value]):
            possibleSpots.add(neighborsUpRight[value])
        if isClear(board, neighborsDownRight[value]):
            possibleSpots.add(neighborsDownRight[value])
        if isClear(board, neighborsDownLeft[value]):
            possibleSpots.add(neighborsDownLeft[value])
    if None in possibleSpots:
        possibleSpots.remove(None)
    returnSpots = possibleSpots.copy()
    for value in possibleSpots:
        if canFlip(board, value, token) is False:
            returnSpots.remove(value)
    return sorted(list(returnSpots))

def verifyDirections(board, directionList, position, token, opp):
    savePos = position
    for count, value in enumerate(directionList):
        position = savePos
        if value:
            makeFalse = False
            dirList = quickDirection[count]
            position = dirList[position]
            while board[position] != token:
                if board[position] != opp:
                    makeFalse = True
                    break
                position = dirList[position]
                if position is None:
                    makeFalse = True
                    break
            if makeFalse:
                directionList[count] = False
    return directionList

def move(board, token, position):
    if token == 'o':
        opp = 'x'
    else:
        opp = 'o'
    if token == 'x':
        use = 1
    else:
        use = 2
    # Up, UpRight, Right, DownRight, Down, DownLeft, Left, UpLeft
    directions = [False for x in range(0, 8)]
    possible = set()
    for value in neighbors[position]:
        if value is not None and board[0][value] == opp:
            possible.add(value)
            groupNum = whichGroup(value, position)
            directions[groupNum] = True
    directions = verifyDirections(board[0], directions, position, token, opp)
    original = position
    newBoard = [board[0], board[1].copy(), board[2].copy()]
    newBoard[0] = newBoard[0][:position] + token + newBoard[0][position + 1:]
    newBoard[use].add(position)
    if use == 1:
        notUse = 2
    else:
        notUse = 1
    while any(directions):
        position = original
        saveIndex = directions.index(True)
        direct = quickDirection[saveIndex]
        position = direct[position]
        while newBoard[0][position] != token:
            if newBoard[0][position] == opp:
                newBoard[notUse].remove(position)
            newBoard[0] = newBoard[0][:position] + \
                token + newBoard[0][position + 1:]
            newBoard[use].add(position)
            position = direct[position]
        directions[saveIndex] = False
    return newBoard

def newBoardState(board, token, position):
    setX, setO = createSets(board)
    createNeighbors()
    return move([board, setX, setO], token, position)[0]

def findPossibleMoves(board, token):
    setX, setO = createSets(board)
    createNeighbors()
    return possibleMoves([board, setX, setO], token)