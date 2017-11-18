"""Solves a Sudoku Puzzle"""
import numpy as np
__all__ = ['solve']

# precompute the constraints
def _constraints():
    """Returns the constraints of a Sudoku puzzle."""
    # generate the constraints
    variables = np.arange(729, dtype=np.int16).reshape((9, 9, 9))
    constraints = []
    for axis in [2, 1, 0]:
        # axis = 0: each column has a single instance of each label
        # axis = 1: each row has a single instance of each label
        # axis = 2: each cell has a single label
        constraint = np.swapaxes(variables, -1, axis).reshape((81, 9))
        constraints.append(constraint)
    # each square has a single instance of each label
    variables = np.transpose(variables.reshape((3,3,3,3,9)), (0,2,4,1,3))
    constraint = variables.reshape((81, 9))
    constraints.append(constraint)

    # generate the variables
    variables = np.empty((729, 4), dtype=np.int16)
    for index, constraint_type in enumerate(constraints):
        constraint_index = np.arange(81*index, 81*(index+1))[:, np.newaxis]
        variables[constraint_type, index] = constraint_index

    return np.concatenate(constraints), variables
# constraint to variables, variable to constraints
c2v, v2c = _constraints()

def _neighbors():
    """Returns the neighbors of a variable"""
    neighbors = np.empty((729, 28), dtype=np.int16)
    for variable, constraints in enumerate(v2c):
        neighbors[variable] = sorted({neighbor 
            for constraint in constraints
            for neighbor in c2v[constraint]}
            - {variable})
    return neighbors
# variable to neighbors
v2n = _neighbors()

def parse(hints):
    """Yields the givens from a hint string."""
    for index, label in enumerate(hints[:81]):
        if label not in '123456789': continue
        yield 9 * index + (int(label) - 1)

def solve(givens):
    """Solves a Sudoku puzzle."""
    try: board = Board(givens)
    except ValueError: raise
    for solution in _solve(board):
        yield solution

def _solve(board):
    """Solves a Sudoku board"""
    if board.solved():
        yield board
        return
    pivot = board.pivot()
    for variable in c2v[pivot]:
        if board.eliminated[variable]: continue
        board.select(variable)
        yield from _solve(board)
        board.deselect(variable)

class Board(object):
    """A Sudoku board."""

    def __init__(self, givens=None):
        # whether or not a constraint is satisfied
        self.satisifed = np.zeros((324,), dtype=np.bool)
        # the number of times a variable has been eliminated
        self.eliminated = np.zeros((729,), dtype=np.int8)
        # the number of variables that could satisfy the constraint
        self.options = np.empty((324,), dtype=np.int8)
        self.options[:] = 9
        # the number of filled cells
        self.filled = 0
        # initialize from givens
        if givens is None: return
        for given in givens:
            self.select(given)

    def solved(self):
        """Returns True if the board is solved."""
        return self.filled == 81
        
    def select(self, variable):
        """Selects a variable to be an element of the solution."""
        if self.eliminated[variable]:
            raise ValueError('Attempted to select an eliminated variable.')
        self.eliminated[variable] = -1
        self.satisifed[v2c[variable]] = True
        for neighbor in v2n[variable]:
            if not self.eliminated[neighbor]:
                self.options[v2c[neighbor]] -= 1
            self.eliminated[neighbor] += 1
        self.filled += 1

    def deselect(self, variable):
        """Deselects a variable to be an element of the solution."""
        if self.eliminated[variable] != -1:
            raise ValueError('Atempted to deselect an unselected variable.')
        self.eliminated[variable] = 0
        self.satisifed[v2c[variable]] = False
        for neighbor in v2n[variable]:
            self.eliminated[neighbor] -= 1
            if not self.eliminated[neighbor]:
                self.options[v2c[neighbor]] += 1
        self.filled -= 1

    def pivot(self):
        """Selects a unsatisifed constraint with a minimal number of options."""
        pivot = None
        min_options = 10
        for constraint in range(324):
            if self.satisifed[constraint]: continue
            if self.options[constraint] < min_options:
                pivot = constraint
                min_options = self.options[constraint]
                if min_options <= 1: break
        return pivot

    def __str__(self):
        """Returns str(self)"""
        givens = np.flatnonzero(self.eliminated == -1)
        hints = np.zeros((81,), dtype=np.int16)
        hints[givens // 9] = givens % 9 + 1
        return ''.join(str(label) for label in hints)

if __name__ == '__main__':
    import sys
    print(str(next(solve(parse(sys.argv[1])))))
