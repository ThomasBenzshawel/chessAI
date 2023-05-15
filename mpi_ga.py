import chess
import copy
import numpy as np
import chess.polyglot
import chess.svg
import chess.pgn
import chess.engine
import pickle
from mpi4py import MPI
import simulate_and_evaluate as sim

pawntable = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0]

knightstable = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50]

bishopstable = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20]

rookstable = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0]

queenstable = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20]

kingstable = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30]


def evaluate_board(board, organism=None):
    if board.is_checkmate():
        if board.turn:
            return -9999
        else:
            return 9999
    if board.is_stalemate():
        return 0
    if board.is_insufficient_material():
        return 0
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    material = 100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)

    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)]
                           for i in board.pieces(chess.PAWN, chess.BLACK)])

    knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.KNIGHT, chess.BLACK)])

    bishopsq = sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                               for i in board.pieces(chess.BISHOP, chess.BLACK)])

    rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.ROOK, chess.BLACK)])

    queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                             for i in board.pieces(chess.QUEEN, chess.BLACK)])

    kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                           for i in board.pieces(chess.KING, chess.BLACK)])

    if organism == None:
        eval = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq
    else:
        eval = organism.predict(
            np.array([material, pawnsq, knightsq, bishopsq, rooksq, queensq, kingsq]).reshape((1, -1)))

        # compare performance of below
        #         if type(eval) is np.ndarray:
        #             eval = eval[0][0]

        eval = np.array(eval).flatten()
        eval = eval[0]

    if board.turn:
        return eval
    else:
        return -eval

def selectmove(depth, board, organism=None):
    bestMove = chess.Move.null()
    bestValue = -99999
    alpha = -100000
    beta = 100000
    for move in board.legal_moves:
        board.push(move)
        boardValue = -alphabeta(-beta, -alpha, depth - 1, board, organism)
        if boardValue > bestValue:
            bestValue = boardValue
            bestMove = move
        if (boardValue > alpha):
            alpha = boardValue
        board.pop()
    return (bestMove, bestValue)


def alphabeta(alpha, beta, depthleft, board, organism=None):
    bestscore = -9999
    if (depthleft == 0):
        return quiesce(alpha, beta, board, organism)
    for move in board.legal_moves:
        board.push(move)
        score = -alphabeta(-beta, -alpha, depthleft - 1, board, organism)
        board.pop()
        if (score >= beta):
            return score
        if (score > bestscore):
            bestscore = score
        if (score > alpha):
            alpha = score
    return bestscore

def quiesce(alpha, beta, board, organism=None):
    stand_pat = evaluate_board(board, organism)
    if (stand_pat >= beta):
        return beta
    if (alpha < stand_pat):
        alpha = stand_pat

    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiesce(-beta, -alpha, board, organism)
            board.pop()

            if (score >= beta):
                return beta
            if (score > alpha):
                alpha = score
    return alpha


class Organism():
    def __init__(self, dimensions, use_bias=True, output='softmax'):
        self.score = 0

        self.winner = False

        self.layers = []
        self.biases = []
        self.use_bias = use_bias
        self.output = self._activation(output)
        self.dimensions = dimensions
        for i in range(len(dimensions) - 1):
            shape = (dimensions[i], dimensions[i + 1])
            std = np.sqrt(2 / sum(shape))
            layer = np.random.normal(0, std, shape)
            bias = np.random.normal(0, std, (1, dimensions[i + 1])) * use_bias
            self.layers.append(layer)
            self.biases.append(bias)

    def _activation(self, output):
        if output == 'softmax':
            return lambda X: np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1, 1)
        if output == 'sigmoid':
            return lambda X: (1 / (1 + np.exp(-X)))
        if output == 'linear':
            return lambda X: X
        if output == 'relu':
            return lambda X: max(0, X)

    def predict(self, X):
        if not X.ndim == 2:
            raise ValueError(f'Input has {X.ndim} dimensions, expected 2')
        if not X.shape[1] == self.layers[0].shape[0]:
            raise ValueError(f'Input has {X.shape[1]} features, expected {self.layers[0].shape[0]}')
        for index, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            X = X @ layer + np.ones((X.shape[0], 1)) @ bias
            if index == len(self.layers) - 1:
                X = self.output(X)  # output activation
            else:
                X = np.clip(X, 0, np.inf)  # ReLU

        return X

    def predict_choice(self, X, deterministic=True):
        probabilities = self.predict(X)
        if deterministic:
            return np.argmax(probabilities, axis=1).reshape((-1, 1))
        if any(np.sum(probabilities, axis=1) != 1):
            raise ValueError(f'Output values must sum to 1 to use deterministic=False')
        if any(probabilities < 0):
            raise ValueError(f'Output values cannot be negative to use deterministic=False')
        choices = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            U = np.random.rand(X.shape[0])
            c = 0
            while U > probabilities[i, c]:
                U -= probabilities[i, c]
                c += 1
            else:
                choices[i] = c
        return choices.reshape((-1, 1))

    def mutate(self, stdev=0.03):
        for i in range(len(self.layers)):
            self.layers[i] += np.random.normal(0, stdev, self.layers[i].shape)
            if self.use_bias:
                self.biases[i] += np.random.normal(0, stdev, self.biases[i].shape)

    def mate(self, other, mutate=True):
        if self.use_bias != other.use_bias:
            raise ValueError('Both parents must use bias or not use bias')
        if not len(self.layers) == len(other.layers):
            raise ValueError('Both parents must have same number of layers')
        if not all(self.layers[x].shape == other.layers[x].shape for x in range(len(self.layers))):
            raise ValueError('Both parents must have same shape')

        child = copy.deepcopy(self)
        for i in range(len(child.layers)):
            pass_on = np.random.rand(1, child.layers[i].shape[1]) < 0.5
            child.layers[i] = pass_on * self.layers[i] + ~pass_on * other.layers[i]
            child.biases[i] = pass_on * self.biases[i] + ~pass_on * other.biases[i]
        if mutate:
            child.mutate()
        return child

    def save_human_readable(self, filepath):
        file = open(filepath, 'w')
        file.write('----------NEW MODEL----------\n')
        file.write('DIMENSIONS\n')
        for dimension in self.dimensions:
            file.write(str(dimension) + ',')
        file.write('\nWEIGHTS\n')
        for layer in self.layers:
            file.write('NEW LAYER\n')
            for node in layer:
                for weight in node:
                    file.write(str(weight) + ',')
                file.write('\n')
            file.write('\n')
        if self.use_bias:
            file.write('BIASES:\n')
            for layer in self.biases:
                file.write('\nNEW LAYER\n')
                for connection in layer:
                    file.write(str(connection) + ',')
        file.close()

        def save(self, filepath):
            pickle.dump(self)


def pairwise(iterable):
    # s -> (s0, s1), (s2, s3), (s4, s5), ...
    a = iter(iterable)
    return zip(a, a)

class Ecosystem():
    def __init__(self, orginism_creator, scoring_function, population_size=100, holdout='sqrt', mating=True):
        """
        origanism_creator must be a function to produce Organisms, used for the original population
        scoring_function must be a function which accepts an Organism as input and returns a float
        """
        self.population_size = population_size

        self.population = [organism_creator() for _ in range(population_size)]
        self.mating = mating

        self.rewards = []

        self.scoring_function = scoring_function
        if holdout == 'sqrt':
            self.holdout = max(1, int(np.sqrt(population_size)))
        elif holdout == 'log':
            self.holdout = max(1, int(np.log(population_size)))
        elif holdout > 0 and holdout < 1:
            self.holdout = max(1, int(holdout * population_size))
        else:
            self.holdout = max(1, int(holdout))

    def generation(self, repeats=1, keep_best=True):
        self.rewards = [self.scoring_function(x, y) for x, y in pairwise(self.population)]
        # print("Before flatten, ", self.rewards)
        self.rewards = [item for sublist in self.rewards for item in sublist]
        # print("After flatten, ", self.rewards)

        self.population = [self.population[x] for x in np.argsort(self.rewards)[::-1]]
        self.population_size = len(self.population)

        #         self.population_new = [pop for pop in self.population if pop.winner]

        #         if(len(self.population_new) > 2):
        #             self.population = self.population_new
        #             self.population_size = len(self.population)

        new_population = []
        for i in range(self.population_size):
            parent_1_idx = i % self.holdout
            # print(parent_1_idx)

            if self.mating:
                parent_2_idx = min(self.population_size - 1, int(np.random.exponential(self.holdout)))
            else:
                parent_2_idx = parent_1_idx
            offspring = self.population[parent_1_idx].mate(self.population[parent_2_idx])
            new_population.append(offspring)
        if keep_best:
            new_population[-1] = self.population[0]  # Ensure best organism survives
        self.population = new_population
        # return -1 * max(rewards)

    def mpi_generation(self, repeats=1, keep_best=True):

        # make the population and score stuff array
        population = np.array(self.population)  # parameters to send to simulate_and_evaluate
        n = population.shape[0]


        count = n // size  # number of catchments for each process to analyze
        remainder = n % size  # extra catchments if n is not a multiple of size

        if rank < remainder:  # processes with rank < remainder analyze one extra catchment
            start = rank * (count + 1)  # index of first catchment to analyze
            stop = start + count + 1  # index of last catchment to analyze
        else:
            start = rank * count + remainder
            stop = start + count

        local_pop = population[start:stop]  # get the portion of the array to be analyzed by each rank
        # run the function for each parameter set and rank
        local_results = np.array([self.scoring_function(x, y) for x, y in pairwise(local_pop)])

        if rank > 0:
            comm.Send(local_results, dest=0, tag=14)  # send results to process 0
        else:
            final_results = np.copy(local_results)  # initialize final results with results from process 0
            for i in range(1, size):  # determine the size of the array to be received from each process
                if i < remainder:
                    rank_size = count + 1
                else:
                    rank_size = count
                tmp = np.empty(len(final_results),
                               dtype=np.float)  # create empty array to receive results
                comm.Recv(tmp, source=i, tag=14)  # receive results from the process
                final_results = np.hstack((final_results, tmp))  # add the received results to the final results
            print("results")

            # Saving the value pairings in self.rewards
            # self.rewards = [item for sublist in final_results for item in sublist]
            # print("#################")
            # print(self.rewards, len(self.rewards))

            self.rewards = [x.score for x in self.population]

            print(self.rewards)
            #todo this might run into issues with lining up the organism with its actual score with mpi

            self.population = [self.population[x] for x in np.argsort(self.rewards)[::-1]]
            self.population_size = len(self.population)

            #         self.population_new = [pop for pop in self.population if pop.winner]

            #         if(len(self.population_new) > 2):
            #             self.population = self.population_new
            #             self.population_size = len(self.population)

            new_population = []
            for i in range(self.population_size):
                parent_1_idx = i % self.holdout
                # print(parent_1_idx)

                if self.mating:
                    parent_2_idx = min(self.population_size - 1, int(np.random.exponential(self.holdout)))
                else:
                    parent_2_idx = parent_1_idx
                offspring = self.population[parent_1_idx].mate(self.population[parent_2_idx])
                new_population.append(offspring)
            if keep_best:
                new_population[-1] = self.population[0]  # Ensure best organism survives
            self.population = new_population
            # return -1 * max(rewards)



    def get_best_organism(self, repeats=1, include_reward=False):
        # rewards = [np.mean(self.scoring_function(x)) for _ in range(repeats) for x in self.population]
        if include_reward:
            best = np.argsort(self.rewards)[-1]
            return self.population[best], self.rewards[best]
        else:
            return self.population[np.argsort(self.rewards)[-1]]

organism_creator = lambda: Organism([7, 32, 8, 1], output='relu')


scoring_function = lambda organism_1, organism_2 : sim.simulate_and_evaluate(organism_1, organism_2, print_game=False, trials=1)
ecosystem = Ecosystem(organism_creator, scoring_function, population_size=40, holdout=0.1, mating=True)

generations = 30
best_ai_list = []
best_ai_models = []

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print("Starting simulations")

for i in range(generations):
    if rank == 0:
        print("Starting generation", i + 1)
    ecosystem.mpi_generation()
    if rank == 0:
        best_ai = ecosystem.get_best_organism(repeats=1, include_reward=True)
        best_ai_models.append(best_ai[0])
    #ecosystem.get_best_organism().save("model.txt")
    #     best_ai_list.append(best_ai[1])
    #     print("Best AI = ", best_ai[1])