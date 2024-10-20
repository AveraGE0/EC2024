from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.mutation import Mutation
import numpy as np


class PMGlobalLocalProb(Mutation):

    def __init__(self, individual_mutation_prob, gene_mutation_prob, eta):
        super().__init__()
        self.mutation_operator = PolynomialMutation(prob=gene_mutation_prob, eta=eta)
        self.individual_mutation_prob = individual_mutation_prob

    def _do(self, problem, X, **kwargs):
        # Perform mutation on population
        X_mutated = self.mutation_operator._do(problem, X, **kwargs)
        # Create mask to only perform mutation with global_prob on individuals
        global_mask = np.random.rand(X.shape[0]) < self.individual_mutation_prob

        # Step 3: Use the global mask to keep mutated or original individuals
        X_final = np.where(global_mask[:, None], X_mutated, X)  # Apply mask to retain original or mutated individuals
        return X_final
