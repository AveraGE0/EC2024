from evoman_framework.neural_controller import NeuralController
from demo_controller import player_controller
import numpy as np


def test_controller_equivalence():
    np.random.seed(42)
    # test 3 just in case
    genome = np.random.uniform(low=-10, high=10, size=265)
    genome2 = np.random.uniform(low=-10, high=10, size=265)
    genome3 = np.random.uniform(low=-10, high=10, size=265)

    nc = NeuralController(
        n_inputs=20,
        n_outputs=5,
        hidden_size=10
    )
    org_controller = player_controller(10)
    inputs = list(range(20))

    for genome in [genome, genome2, genome3]:
        nc.set(genome)
        nc_result = nc.control(inputs)

        org_controller.set(genome, 20)
        org_result = org_controller.control(np.array(inputs), None)

        assert (nc_result == org_result), f"Results where not equal: {nc_result}!={org_result}"
