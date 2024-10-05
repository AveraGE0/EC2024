"""Neural controller equivalent to the one given in demo_controller (just made cleaner)"""
from evoman.controller import Controller
import numpy as np


def sigmoid_activation(x: np.ndarray) -> np.ndarray:
    """Simple sigmoid function for a numpy array (taken from demo_controller.py)

    Args:
        x (np.ndarray): input

    Returns:
        np.ndarray: element-wise sigmoid
    """
    return 1./(1. + np.exp(-x))

# TODO: Synchronize structure of own neural controller with the pre-implemented one
# implements controller structure for player
class NeuralController(Controller):
    """Neural network based controller. For simplicity always has one hidden layer."""
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            hidden_size: int
    ):
        super().__init__()
        self.n_inputs, self.n_outputs = n_inputs, n_outputs

        # this controller will only have one hidden layer
        self.w1_matrix = np.zeros(shape=(n_inputs, hidden_size))
        self.w2_matrix = np.zeros(shape=(hidden_size, n_outputs))

        self.bias1 = np.zeros(shape=(1, hidden_size))
        self.bias2 = np.zeros(shape=(1, n_outputs))

    def get_genome_size(self) -> int:
        """Calculates the correct size of the genome (the number of network parameters).

        Returns:
            int: Size of the expected genome (equivalent to the network parameters including bias)
        """
        return self.w1_matrix.size + self.w2_matrix.size + self.bias1.size + self.bias2.size

    def set(self, genome: list, _=None) -> None:
        """Sets the GENOME of the controller.

        Args:
            genome (list): Genome being updated.
            _ (): Unused parameter.
        """
        assert self.get_genome_size() == len(genome), ValueError(
            f"The length of the given genome ({len(genome)}, does not "\
            f"align with the expected size ({self.get_genome_size()}))"
        )

        if not isinstance(genome, np.ndarray):
            genome = np.array(genome)

        assert isinstance(genome, np.ndarray)

        b1_end = self.bias1.size
        w1_end = b1_end + self.w1_matrix.size
        b2_end = w1_end + self.bias2.size
        w2_end = b2_end + self.w2_matrix.size

        self.bias1 = genome[:b1_end].reshape(self.bias1.shape)
        self.w1_matrix = genome[b1_end:w1_end].reshape(self.w1_matrix.shape)
        self.bias2 = genome[w1_end:b2_end].reshape(self.bias2.shape)
        self.w2_matrix = genome[b2_end:w2_end].reshape(self.w2_matrix.shape)

    def control(self, params: list, cont=None) -> list[bool]:
        """Method being called with the current input from the game (states).
        It returns the output actions as boolean value per action for the next
        move. The values are determined by the Neural Network.

        Args:
            params (list): Inputs (usually 20 sensor inputs).
            cont (any): Unused parameter.

        Returns:
            list[bool]: Actions in the following order: [left, right, jump, shoot, release].
        """
        inputs = np.expand_dims(np.array(params), axis=0)
        # Perform min-max normalization over inputs (optional)
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min()).astype(np.float64)
        # input -> hidden
        output = sigmoid_activation(inputs.dot(self.w1_matrix) + self.bias1)
        # hidden -> output
        output = sigmoid_activation(output.dot(self.w2_matrix)+ self.bias2)[0]
        # thresholding the output with 0.5
        output = np.where(output > 0.5, 1, 0)

        return output.tolist()
