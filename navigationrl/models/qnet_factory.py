from .qnetsimple import QNetworkSimple
from .qnetdueling import QNetworkDueling

class QNetFactory:

    def __init__(self, state_size, action_size, seed):
        """A factory class for generating Q-Network class objects

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of actions
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

    def __call__(self, qnet_params):
        """Returns a Q-Network defined by qnet_params

        Args:
            qnet_params: defines the Q-Network to be generated

        Returns:
            QNetwork object
        """
        qnet_type = qnet_params['type'].lower()
        params = {key: value for key, value in qnet_params.items() if key != 'type'}
        if qnet_type == "simple":
            qnet = QNetworkSimple(self.state_size, self.action_size, self.seed,
                                  **params)
            return qnet

        elif qnet_type == 'dueling':
            qnet = QNetworkDueling(self.state_size, self.action_size, self.seed, **params)
            return qnet

        else:
            raise Exception("Q-Network type not recognized")