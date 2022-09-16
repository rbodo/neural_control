import logging
import sys
from typing import Optional

import mxnet as mx
from mxnet import autograd
from torch.utils.data import DataLoader

from examples import configs
from examples.linear_rnn_lqr import LqrPipeline
from src.utils import get_data


class LqgPipeline(LqrPipeline):
    def get_loss_function(self, *args, **kwargs) -> mx.gluon.HybridBlock:
        """Define loss function as the mean square error between RNN output
        and LQG oracle."""
        return mx.gluon.loss.L2Loss()

    def get_loss(self, data: mx.nd.NDArray,
                 label: Optional[mx.nd.NDArray] = None) -> mx.nd.NDArray:
        # Move time axis from last to first position to conform to RNN
        # convention.
        data = mx.nd.moveaxis(data, -1, 0)
        data = data.as_in_context(self.device)
        label = label.as_in_context(self.device)
        with autograd.record():
            u = self.model(data)
            u = mx.nd.moveaxis(u, 0, -1)
            return self.loss_function(u, label)

    def evaluate(self, data_loader: DataLoader,
                 filename: Optional[str] = None) -> float:
        """
        Evaluate model.

        Parameters
        ----------
        data_loader
            Data loaders for train and test set. Contain trajectories in state
            space to provide initial values or learning signal.
        filename
            If specified, create an example phase diagram and save under given
            name.

        Returns
        -------
        loss
            The average performance when evaluating `loss_function` on the
            samples in `data_loader`.
        """
        validation_loss = 0
        for data, label in data_loader:
            data = mx.nd.moveaxis(data, -1, 0)
            data = data.as_in_context(self.device)
            label = label.as_in_context(self.device)
            neuralsystem_outputs = self.model(data)
            neuralsystem_outputs = mx.nd.moveaxis(neuralsystem_outputs, 0, -1)
            loss = self.loss_function(neuralsystem_outputs, label)
            validation_loss += loss.mean().asscalar()
        return validation_loss / len(data_loader)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    _config = configs.linear_rnn_lqg.get_config()

    # Get training set for RNN. The data consists of partial noisy observations
    # of a classic LQG controller in the double integrator state space. Labels
    # are provided by the LQG control output.
    _data_dict = get_data(_config, 'observations')

    pipeline = LqgPipeline(_config, _data_dict)
    pipeline.main()

    sys.exit()
