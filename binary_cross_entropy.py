from logger import Logging, LogLevel

import torch
import torch.nn.functional as F

if __name__ == '__main__':
    Logging.set_log_level(LogLevel.INFO)

    y_pred = torch.tensor([
        [0.3, 0.6, 0.1]
    ])
    y_target = torch.tensor([ 
        [0.0, 1.0, 0.0]
    ])

    loss = F.binary_cross_entropy(y_pred, y_target)

    Logging.log(LogLevel.INFO, "loss : {}, shape: {}".format(loss, loss.shape))
    