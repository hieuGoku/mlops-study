'''This is the main file for the project.'''

import logging
import hydra
from ml.utils import delete_checkpoints


logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config", version_base="1.2")
def main(cfg):
    '''Main function for the project'''
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from ml.train import train

    return train(cfg)


if __name__ == "__main__":
    main()
