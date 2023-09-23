'''Converts the trained model to ONNX format.'''

import torch
import hydra
import logging
from ml.data_module import Food3DataModule
from ml.lightning_module import Food3LM


logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def convert_model(cfg):
    '''
    Converts the trained model to ONNX format and saves it in the models directory.
    '''
    model_path = cfg.onnx.model_path
    logger.info(f"Loading pre-trained model from: {model_path}")

    model = Food3LM.load_from_checkpoint(model_path)

    data_module = Food3DataModule(
                    cfg.data_module.batch_size,
                    cfg.data_module.num_workers
                )
    data_module.prepare_data()
    data_module.setup()

    input_batch = next(iter(data_module.train_dataloader()))
    input_sample = {
        # input_batch[0] is the images tensor
        # input_batch[1] is the labels tensor
        "images": input_batch[0],
    }

    # Export the model
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        model,                         # model being run
        input_sample["images"],        # model input (or a tuple for multiple inputs)
        cfg.onnx.onnx_path,            # where to save the model
        export_params=True,
        opset_version=10,
        input_names=["input"],         # the model's input names
        output_names=["output"],       # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    print(
        f"Model converted successfully. ONNX format model is at: {cfg.onnx.onnx_path}"
    )


if __name__ == "__main__":
    convert_model()
