'''Test model'''

from ml.model import TinyVGG

if __name__ == "__main__":
    model = TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=3,
    )
    from torchinfo import summary
    summary(model, input_size=[1, 3, 64, 64])
