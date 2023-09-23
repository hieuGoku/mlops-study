'''Test data module'''

from ml.data_module import Food3DataModule


def test_data_module():
    '''Tests the data module.'''
    data_module = Food3DataModule(batch_size=8, num_workers=2)
    data_module.prepare_data()
    data_module.setup()

    assert len(data_module.train_dataloader()) == 1500 // 8
    assert len(data_module.val_dataloader()) == 500 // 8

    for batch in data_module.train_dataloader():
        assert batch[0].shape == (8, 3, 64, 64)
        assert batch[1].shape == (8,)
        break

    for batch in data_module.val_dataloader():
        assert batch[0].shape == (8, 3, 64, 64)
        assert batch[1].shape == (8,)
        break


if __name__ == "__main__":
    # pylint: disable=line-too-long
    data_module = Food3DataModule(batch_size=8, num_workers=2)
    data_module.prepare_data()
    data_module.setup()

    # print(f"Number of training batches: {len(data_module.train_dataloader())}")
    # print(f"Number of validation batches: {len(data_module.val_dataloader())}")

    # for batch in data_module.train_dataloader():
    #     print(f"Shape of batch train: {batch[0].shape}")
    #     break

    # for batch in data_module.val_dataloader():
    #     print(f"Shape of batch val: {batch[0].shape}")
    #     break

