import argparse

import tensorgrad
from .model import Model
from .util import Dataset, DataLoader, Evaluator, Trainer


def main(
    device='cpu',
    epochs=10,
    batch_size=32,
    truncate_train=1000,
    truncate_test=100,
):
    dataset_train = Dataset(is_train=True, truncate=truncate_train)
    dataset_test = Dataset(is_train=False, truncate=truncate_test)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size)
    model = Model(in_channels=1, num_classes=10)
    loss_fn = tensorgrad.nn.CrossEntropyLoss()
    optim = tensorgrad.optim.Adam(model.parameters())
    evaluator = Evaluator(dataloader=dataloader_test, model=model, device=device)
    trainer = Trainer(
        dataloader=dataloader_train,
        model=model,
        loss_fn=loss_fn,
        optimizer=optim,
        device=device,
        epochs=epochs,
        evaluator=evaluator,
    )
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        default='cpu',
        type=str,
        help=''
    )
    parser.add_argument(
        '--epochs',
        default=10,
        type=int,
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
    )
    parser.add_argument(
        '--truncate_train',
        default=1000,
        type=int,
    )
    parser.add_argument(
        '--truncate_test',
        default=100,
        type=int,
    )
    args = parser.parse_args()
    main(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        truncate_train=args.truncate_train,
        truncate_test=args.truncate_test,
    )
