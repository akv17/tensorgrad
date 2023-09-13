import argparse

import tensorgrad
from .model import ViT
from .dataset import Dataset
from ..util import DataLoader, Evaluator, Trainer


def main(
    device='cpu',
    epochs=10,
    batch_size=32,
    truncate_train=1000,
    truncate_test=100,
    patch_size=(4, 4),
    embedding_dim=128,
    num_heads=2,
    num_layers=1,
):
    dataset_train = Dataset(patch_size=patch_size, is_train=True, truncate=truncate_train)
    dataset_test = Dataset(patch_size=patch_size, is_train=False, truncate=truncate_test)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size)
    model = ViT(
        in_features=dataset_train.num_features,
        seq_len=dataset_train.seq_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--device',
        default='cpu',
        type=str,
        help='device to train on',
    )
    parser.add_argument(
        '--epochs',
        default=20,
        type=int,
        help='number of epochs',
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='batch size',
    )
    parser.add_argument(
        '--truncate_train',
        default=10000,
        type=int,
        help='number of training samples',
    )
    parser.add_argument(
        '--truncate_test',
        default=1000,
        type=int,
        help='number of test samples'
    )
    parser.add_argument(
        '--patch_size',
        default=4,
        type=int,
        help='ViT patch size'
    )
    parser.add_argument(
        '--embed',
        default=128,
        type=int,
        help='ViT embedding dimensionality (d_model)'
    )
    parser.add_argument(
        '--num_heads',
        default=2,
        type=int,
        help='ViT number of attention heads per single encoder layer'
    )
    parser.add_argument(
        '--num_layers',
        default=1,
        type=int,
        help='ViT number of encoder layers'
    )
    args = parser.parse_args()
    main(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        truncate_train=args.truncate_train,
        truncate_test=args.truncate_test,
        patch_size=(args.patch_size, args.patch_size),
        embedding_dim=args.embed,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
