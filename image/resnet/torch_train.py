import os
from pathlib import Path

import colossalai
import psutil
import torch
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer
from titans.utils import barrier_context
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

DATA_ROOT = Path(os.environ.get('DATA', './data'))


def main(rank=0):
    parser = colossalai.get_default_parser()
    parser.add_argument('--use_trainer', action='store_true', help='whether to use trainer')
    parser.add_argument('--torch_debug', action='store_true', help="whether to debug")
    args = parser.parse_args()

    # if args.torch_debug:
    #     colossalai.launch(config='./config.py', rank=rank, world_size=1, host='localhost', port=29500, backend='nccl')
    # else:
    #     colossalai.launch_from_torch(config='./config.py')

    logger = get_dist_logger()

    import torchvision.models as models
    model = models.shufflenet_v2_x1_0(num_classes=10)
    # # build resnet
    # model = ResNet18()

    # build dataloaders
    train_dataset = CIFAR10(root=DATA_ROOT,
                            download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(size=32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                            ]))

    test_dataset = CIFAR10(root=DATA_ROOT,
                           train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                           ]))

    import config
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        pin_memory=True,
        num_workers=psutil.cpu_count(False) - 2 - 2
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=True,
    )

    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    # engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
    #     model,
    #     optimizer,
    #     criterion,
    #     train_dataloader,
    #     test_dataloader,
    # )

    model = model.cuda()
    if not args.use_trainer:
        for epoch in range(config.NUM_EPOCHS):
            torch.set_grad_enabled(True)
            model.train()
            if rank == 0:
                train_dl = tqdm(train_dataloader)
            else:
                train_dl = train_dataloader
            for img, label in train_dl:
                img = img.cuda()
                label = label.cuda()

                optimizer.zero_grad()
                output = model(img)
                train_loss = criterion(output, label)
                train_loss.backward(train_loss)
                optimizer.step()
            lr_scheduler.step()

            model.eval()
            torch.set_grad_enabled(False)
            correct = 0
            total = 0
            for img, label in test_dataloader:
                img = img.cuda()
                label = label.cuda()

                with torch.no_grad():
                    output = model(img)
                    test_loss = criterion(output, label)
                pred = torch.argmax(output, dim=-1)
                correct += torch.sum(pred == label)
                total += img.size(0)

            print(
                f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {test_loss:.5}, acc: {correct / total:.5}, "
                f"lr: {lr_scheduler.get_last_lr()[0]:.5g}, ranks=[0]",
                )
    else:
        # build a timer to measure time
        timer = MultiTimer()

        # create a trainer object
        trainer = Trainer(engine=None, timer=timer, logger=logger)

        # define the hooks to attach to the trainer
        hook_list = [
            hooks.LossHook(),
            hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
            hooks.AccuracyHook(accuracy_func=Accuracy()),
            hooks.LogMetricByEpochHook(logger),
            hooks.LogMemoryByEpochHook(logger),
            hooks.LogTimingByEpochHook(timer, logger),

            # you can uncomment these lines if you wish to use them
            # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
            # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
        ]

        # start training
        trainer.fit(train_dataloader=train_dataloader,
                    epochs=gpc.config.NUM_EPOCHS,
                    test_dataloader=test_dataloader,
                    test_interval=1,
                    hooks=hook_list,
                    display_progress=True)


if __name__ == '__main__':
    main()
