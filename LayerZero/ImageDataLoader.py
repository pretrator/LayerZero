from torch.utils.data import DataLoader
from torchvision import transforms

class ImageDataLoader:
    def __init__(
        self,
        dataset_cls,
        image_size,
        data_dir="data",
        batch_size=64,
        channels=3,
        num_workers=4,
        shuffle_train=True,
        download=True,
        mean=None,
        std=None,
        use_trivialaugment=True,
        use_randaugment=False,
        rand_n=2,
        rand_m=9,
        color_jitter=(0.4, 0.4, 0.4, 0.1),
        random_erase_p=0.25,
    ):
        if(dataset_cls is None):
            raise Exception("Dataset Class (dataset_cls) is not Provided")
        if(image_size is None):
            raise Exception("Image Size(image_size) is not provided")
        self.dataset_cls = dataset_cls
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.download = download

        if mean is None or std is None:
            if channels == 1:
                self.mean, self.std = (0.5,), (0.5,)
            else:
                self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            self.mean, self.std = mean, std

        self.use_trivialaugment = use_trivialaugment
        self.use_randaugment = use_randaugment
        self.rand_n = rand_n
        self.rand_m = rand_m
        self.color_jitter = color_jitter
        self.random_erase_p = random_erase_p

    def build_transforms(self, train=True):
        ops = []
        if train:
            ops.append(transforms.RandomResizedCrop(self.image_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)))
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
            ops.append(transforms.RandomApply([transforms.ColorJitter(*self.color_jitter)], p=0.8))
            ops.append(transforms.RandomRotation(degrees=10))
            
            if self.use_trivialaugment:
                try:
                    ops.append(transforms.TrivialAugmentWide())
                except AttributeError:
                    ops.append(transforms.RandAugment(num_ops=self.rand_n, magnitude=self.rand_m))
            elif self.use_randaugment:
                ops.append(transforms.RandAugment(num_ops=self.rand_n, magnitude=self.rand_m))
        else:
            ops.append(transforms.Resize(int(self.image_size * 1.14)))
            ops.append(transforms.CenterCrop(self.image_size))
        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(self.mean, self.std))
        if train and self.random_erase_p > 0:
            ops.append(transforms.RandomErasing(p=self.random_erase_p, scale=(0.02, 0.33), ratio=(0.3, 3.3)))
        return transforms.Compose(ops)

    def get_loaders(self):
        train_dataset = self.dataset_cls(
            root=self.data_dir,
            train=True,
            download=self.download,
            transform=self.build_transforms(train=True)
        )
        test_dataset = self.dataset_cls(
            root=self.data_dir,
            train=False,
            download=self.download,
            transform=self.build_transforms(train=False)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, test_loader
