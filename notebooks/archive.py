import tensorflow as tf
import numpy as np
import math
import multiprocessing
import logging
import pathlib
import typing
import albumentations as A

class Dataloaderv1(tf.keras.utils.PyDataset):
    def __init__(self, images, masks, batch_size, target_size, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.num_samples = len(images)
        self.target_size = target_size

    def __len__(self):
        length = math.ceil(self.num_samples / self.batch_size)
        print(f"Number of batches: {length}")
        return length

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")

        batch_images = self.images[start_idx:end_idx]
        batch_masks = self.masks[start_idx:end_idx]

        print(f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}")

        batch_images = [
            tf.keras.utils.load_img(
                img,
                target_size=self.target_size,
                color_mode="rgb",
                interpolation="bilinear",
            )
            for img in batch_images
        ]
        batch_images = [tf.keras.utils.img_to_array(img) for img in batch_images]

        batch_masks = [
            tf.keras.utils.load_img(
                mask,
                target_size=self.target_size,
                color_mode="grayscale",
                interpolation="nearest",
            )
            for mask in batch_masks
        ]
        batch_masks = [
            tf.keras.utils.img_to_array(mask).squeeze() for mask in batch_masks
        ]

        return np.asarray(batch_images), np.asarray(batch_masks)
class Dataloaderv2(tf.keras.utils.PyDataset):
    def __init__(self, images, masks, batch_size, target_size, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.num_samples = len(images)
        self.target_size = target_size

    def __len__(self):
        length = math.ceil(self.num_samples / self.batch_size)
        print(f"Number of batches: {length}")
        return length

    def load_img_to_array(self, img_path):
        img = tf.keras.utils.load_img(
            img_path,
            target_size=self.target_size,
            color_mode="rgb",
            interpolation="bilinear",
        )
        return tf.keras.utils.img_to_array(img)

    def load_mask_to_array(self, mask_path):
        mask = tf.keras.utils.load_img(
            mask_path,
            target_size=self.target_size,
            color_mode="grayscale",
            interpolation="nearest",
        )
        return tf.keras.utils.img_to_array(mask).squeeze()

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")

        batch_images = self.images[start_idx:end_idx]
        batch_masks = self.masks[start_idx:end_idx]

        print(f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}")

        with multiprocessing.Pool(self.workers) as pool:
            batch_images = pool.map(self.load_img_to_array, batch_images)
            batch_masks = pool.map(self.load_mask_to_array, batch_masks)

        return np.asarray(batch_images), np.asarray(batch_masks)
class Dataloaderv2b(tf.keras.utils.Sequence):
    def __init__(self, images, masks, batch_size, target_size, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_samples = len(images)

    def __len__(self):
        length = math.ceil(self.num_samples / self.batch_size)
        print(f"Number of batches: {length}")
        return length

    def load_img_to_array(self, img_path):
        img = tf.keras.utils.load_img(
            img_path,
            target_size=self.target_size,
            color_mode="rgb",
            interpolation="bilinear",
        )
        return tf.keras.utils.img_to_array(img)

    def load_mask_to_array(self, mask_path):
        mask = tf.keras.utils.load_img(
            mask_path,
            target_size=self.target_size,
            color_mode="grayscale",
            interpolation="nearest",
        )
        return tf.keras.utils.img_to_array(mask).squeeze()

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")
        batch_images = self.images[start_idx:end_idx]
        batch_masks = self.masks[start_idx:end_idx]
        print(f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}")

        # Traitement séquentiel ici
        batch_images = [self.load_img_to_array(img) for img in batch_images]
        batch_masks = [self.load_mask_to_array(mask) for mask in batch_masks]

        return np.asarray(batch_images), np.asarray(batch_masks)
class Dataloaderv2c(tf.keras.utils.Sequence):
    def __init__(self, images, masks, batch_size, target_size, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_samples = len(images)

    def __len__(self):
        length = math.ceil(self.num_samples / self.batch_size)
        print(f"Number of batches: {length}")
        return length

    def load_img_to_array(self, img_path):
        img = tf.keras.utils.load_img(
            img_path,
            target_size=self.target_size,
            color_mode="rgb",
            interpolation="bilinear",
        )
        return tf.keras.utils.img_to_array(img)

    def load_mask_to_array(self, mask_path):
        mask = tf.keras.utils.load_img(
            mask_path,
            target_size=self.target_size,
            color_mode="grayscale",
            interpolation="nearest",
        )
        return tf.keras.utils.img_to_array(mask).squeeze()

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")
        batch_images = self.images[start_idx:end_idx]
        batch_masks = self.masks[start_idx:end_idx]
        print(f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}")

        images, masks = [], []
        # Traitement séquentiel ici
        for img, mask in zip(batch_images, batch_masks):
            images.append(self.load_img_to_array(img))
            masks.append(self.load_mask_to_array(mask))

        return np.asarray(images), np.asarray(masks)
class Dataloaderv3(tf.keras.utils.PyDataset):
    def __init__(self, images, masks, batch_size, target_size, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.num_samples = len(images)
        self.num_classes = num_classes
        self.target_size = target_size

    def __len__(self):
        length = math.ceil(self.num_samples / self.batch_size)
        print(f"Number of batches: {length}")
        return length

    def load_img_to_array(self, img_path):
        img = tf.keras.utils.load_img(
            img_path,
            target_size=self.target_size,
            color_mode="rgb",
            interpolation="bilinear",
        )
        return tf.keras.utils.img_to_array(img)

    def load_mask_to_array(self, mask_path):
        mask = tf.keras.utils.load_img(
            mask_path,
            target_size=self.target_size,
            color_mode="grayscale",
            interpolation="nearest",
        )
        return tf.keras.utils.img_to_array(mask).squeeze()

    def transform_mask_to_categorical(self, mask):
        return tf.keras.utils.to_categorical(mask, num_classes=self.num_classes)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")

        batch_images = self.images[start_idx:end_idx]
        batch_masks = self.masks[start_idx:end_idx]

        print(f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}")

        with multiprocessing.Pool(self.workers) as pool:
            batch_images = pool.map(self.load_img_to_array, batch_images)
            batch_masks = pool.map(self.load_mask_to_array, batch_masks)

        return np.asarray(batch_images), np.asarray(batch_masks)
class Dataloaderv3_(tf.keras.utils.PyDataset):
    def __init__(self, images, masks, batch_size, target_size, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.num_samples = len(images)
        self.num_classes = num_classes
        self.target_size = target_size

    def __len__(self):
        length = math.ceil(self.num_samples / self.batch_size)
        print(f"Number of batches: {length}")
        return length

    def load_img_to_array(self, img_path):
        img = tf.keras.utils.load_img(
            img_path,
            target_size=self.target_size,
            color_mode="rgb",
            interpolation="bilinear",
        )
        return tf.keras.utils.img_to_array(img)

    def load_mask_to_array(self, mask_path):
        mask = tf.keras.utils.load_img(
            mask_path,
            target_size=self.target_size,
            color_mode="grayscale",
            interpolation="nearest",
        )
        return tf.keras.utils.img_to_array(mask).squeeze()

    def transform_mask_to_categorical(self, mask):
        return tf.keras.utils.to_categorical(mask, num_classes=self.num_classes)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")

        batch_images = self.images[start_idx:end_idx]
        batch_masks = self.masks[start_idx:end_idx]

        print(f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}")

        with multiprocessing.Pool(self.workers) as pool:
            batch_images = pool.map(self.load_img_to_array, batch_images)
            batch_masks = pool.map(self.load_mask_to_array, batch_masks)
            batch_masks = pool.map(self.transform_mask_to_categorical, batch_masks)

        return np.asarray(batch_images), np.asarray(batch_masks)
class Dataloaderv3b(tf.keras.utils.PyDataset):
    def __init__(self, images, masks, batch_size, target_size, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.num_samples = len(images)
        self.num_classes = num_classes
        self.target_size = target_size

    def __len__(self):
        length = math.ceil(self.num_samples / self.batch_size)
        print(f"Number of batches: {length}")
        return length

    def load_img_to_array(self, img_path):
        img = tf.keras.utils.load_img(
            img_path,
            target_size=self.target_size,
            color_mode="rgb",
            interpolation="bilinear",
        )
        return tf.keras.utils.img_to_array(img)

    def load_mask_to_array(self, mask_path):
        mask = tf.keras.utils.load_img(
            mask_path,
            target_size=self.target_size,
            color_mode="grayscale",
            interpolation="nearest",
        )
        return tf.keras.utils.img_to_array(mask).squeeze()

    def transform_mask_to_categorical(self, mask):
        return tf.keras.utils.to_categorical(mask, num_classes=self.num_classes)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")

        batch_images = self.images[start_idx:end_idx]
        batch_masks = self.masks[start_idx:end_idx]

        print(f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}")

        # Traitement séquentiel ici
        batch_images = [self.load_img_to_array(img) for img in batch_images]
        batch_masks = [self.load_mask_to_array(mask) for mask in batch_masks]

        return np.asarray(batch_images), np.asarray(batch_masks)
class Dataloaderv3b_(tf.keras.utils.PyDataset):
    def __init__(self, images, masks, batch_size, target_size, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.num_samples = len(images)
        self.num_classes = num_classes
        self.target_size = target_size

    def __len__(self):
        length = math.ceil(self.num_samples / self.batch_size)
        print(f"Number of batches: {length}")
        return length

    def load_img_to_array(self, img_path):
        img = tf.keras.utils.load_img(
            img_path,
            target_size=self.target_size,
            color_mode="rgb",
            interpolation="bilinear",
        )
        return tf.keras.utils.img_to_array(img)

    def load_mask_to_array(self, mask_path):
        mask = tf.keras.utils.load_img(
            mask_path,
            target_size=self.target_size,
            color_mode="grayscale",
            interpolation="nearest",
        )
        return tf.keras.utils.img_to_array(mask).squeeze()

    def transform_mask_to_categorical(self, mask):
        return tf.keras.utils.to_categorical(mask, num_classes=self.num_classes)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")

        batch_images = self.images[start_idx:end_idx]
        batch_masks = self.masks[start_idx:end_idx]

        print(f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}")

        # Traitement séquentiel ici
        batch_images = [self.load_img_to_array(img) for img in batch_images]
        batch_masks = [
            self.transform_mask_to_categorical(
                self.load_mask_to_array(mask), num_classes=self.num_classes
            )
            for mask in batch_masks
        ]

        return np.asarray(batch_images), np.asarray(batch_masks)


class Dataloaderv4(tf.keras.utils.PyDataset):
    def __init__(
        self,
        images,
        masks,
        batch_size,
        target_size,
        num_classes,
        label_to_categorical=False,
        threadpool=False,
        rgb_norm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.num_samples = len(images)
        self.num_classes = num_classes
        self.threadpool = threadpool  # if True, use multiprocessing.Pool and set self.use_multiprocessing=False
        self.label_to_categorical = label_to_categorical
        self.rgb_norm = rgb_norm
        self.target_size = target_size

    def __len__(self):
        length = math.ceil(self.num_samples / self.batch_size)
        print(f"Number of batches: {length}")
        return length

    def load_img_to_array(self, img_path):
        img = tf.keras.utils.load_img(
            img_path,
            target_size=self.target_size,
            color_mode="rgb",
            interpolation="bilinear",
        )
        return (
            tf.keras.utils.img_to_array(img) / 255.0
            if self.rgb_norm
            else tf.keras.utils.img_to_array(img)
        )

    def load_mask_to_array(self, mask_path):
        mask = tf.keras.utils.load_img(
            mask_path,
            target_size=self.target_size,
            color_mode="grayscale",
            interpolation="nearest",
        )
        return tf.keras.utils.img_to_array(mask).squeeze()

    def transform_mask_to_categorical(self, mask):
        return tf.keras.utils.to_categorical(mask, num_classes=self.num_classes)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")

        batch_images = self.images[start_idx:end_idx]
        batch_masks = self.masks[start_idx:end_idx]

        print(f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}")

        if self.threadpool:
            self.use_multiprocessing = False
            with multiprocessing.Pool(self.workers) as pool:
                batch_images = pool.map(self.load_img_to_array, batch_images)
                batch_masks = pool.map(self.load_mask_to_array, batch_masks)
                if self.label_to_categorical:
                    batch_masks = pool.map(
                        self.transform_mask_to_categorical, batch_masks
                    )
        else:
            # Traitement séquentiel ici
            batch_images = [self.load_img_to_array(img) for img in batch_images]
            batch_masks = [self.load_mask_to_array(mask) for mask in batch_masks]
            if self.label_to_categorical:
                batch_masks = [
                    self.transform_mask_to_categorical(mask) for mask in batch_masks
                ]

        return np.asarray(batch_images), np.asarray(batch_masks)


class Dataloaderv6(tf.keras.utils.PyDataset):
    def __init__(
        self,
        image_folder: pathlib.Path,
        labels: typing.NamedTuple,
        batch_size: int,
        target_size: typing.Tuple[int, int],
        preview: int = None,  # Load only a subset of the dataset for preview.
        rgb_norm: bool = True,
        label_onehot: bool = False,
        threadpool: bool = False,
        **kwargs,
    ):
        """
        Initialize the Dataloaderv6 data generator.
        """
        super().__init__(**kwargs)
        self.image_folder = image_folder

        self.labels = labels
        self.batch_size = batch_size
        self.preview = preview
        self.rgb_norm = rgb_norm
        self.label_onehot = label_onehot
        self.target_size = target_size
        self.threadpool = threadpool  # If True, use multiprocessing.Pool

        # Get image and mask file paths using pathlib
        self.image_files = sorted(list(image_folder.glob("*leftImg8bit.png")))
        self.mask_files = sorted(list(image_folder.glob("*labelIds.png")))
        if len(self.image_files) != len(self.mask_files):
            self.logger.error(
                "Number of images (%d) and masks (%d) must be equal.",
                len(self.image_files),
                len(self.mask_files),
            )
            raise ValueError("Number of images and masks must be equal.")

        # Apply preview if requested
        if self.preview is not None:
            self.image_files = self.image_files[: self.preview]
            self.mask_files = self.mask_files[: self.preview]

        self.num_samples = len(self.image_files)
        self.num_classes = len(self.labels)
        self.table_id2category = {label.id: label.categoryId for label in labels}

        # Setup dedicated logger for this class
        self.logger = logging.getLogger("dataloader")
        self.logger.setLevel(logging.DEBUG)

        # Disable PIL logging DEBUG
        logging.getLogger("PIL").setLevel(logging.WARNING)

    def __len__(self) -> int:
        length = math.ceil(self.num_samples / self.batch_size)
        self.logger.debug(f"Number of batches: {length}")
        return length

    def load_img_to_array(self, img_path: pathlib.Path):
        img = tf.keras.utils.load_img(
            str(img_path),
            target_size=self.target_size,
            color_mode="rgb",
            interpolation="bilinear",
        )
        img_array = tf.keras.utils.img_to_array(img)
        return img_array / 255.0 if self.rgb_norm else img_array

    def load_mask_to_array(self, mask_path: pathlib.Path):
        mask = tf.keras.utils.load_img(
            str(mask_path),
            target_size=self.target_size,
            color_mode="grayscale",
            interpolation="nearest",
        )
        mask_array = tf.keras.utils.img_to_array(mask)
        # Map mask ids to categories
        mapped = np.vectorize(self.table_id2category.get)(mask_array)
        return mapped.squeeze()

    def transform_mask_to_categorical(self, mask):
        return tf.keras.utils.to_categorical(mask, num_classes=self.num_classes)

    def __getitem__(self, index: int):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")

        batch_images_paths = self.image_files[start_idx:end_idx]
        batch_masks_paths = self.mask_files[start_idx:end_idx]

        self.logger.debug(
            f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}"
        )

        if self.threadpool:
            self.use_multiprocessing = False
            with multiprocessing.Pool() as pool:
                batch_images = pool.map(self.load_img_to_array, batch_images_paths)
                batch_masks = pool.map(self.load_mask_to_array, batch_masks_paths)
                if self.label_onehot:
                    batch_masks = pool.map(
                        self.transform_mask_to_categorical, batch_masks
                    )
        else:
            batch_images = [self.load_img_to_array(path) for path in batch_images_paths]
            batch_masks = [self.load_mask_to_array(path) for path in batch_masks_paths]
            if self.label_onehot:
                batch_masks = [
                    self.transform_mask_to_categorical(mask) for mask in batch_masks
                ]

        return np.asarray(batch_images), np.asarray(batch_masks)


class Dataloaderv7(tf.keras.utils.PyDataset):
    def __init__(
        self,
        image_folder: pathlib.Path,
        labels: typing.NamedTuple,
        batch_size: int,
        target_size: typing.Tuple[int, int],
        preview: int = None,  # Load only a subset of the dataset for preview.
        rgb_norm: bool = True,
        shuffle: bool = True,
        label_onehot: bool = False,
        threadpool: bool = False,
        **kwargs,
    ):
        """
        Initialize the Dataloaderv6 data generator.
        """
        super().__init__(**kwargs)
        self.image_folder = image_folder
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.preview = preview
        self.rgb_norm = rgb_norm
        self.shuffle = shuffle
        self.label_onehot = label_onehot
        self.threadpool = threadpool  # If True, use multiprocessing.Pool

        # Get image and mask file paths using pathlib
        self.image_paths = sorted(list(image_folder.glob("*leftImg8bit.png")))
        self.mask_paths = sorted(list(image_folder.glob("*labelIds.png")))
        if len(self.image_paths) != len(self.mask_paths):
            self.logger.error(
                "Number of images (%d) and masks (%d) must be equal.",
                len(self.image_paths),
                len(self.mask_paths),
            )
            raise ValueError("Number of images and masks must be equal.")

        # Apply preview if requested
        if self.preview is not None:
            self.image_paths = self.image_paths[: self.preview]
            self.mask_paths = self.mask_paths[: self.preview]

        # Shuffle the dataset before starting
        if self.shuffle:
            self.on_epoch_end()

        self.num_samples = len(self.image_paths)
        self.table_id2category = {label.id: label.categoryId for label in self.labels}
        self.num_classes = len(set(self.table_id2category.values()))
        # Setup dedicated logger for this class
        self.logger = logging.getLogger("dataloader")
        self.logger.setLevel(logging.DEBUG)

        # Disable PIL logging DEBUG
        logging.getLogger("PIL").setLevel(logging.WARNING)

    def __len__(self) -> int:
        length = math.ceil(self.num_samples / self.batch_size)
        self.logger.debug(f"Number of batches: {length}")
        return length

    def __getitem__(self, index: int):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")

        batch_images_paths = self.image_paths[start_idx:end_idx]
        batch_masks_paths = self.mask_paths[start_idx:end_idx]

        self.logger.debug(
            f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}"
        )

        if self.threadpool:
            self.use_multiprocessing = False
            with multiprocessing.Pool() as pool:
                batch_images = pool.map(self.load_img_to_array, batch_images_paths)
                batch_masks = pool.map(self.load_mask_to_array, batch_masks_paths)
                if self.label_onehot:
                    batch_masks = pool.map(
                        self.transform_mask_to_categorical, batch_masks
                    )
        else:
            batch_images = [self.load_img_to_array(path) for path in batch_images_paths]
            batch_masks = [self.load_mask_to_array(path) for path in batch_masks_paths]
            if self.label_onehot:
                batch_masks = [
                    self.transform_mask_to_categorical(mask) for mask in batch_masks
                ]

        return np.asarray(batch_images), np.asarray(batch_masks)

    def on_epoch_end(self):
        if self.shuffle:
            zip_paths = list(zip(self.image_paths, self.mask_paths))
            np.random.shuffle(zip_paths)
            self.image_paths, self.mask_paths = zip(*zip_paths)
        else:
            pass

    def load_img_to_array(self, img_path: pathlib.Path):
        img = tf.keras.utils.load_img(
            str(img_path),
            target_size=self.target_size,
            color_mode="rgb",
            interpolation="bilinear",
        )
        img_array = tf.keras.utils.img_to_array(img)
        return img_array / 255.0 if self.rgb_norm else img_array

    def load_mask_to_array(self, mask_path: pathlib.Path):
        mask = tf.keras.utils.load_img(
            str(mask_path),
            target_size=self.target_size,
            color_mode="grayscale",
            interpolation="nearest",
        )
        mask_array = tf.keras.utils.img_to_array(mask)
        # Map mask ids to categories
        mapped = np.vectorize(self.table_id2category.get)(mask_array)
        return mapped.squeeze()

    def transform_mask_to_categorical(self, mask):
        return tf.keras.utils.to_categorical(mask, num_classes=self.num_classes)


class Dataloaderv8(tf.keras.utils.PyDataset):
    def __init__(
        self,
        image_folder: pathlib.Path,
        labels: typing.NamedTuple,
        batch_size: int,
        target_size: typing.Tuple[int, int],
        preview: int = None,  # Load only a subset of the dataset for preview.
        rgb_norm: bool = True,
        shuffle: bool = True,
        label_onehot: bool = False,
        threadpool: bool = False,
        **kwargs,
    ):
        """
        Initialize the Dataloaderv8 data generator.
        """
        super().__init__(**kwargs)

        # Logger initialization
        self.logger = self._setup_logger()

        # Set class attributes initialized with constructor arguments
        self.image_folder = image_folder
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.rgb_norm = rgb_norm
        self.shuffle = shuffle
        self.label_onehot = label_onehot
        self.threadpool = threadpool  # If True, use multiprocessing.Pool

        # Get image and mask file paths using pathlib
        self.image_paths, self.mask_paths = self._load_paths(preview)
        self._validate_paths()

        # Attributes initialized from labels elements
        self.table_id2category = {label.id: label.categoryId for label in self.labels}
        self.num_classes = len(set(self.table_id2category.values()))
        # Disable PIL logging DEBUG
        logging.getLogger("PIL").setLevel(logging.WARNING)

        # Shuffle the dataset before starting
        if self.shuffle:
            self.on_epoch_end()

    # Initialize logger
    def _setup_logger(self):
        logger = logging.getLogger("dataloader")
        logger.setLevel(logging.DEBUG)
        return logger

    # Load image and mask paths and apply preview if requested to slice the dataset
    def _load_paths(self, preview: int):
        # Get image and mask file paths using pathlib
        image_paths = sorted(list(self.image_folder.glob("*leftImg8bit.png")))
        mask_paths = sorted(list(self.image_folder.glob("*labelIds.png")))

        # Apply preview if requested
        if preview is not None:
            image_paths = image_paths[:preview]
            mask_paths = mask_paths[:preview]

        return image_paths, mask_paths

    # Validate that the number of images and masks are equal before starting
    def _validate_paths(self):
        if len(self.image_paths) != len(self.mask_paths):
            self.logger.error(
                "Number of images (%d) and masks (%d) must be equal.",
                len(self.image_paths),
                len(self.mask_paths),
            )
            raise ValueError("Number of images and masks must be equal.")

    # Return the number of samples in the dataset using dynamic property
    @property
    def num_samples(self):
        return len(self.image_paths)

    # Class method to return the number of batches in the dataset using len()
    def __len__(self) -> int:
        length = math.ceil(self.num_samples / self.batch_size)
        self.logger.debug(f"Number of batches: {length}")
        return length

    # Class method to fetch a batch of images and masks with the given index
    def __getitem__(self, index: int):
        # Calculate start and end indices for the batch
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")
        # Fetch batch image and mask paths using start and end indices
        batch_images_paths = self.image_paths[start_idx:end_idx]
        batch_masks_paths = self.mask_paths[start_idx:end_idx]
        # Log the batch fetching
        self.logger.debug(
            f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}"
        )
        # Load images and masks using multiprocessing if threadpool is True
        if self.threadpool:
            self.use_multiprocessing = False
            with multiprocessing.Pool() as pool:
                batch_images = pool.map(self.load_img_to_array, batch_images_paths)
                batch_masks = pool.map(self.load_mask_to_array, batch_masks_paths)
                # Transform masks to categorical if label_onehot is True
                if self.label_onehot:
                    batch_masks = pool.map(
                        self.transform_mask_to_categorical, batch_masks
                    )
        # Load images and masks using list comprehension if threadpool is False
        else:
            batch_images = [self.load_img_to_array(path) for path in batch_images_paths]
            batch_masks = [self.load_mask_to_array(path) for path in batch_masks_paths]
            # Transform masks to categorical if label_onehot is True
            if self.label_onehot:
                batch_masks = [
                    self.transform_mask_to_categorical(mask) for mask in batch_masks
                ]
        # Return batch images and masks as numpy arrays
        return np.asarray(batch_images), np.asarray(batch_masks)

    # Shuffle the dataset before starting a new epoch
    def on_epoch_end(self):
        if self.shuffle:
            zip_paths = list(zip(self.image_paths, self.mask_paths))
            np.random.shuffle(zip_paths)
            self.image_paths, self.mask_paths = zip(*zip_paths)
        else:
            pass

    # Normalize the image array if rgb_norm is True
    def _normalize_img(self, img_array):
        return img_array / 255.0

    # Load image to array and normalize if rgb_norm is True
    def load_img_to_array(self, img_path: pathlib.Path, display_only=False):
        img = tf.keras.utils.load_img(
            str(img_path),
            target_size=self.target_size,
            color_mode="rgb",
            interpolation="bilinear",
        )
        if display_only:
            return img
        img_array = tf.keras.utils.img_to_array(img, dtype=np.float32)
        return self._normalize_img(img_array) if self.rgb_norm else img_array

    # Load mask to array and map mask ids to categories
    def load_mask_to_array(self, mask_path: pathlib.Path):
        mask = tf.keras.utils.load_img(
            str(mask_path),
            target_size=self.target_size,
            color_mode="grayscale",
            interpolation="nearest",
        )
        mask_array = tf.keras.utils.img_to_array(mask, dtype=np.int8)
        # Map mask ids to categories
        mapped = np.vectorize(self.table_id2category.get)(mask_array)
        return mapped.squeeze()

    # Transform mask to categorical one-hot encoding
    def transform_mask_to_categorical(self, mask):
        return tf.keras.utils.to_categorical(mask, num_classes=self.num_classes)

    # Return an image and mask pair for visualization with the path
    def get_image_mask_pair_and_path(self, index: int):
        img = self.load_img_to_array(self.image_paths[index], display_only=True)
        img_array = tf.keras.utils.img_to_array(img, dtype=np.float32)
        mask = self.load_mask_to_array(self.mask_paths[index])
        return img, self._normalize_img(img_array) if self.rgb_norm else img_array, mask


class Dataloaderv9(tf.keras.utils.PyDataset):
    def __init__(
        self,
        image_folder: pathlib.Path,
        labels: typing.NamedTuple,
        batch_size: int,
        target_size: typing.Tuple[int, int],
        augmentations: False,
        preview: int = None,  # Load only a subset of the dataset for preview.
        normalize: bool = True,
        shuffle: bool = True,
        label_onehot: bool = False,
        threadpool: bool = False,
        **kwargs,
    ):
        """
        Initialize the Dataloaderv9 data generator.
        """
        super().__init__(**kwargs)

        # Logger initialization
        self.logger = self._setup_logger()

        # Set class attributes initialized with constructor arguments
        self.image_folder = image_folder
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmentations = augmentations
        self.normalize = normalize
        self.shuffle = shuffle
        self.label_onehot = label_onehot
        self.threadpool = threadpool  # If True, use multiprocessing.Pool

        # Initialize the augmentation pipeline
        if self.augmentations:
            self.compose = A.Compose(
                [
                    A.HorizontalFlip(p=0.9),
                    A.RandomBrightnessContrast(
                        p=0.2, brightness_limit=0.2, contrast_limit=0.2
                    ),
                    A.Blur(p=0.2, blur_limit=3),
                    A.HueSaturationValue(p=0.2, hue_shift_limit=20, sat_shift_limit=30),
                ]
            )

        # Get image and mask file paths using pathlib
        self.image_paths, self.mask_paths = self._load_paths(preview)
        self._validate_paths()

        # Attributes initialized from labels elements
        self.table_id2category = {label.id: label.categoryId for label in self.labels}
        self.num_classes = len(set(self.table_id2category.values()))
        # Disable PIL logging DEBUG
        logging.getLogger("PIL").setLevel(logging.WARNING)

        # Shuffle the dataset before starting
        if self.shuffle:
            self.on_epoch_end()

    # Initialize logger
    def _setup_logger(self):
        logger = logging.getLogger("dataloader")
        logger.setLevel(logging.DEBUG)
        return logger

    # Load image and mask paths and apply preview if requested to slice the dataset
    def _load_paths(self, preview: int):
        # Get image and mask file paths using pathlib
        image_paths = sorted(list(self.image_folder.glob("*leftImg8bit.png")))
        mask_paths = sorted(list(self.image_folder.glob("*labelIds.png")))

        # Apply preview if requested
        if preview is not None:
            image_paths = image_paths[:preview]
            mask_paths = mask_paths[:preview]

        return image_paths, mask_paths

    # Validate that the number of images and masks are equal before starting
    def _validate_paths(self):
        if len(self.image_paths) != len(self.mask_paths):
            self.logger.error(
                "Number of images (%d) and masks (%d) must be equal.",
                len(self.image_paths),
                len(self.mask_paths),
            )
            raise ValueError("Number of images and masks must be equal.")

    # Return the number of samples in the dataset using dynamic property
    @property
    def num_samples(self):
        return len(self.image_paths)

    # Class method to return the number of batches in the dataset using len()
    def __len__(self) -> int:
        length = math.ceil(self.num_samples / self.batch_size)
        self.logger.debug(f"Number of batches: {length}")
        return length

    # Class method to fetch a batch of images and masks with the given index
    def __getitem__(self, index: int):
        # Calculate start and end indices for the batch
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")

        # Prépare les paires (image_path, mask_path) du batch
        batch_paths = list(
            zip(
                self.image_paths[start_idx:end_idx],
                self.mask_paths[start_idx:end_idx],
            )
        )

        # Log the batch fetching
        self.logger.debug(
            f"Fetching batch {index}: start_idx={start_idx}, end_idx={end_idx}"
        )

        # Load images and masks using multiprocessing if threadpool is True
        if self.threadpool:
            self.use_multiprocessing = False
            with multiprocessing.Pool() as pool:
                results = pool.map(self.load_and_augment, batch_paths)
        # Load images and masks using list comprehension if threadpool is False
        else:
            results = [self.load_and_augment(pair) for pair in batch_paths]

        # Return batch images and masks as numpy arrays
        batch_images, batch_masks = zip(*results)
        return np.asarray(batch_images), np.asarray(batch_masks)

    # Shuffle the dataset before starting a new epoch
    def on_epoch_end(self):
        if self.shuffle:
            zip_paths = list(zip(self.image_paths, self.mask_paths))
            np.random.shuffle(zip_paths)
            self.image_paths, self.mask_paths = zip(*zip_paths)
        else:
            pass

    # Normalize the image array if normalize is True
    def _normalize_img(self, img_array):
        return img_array / 255.0

        # Transform mask to categorical one-hot encoding

    def _transform_mask_to_categorical(self, mask):
        return tf.keras.utils.to_categorical(mask, num_classes=self.num_classes)

    # Load image to array and normalize if normalize is True
    def load_img_to_array(self, img_path: pathlib.Path):
        img = tf.keras.utils.load_img(
            str(img_path),
            target_size=self.target_size,
            color_mode="rgb",
            interpolation="bilinear",
        )
        img_array = tf.keras.utils.img_to_array(img, dtype=np.float32)
        if self.normalize:
            img_array = self._normalize_img(img_array)
        return img_array

    # Load mask to array and map mask ids to categories
    def load_mask_to_array(self, mask_path: pathlib.Path):
        mask = tf.keras.utils.load_img(
            str(mask_path),
            target_size=self.target_size,
            color_mode="grayscale",
            interpolation="nearest",
        )
        mask_array = tf.keras.utils.img_to_array(mask, dtype=np.int8)
        # Map mask ids to categories
        mapped = (np.vectorize(self.table_id2category.get)(mask_array)).squeeze()
        if self.label_onehot:
            mapped = self._transform_mask_to_categorical(mapped)
        return mapped

    def load_and_augment(self, paths):
        img, mask = paths
        img = self.load_img_to_array(img)
        mask = self.load_mask_to_array(mask)
        if self.augmentations:
            augmented = self.compose(image=img, mask=mask)
            return augmented["image"], augmented["mask"]
        else:
            return img, mask

    # Return an image and mask pair for visualization with the path
    def get_image_mask_and_paths(self, index: int):
        paths = self.image_paths[index], self.mask_paths[index]
        img, mask = self.load_and_augment(paths)
        return img, mask, *paths


# Dimensions cibles
TARGET_HEIGHT = 256
TARGET_WIDTH = 512

def tf_map_table(labels: typing.NamedTuple):
    # Create a StaticVocabularyTable to map ids to categories
    table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            [label.id for label in labels],
            [label.categoryId for label in labels],
            key_dtype=tf.int64,  # Needs to be tf.int64 or tf.string
            value_dtype=tf.int64,  # Needs to be tf.int64 or tf.string
        ),
        num_oov_buckets=1,
    )
    return table

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, channels=3)  # Images en couleur
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalisation
    image = tf.image.resize(
        image, [TARGET_HEIGHT, TARGET_WIDTH], method="bilinear"
    )  # Redimensionnement
    return image


def load_mask(mask_path, tf_map_table):
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(
        mask, channels=1, dtype=tf.uint8
    )  # Annotation en niveau de gris
    # annotation = tf.map_fn(fn= lambda x: )
    mask = tf.image.resize(
        mask, [TARGET_HEIGHT, TARGET_WIDTH], method="nearest"
    )  # Redimensionnement
    mask = tf.cast(mask, tf.int64)
    return tf_map_table.lookup(mask)


def load_data(image_path, mask_path):
    image = load_image(image_path)
    mask = load_mask(mask_path)
    return image, mask


def create_dataset(folder_path, batch_size=8, preview=None):
    image_paths = sorted(list(map(str, folder_path.glob("*leftImg8bit.png"))))
    mask_paths = sorted(list(map(str, folder_path.glob("*labelIds.png"))))
    if preview is not None:
        image_paths = image_paths[:preview]
        mask_paths = mask_paths[:preview]
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset