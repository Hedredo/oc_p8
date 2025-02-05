import tensorflow as tf
import numpy as np
import math
import multiprocessing
import logging
import pathlib
import typing

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