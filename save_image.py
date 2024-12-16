import os
import numpy as np
import matplotlib.pyplot as plt

from example import load_dataset, unpickle


def save_images_to_folders(images, labels, label_names, output_dir='./cifar_images'):
    os.makedirs(output_dir, exist_ok=True)

    for i, label_name in enumerate(label_names):
        class_dir = os.path.join(output_dir, label_name.decode('utf-8'))
        os.makedirs(class_dir, exist_ok=True)

    for idx, (image_data, label) in enumerate(zip(images, labels)):
        class_name = label_names[label].decode('utf-8')
        class_dir = os.path.join(output_dir, class_name)

        file_name = f"{class_name}_{idx}.png"
        file_path = os.path.join(class_dir, file_name)

        image = np.array(image_data).reshape((3, 32, 32)).transpose((1, 2, 0))
        plt.imsave(file_path, image)

    print(f"Изображения успешно сохранены в директорию '{output_dir}'")


if __name__ == '__main__':
    images, labels = load_dataset()
    label_names = unpickle('./cifar-10-batches-py/batches.meta')[b'label_names']

    save_images_to_folders(images, labels, label_names)
