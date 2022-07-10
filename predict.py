import paddle
import dataset
import model
import numpy as np
from PIL import Image as PilImage
import matplotlib.pyplot as plt
from paddle.vision.transforms import transforms as T


IMAGE_SIZE = (160, 160)  # 图片输入大小

# 原图和标签图片地址
resources_path = "./resources/Oxford-IIIT Pet"

# 定义训练和验证数据集
predict_dataset = dataset.PetDataset(mode='test', image_size=IMAGE_SIZE)
print("images num predict:{}".format(len(predict_dataset)))

# 定义模型
num_classes = 4
network = model.PetNet(num_classes)
model = paddle.Model(network)

# 加载模型
model.prepare()
model.load('./output/UNet')

# 指定在 CPU 上验证
paddle.device.set_device('cpu')

plt.figure(figsize=(10, 6))

i = 0
mask_idx = 0

with open(resources_path + '/test.txt', 'r') as f:
    for line in f.readlines():
        image_path, label_path = line.strip().split('\t')

        resize_t = T.Resize(IMAGE_SIZE)
        image = resize_t(PilImage.open(image_path))
        label = resize_t(PilImage.open(label_path))

        image = np.array(image).astype('uint8')
        label = np.array(label).astype('uint8')

        if i > 5:
            break
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis("off")

        plt.subplot(2, 3, i + 2)
        plt.imshow(label, cmap='gray')
        plt.title('Label')
        plt.axis("off")

        # 从测试集中取出一张图片
        img, label = predict_dataset[mask_idx]

        # 将图片shape从3*64*64变为1*3*64*64，增加一个batch维度，以匹配模型输入格式要求
        img_batch = np.expand_dims(img.astype('float32'), axis=0)

        # 在测试集上对模型进行验证
        predict_result = model.predict_batch(img_batch)[0]
        data = predict_result[0].transpose((1, 2, 0))
        mask = np.argmax(data, axis=-1)

        plt.subplot(2, 3, i + 3)
        plt.imshow(mask.astype('uint8'), cmap='gray')
        plt.title('Predict')
        plt.axis("off")
        i += 3
        mask_idx += 1

plt.show()
