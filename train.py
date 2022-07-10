import os
import random
import paddle
import dataset
import model

RE_IMAGE = False  # 重新分配训练数据集
TRAIN_RATIO = 0.9  # 训练集比例
IMAGE_SIZE = (160, 160)  # 图片输入大小

# 原图和标签图片地址
resources_path = "./resources/Oxford-IIIT Pet"
origin_images_path = resources_path + "/images"
label_images_path = resources_path + "/masks"


def write_file(mode, images):
    """
    生产训练和测试文件
    """
    with open(resources_path + '/{}.txt'.format(mode), 'w') as f:
        for i in range(len(images)):
            f.write('{}\t{}\n'.format(origin_images_path + '/' + images[i], label_images_path + '/' + images[i]))


if RE_IMAGE is True:
    image_count = len([os.path.join(origin_images_path, image_name)
                       for image_name in os.listdir(origin_images_path)
                       if image_name.endswith('.png')])

    img_name_list = os.listdir(origin_images_path)
    img_name_list.sort()
    random.shuffle(img_name_list)

    train_num = int(image_count * TRAIN_RATIO)
    test_num = image_count - train_num

    write_file('train', img_name_list[:-test_num])
    write_file('test', img_name_list[-test_num:])


# 定义训练和验证数据集
train_dataset = dataset.PetDataset(mode='train', image_size=IMAGE_SIZE)
test_dataset = dataset.PetDataset(mode='test', image_size=IMAGE_SIZE)
print("images num train:{}, test:{}".format(len(train_dataset), len(test_dataset)))

# 定义模型
num_classes = 4
network = model.PetNet(num_classes)
model = paddle.Model(network)
model.summary((-1, 3,) + IMAGE_SIZE)

# 为模型训练做准备，设置优化器及其学习率，并将网络的参数传入优化器，设置损失函数和精度计算方式
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()), 
              loss=paddle.nn.CrossEntropyLoss(axis=1))

# 训练模型
model.fit(train_dataset, epochs=15, batch_size=16, verbose=1)

# 保存模型
model.save('./output/UNet')

# 加载模型
model.load('./output/UNet')

# 用 evaluate 在测试集上对模型进行验证
eval_result = model.evaluate(test_dataset, verbose=1)
print(eval_result)