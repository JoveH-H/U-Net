import paddle
import dataset
import model

IMAGE_SIZE = (160, 160)  # 图片输入大小

# 原图和标签图片地址
resources_path = "./resources/Oxford-IIIT Pet"

# 定义训练和验证数据集
test_dataset = dataset.PetDataset(mode='test', image_size=IMAGE_SIZE)
print("images num test:{}".format(len(test_dataset)))

# 定义模型
num_classes = 4
network = model.PetNet(num_classes)
model = paddle.Model(network)
model.summary((-1, 3,) + IMAGE_SIZE)

# 加载模型
model.prepare(loss=paddle.nn.CrossEntropyLoss(axis=1))
model.load('./output/UNet')

# 在测试集上对模型进行验证
test_result = model.evaluate(test_dataset, verbose=1)
print(test_result)
