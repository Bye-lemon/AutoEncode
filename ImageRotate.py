from PIL import Image

def rotate(angle):
    rootPath = './testImage.png'

    with Image.open(rootPath) as image:
        # 图像左右翻转
        out = image.rotate(angle)
        # 图像存储
        out.save('out.png', quality=100)

rotate(90)

