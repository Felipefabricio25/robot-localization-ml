import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import random
from PIL import Image
import os

ia.seed(4)

for i in range(0,3420,5):
    filename = (i/10)

    if filename - int(filename) != 0.5:
        filename = int(filename)

    image = imageio.imread(f"/home/admsistemas/Documents/ReducedDatasetGMaps/{filename}/botanic_map_{filename}.png")

    print(f"Original ({filename}):")
    #ia.imshow(image)

    for i in range(6):

        rand = random.randrange(1,5)

        seq = iaa.SomeOf(rand,[
            iaa.Affine(rotate=(-90, 90), mode="edge"),
            iaa.AdditiveGaussianNoise(scale=(10, 60)),
            iaa.Crop(percent=(0, 0.2)),
            iaa.CropAndPad(percent=(-0.4, 0.4), pad_mode="edge"),
            iaa.ElasticTransformation(alpha=90, sigma=9),
            iaa.Cutout(),
            iaa.Dropout(p=(0, 0.2)),
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
            iaa.SaltAndPepper(0.1),
            iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1)),
            iaa.Salt(0.1),
            iaa.CoarseSalt(0.05, size_percent=(0.01, 0.1)),
            iaa.Pepper(0.1),
            iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1)),
            iaa.JpegCompression(compression=(70, 99)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(scale=(1.0, 3.5), mode="edge"),
            iaa.Affine(translate_percent={"x": (-0.4, 0.4), "y": (-0.4, 0.4)}, mode="edge"),
            iaa.Affine(shear=(-40, 40), mode="edge"),
            iaa.ScaleX((1.0, 3.5), mode="edge"),
            iaa.ScaleY((1.0, 3.5), mode="edge"),
            iaa.PiecewiseAffine(scale=(0.01, 0.05), mode="edge"),
            iaa.PerspectiveTransform(scale=(0.01, 0.15)),
            iaa.Rot90((1, 3)),
            iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.1, 0.1))),
            iaa.Jigsaw(nb_rows=10, nb_cols=10),
            iaa.Jigsaw(nb_rows=(1, 4), nb_cols=(1, 4)),
            iaa.imgcorruptlike.ShotNoise(severity=2),
            iaa.imgcorruptlike.ImpulseNoise(severity=2),
            iaa.imgcorruptlike.Spatter(severity=2),
        ], random_order=True)

        images_aug = seq(image=image)

        print(f"Augmented ({rand}):")
        #ia.imshow(images_aug)

        imageio.imwrite(f'/home/admsistemas/Documents/ReducedDatasetGMaps/{filename}/botanic_map_{filename}_{i}_{rand}.png', images_aug)


