import numpy as np

class AugmentationGenerator:
    def __init__(self, im_gen, mask_gen, aug):
        self.im_gen, self.mask_gen, self.aug = im_gen, mask_gen, aug

    def __iter__(self):
        while True:
            nxt_img = self.im_gen.next()
            nxt_mask = self.mask_gen.next()
            img = []
            mask = []
            for (x, y) in zip(nxt_img, nxt_mask):
                augmented = self.aug(image=x, mask=y)
                img.append(augmented["image"])
                mask.append(augmented["mask"])
            img = np.stack(img, axis=0)
            mask = np.stack(mask, axis=0)
            # print(img.shape, mask.shape)
            yield img, mask

# example code:
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# image_datagen = ImageDataGenerator()
# mask_datagen = ImageDataGenerator()
# seed = 1
# image_generator = image_datagen.flow(X_train, batch_size=8, seed=seed)
# mask_generator = mask_datagen.flow(y_train, batch_size=8, seed=seed)
# aug = Compose([
#     VerticalFlip(p=0.5),
#     HorizontalFlip(p=0.5),
#     RandomRotate90(p=0.5),
# ])
# data_gen = AugmentationGenerator(image_generator, mask_generator, aug)
