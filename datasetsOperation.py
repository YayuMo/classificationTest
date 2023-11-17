import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def dataOperation(base_dir):

    # data split
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    return train_dir, validation_dir

def showImgs(train_cats_dir, train_dogs_dir, train_cat_fnames, train_dog_fnames):
    nrows = 4
    ncols = 4
    pic_index = 0

    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)

    pic_index += 8
    next_cat_pix = [os.path.join(train_cats_dir, fname) for fname in train_cat_fnames[ pic_index-8:pic_index ]]
    next_dog_pix = [os.path.join(train_dogs_dir, fname) for fname in train_dog_fnames[ pic_index-8:pic_index ]]

    for i, img_path in enumerate(next_cat_pix+next_dog_pix):
        sp = plt.subplot(nrows, ncols, i+1)
        sp.axis('Off')
        img = mpimg.imread(img_path)
        plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    base_dir = 'catdogdata/cats_and_dogs_filtered'
    train_dir, validation_dir = dataOperation(base_dir)
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)

    # print first 10th images
    print(train_cat_fnames[:10])
    print(train_dog_fnames[:10])

    # The length of datasets
    print('total training cat images', len(os.listdir(train_cats_dir)))
    print('total training dog images', len(os.listdir(train_dogs_dir)))
    print('total validation cat images', len(os.listdir(validation_cats_dir)))
    print('total validation dog images', len(os.listdir(validation_dogs_dir)))
    showImgs(train_cats_dir, train_dogs_dir, train_cat_fnames, train_dog_fnames)