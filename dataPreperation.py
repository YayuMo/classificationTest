import os
import shutil
from keras.preprocessing.image import ImageDataGenerator

# Array Split by ratio
def arrSplit(arr, ratio):
    print(int(ratio*len(arr)))
    arr1 = arr[:int(ratio*len(arr))]
    arr2 = arr[int(ratio*len(arr)):len(arr)]
    return arr1, arr2

# file Copy
def fileMatching(base_dir,target_dir, arrFileName, type, obj):
    src_path = os.path.join(base_dir, type)
    target_path = os.path.join(target_dir,obj+'/'+type)
    for filename in arrFileName:
        src_path_file = os.path.join(src_path, filename)
        target_path_file = os.path.join(target_path, filename)
        shutil.copy(src_path_file, target_path_file)

def dataPre(base_dir, target_dir):
    # get original data path
    defocused_path = os.path.join(base_dir, 'defocused_blurred')
    motionblur_path = os.path.join(base_dir, 'motion_blurred')
    sharp_path = os.path.join(base_dir, 'sharp')

    # list file names
    defocused_fnames = os.listdir(defocused_path)
    motionblur_fnames = os.listdir(motionblur_path)
    sharp_fnames = os.listdir(sharp_path)

    # train set and validation set split
    split_ratio = 0.7 # train:valid = 7:3
    defocused_train, defocused_valid = arrSplit(defocused_fnames, split_ratio)
    motionblur_train, motionblur_valid = arrSplit(motionblur_fnames, split_ratio)
    sharp_train, sharp_valid = arrSplit(sharp_fnames, split_ratio)

    # copy file into new path
    fileMatching(base_dir, target_dir, defocused_train, 'defocused_blurred', 'train')
    fileMatching(base_dir, target_dir, defocused_valid, 'defocused_blurred', 'validation')
    fileMatching(base_dir, target_dir, motionblur_train, 'motion_blurred', 'train')
    fileMatching(base_dir, target_dir, motionblur_valid, 'motion_blurred', 'validation')
    fileMatching(base_dir, target_dir, sharp_train, 'sharp', 'train')
    fileMatching(base_dir, target_dir, sharp_valid, 'sharp', 'validation')

def dataGenerator(train_dir, valid_dir):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
    class_names = ['defocused_blurred', 'motion_blurred', 'sharp']
    # generate 20 batches 150x150
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=20,
                                                        class_mode="categorical",
                                                        target_size=(32, 32))

    validation_generator = valid_datagen.flow_from_directory(valid_dir,
                                                             batch_size=20,
                                                             class_mode='categorical',
                                                             target_size=(32, 32))

    return train_generator, validation_generator

if __name__ == '__main__':
    base_dir = 'archive'
    target_dir = 'dataset/blurredData'
    dataPre(base_dir,target_dir)