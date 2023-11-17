import tensorflow as tf
import zipfile

if __name__ == '__main__':
    path=tf.keras.utils.get_file('cats_and_dogs_filtered.zip',origin='https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip')
    print(path)#会输出其路径，和当前notebook文件路径相同
    zip_ref = zipfile.ZipFile(path, 'r')
    zip_ref.extractall('/home/arthur/PycharmProjects/classificationTest/catdogdata')
    zip_ref.close()