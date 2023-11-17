from keras.preprocessing.image import ImageDataGenerator

# normalize [0,1]
def dataGeneration(train_dir, valid_dir):

    train_datagen = ImageDataGenerator( rescale=1.0/255 )
    valid_datagen = ImageDataGenerator( rescale=1.0/255 )

    # generate 20 batches 150x150
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

    validation_generator = valid_datagen.flow_from_directory(valid_dir,
                                                             batch_size=20,
                                                             class_mode='binary',
                                                             target_size=(150,150))

    return train_generator, validation_generator