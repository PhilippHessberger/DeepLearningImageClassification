import cv2
import os
import tensorflow as tf
import random


from image_augmenter import rotate_image_random, change_white_background, reshape_image
from image_filter import filter_by_white_background


def load_images_from_folder(folder_path, start_index=0, amount=-1):
    # containers
    images = []
    filenames = []

    # get all filenames:
    for filename in os.listdir(folder_path):
        filenames.append(filename)

    # if wanted amount is bigger than actual amount of files existing in the directory
    if amount > len(filenames) or amount == -1:
        amount = len(filenames)

    # remove file extension from filename
    filenames = sorted(filenames, key=lambda x: int(os.path.splitext(x)[0]))

    # only load the files we want (from start_index to start_index + amount)
    for filename in filenames[start_index:start_index + amount]:
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            images.append(img)

    return images, filenames[start_index:start_index + amount]

# displays the given images, space to go to next image, ESC to abort
def display_images(images):
    for img in images:
        cv2.namedWindow('image')
        cv2.moveWindow('image', 50, 50)
        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit display mode
            cv2.destroyAllWindows()
            break
        elif k == 32:       # wait for space bar to show next image
            continue
    cv2.destroyAllWindows()

# displays just one image until a any key is pressed
def display_image(img, window_title=None):
    if window_title is not None:
        cv2.namedWindow(window_title)
        cv2.moveWindow(window_title, 50, 50)
        cv2.imshow(window_title, img)
    else:
        cv2.namedWindow('image')
        cv2.moveWindow('image', 50, 50)
        cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# used to load a dataset into a dict. structure is like this: dict{'label' : (images_for_label, filenames_for_images)}
def load_dataset(filepath, label_names, image_shape=(299, 299), multiplier=1, limit=None, use_originals=False, filter_white_background=False, augment_filtered_dataset=False, backgrounds_foldername=None, only_resize_and_rotate=False):
    if only_resize_and_rotate:
        augment_filtered_dataset = False
        filter_white_background = False

    # dict of 'latin name' : (images, filenames)
    dataset = {}

    # load every image for each wanted label
    for label_name in label_names:
        print(f'now loading: {filepath}/{label_name}')
        images, filenames = load_images_from_folder(f'{filepath}/{label_name}')

        # if the background should be filled with a random image instead of a random color
        if backgrounds_foldername is not None:
            background_images, background_filenames = load_images_from_folder('D:/Repositories/Datasets/PlantCLEF/our_selected_dataset/random_backgrounds') # f'{filepath}/{backgrounds_foldername}')

        # filter for white background if wanted
        if images is not None and filenames is not None:

            if filter_white_background:
                filtered_images, filenames = filter_by_white_background(images, filenames)

                # rotate image and change background color if wanted
                if augment_filtered_dataset:

                    # calculate how to reach limit if wanted
                    if limit is not None:
                        amount_of_images = len(filtered_images)
                        amount_of_images_to_create = limit - amount_of_images
                        multiplier = amount_of_images_to_create // amount_of_images
                        rest = amount_of_images_to_create % amount_of_images
                    else:
                        rest = 0

                    augmented_filtered_images = []
                    filenames_for_augmented_filtered_images = []

                    for i, image in enumerate(filtered_images):

                        if rest > 0:
                            actual_multiplier = multiplier + 1
                            rest -= 1
                        else:
                            actual_multiplier = multiplier

                        if use_originals:
                            augmented_filtered_images.append(reshape_image(image, image_shape))
                            augmented_filename = filenames[i].split('.')
                            augmented_filename = str(augmented_filename[0]) + '_original.' + augmented_filename[
                                1]
                            filenames_for_augmented_filtered_images.append(augmented_filename)

                        for j in range(actual_multiplier):
                            # changes the images dimensions (given shape) and rotates the image
                            augmented_image = rotate_image_random(image, image_shape)

                            # select random image as background
                            """
                            if backgrounds_foldername is not None:
                                background_image = random.choice(background_images)
                            else:
                                background_image = None
                            """
                            # changes the background color of the image
                            augmented_image = change_white_background(augmented_image) # , background_image=background_image, shape=image_shape)

                            augmented_filtered_images.append(augmented_image)

                            # different image names are needed for every image, but we still want to know where they were derived from
                            augmented_filename = filenames[i].split('.')
                            augmented_filename = str(augmented_filename[0]) + '_v' + str(j) + '.' + augmented_filename[1]
                            filenames_for_augmented_filtered_images.append(augmented_filename)

                    dataset[label_name] = (augmented_filtered_images, filenames_for_augmented_filtered_images)

                else:
                    dataset[label_name] = (filtered_images, filenames)

            elif only_resize_and_rotate:
                # calculate amount of images to generate until limit per label is reached
                if limit is not None:
                    amount_of_images = len(images)
                    amount_of_images_to_create = limit - amount_of_images
                    multiplier = amount_of_images_to_create // amount_of_images
                    rest = amount_of_images_to_create % amount_of_images
                else:
                    rest = 0

                augmented_images = []
                filenames_for_augmented_images = []

                for i, image in enumerate(images):

                    if rest > 0:
                        actual_multiplier = multiplier + 1
                        rest -= 1
                    else:
                        actual_multiplier = multiplier

                    if use_originals:
                        augmented_images.append(reshape_image(image, image_shape))
                        augmented_filename = filenames[i].split('.')
                        augmented_filename = str(augmented_filename[0]) + '_original.' + augmented_filename[1]
                        filenames_for_augmented_images.append(augmented_filename)

                    for j in range(actual_multiplier):
                        # changes the images dimensions (given shape) and rotates the image
                        augmented_image = rotate_image_random(image, image_shape)

                        augmented_images.append(augmented_image)

                        # different image names are needed for every image, but we still want to know where they were derived from
                        augmented_filename = filenames[i].split('.')
                        augmented_filename = str(augmented_filename[0]) + '_v' + str(j) + '.' + augmented_filename[1]
                        filenames_for_augmented_images.append(augmented_filename)

                dataset[label_name] = (augmented_images, filenames_for_augmented_images)

            else:
                dataset[label_name] = (images, filenames)
        else:
            print('ERROR: length of list of images and length of list of images filenames are different')

    return dataset

# converts our dataset dict to two lists of images and filenames
def convert_dataset_to_arrays(old_dataset):
    images = []
    labels = []

    for label in old_dataset:
        for i in range(len(old_dataset[label][0])):
            try:
                images.append(old_dataset[label][0][i])
                labels.append(label)
            except TypeError:
                print(f"WARNING: label '{label}' has no images in the dataset")

    return images, labels

def augment_and_save_dataset(filepath_to_data, directory_names, dataset_name, limit, random_backgrounds=None, only_resize_and_rotate=False):
    filepath_to_data = filepath_to_data + dataset_name

    my_dataset = load_dataset(filepath_to_data,
                              directory_names,
                              image_shape=(224, 224),
                              # image_shape=(299, 299),
                              use_originals=True,
                              limit=limit,
                              filter_white_background=True,
                              augment_filtered_dataset=True,
                              only_resize_and_rotate=only_resize_and_rotate,
                              backgrounds_foldername=random_backgrounds)
    save_dataset(my_dataset, folder_name=dataset_name)

def load_tensorflow_dataset_from_folder(filepath=None, folder_name=None):
    # catch unwanted behavior
    if filepath is not None and folder_name is not None:
        raise Exception('Set only one of the two parameters')
    elif filepath is not None:
        filepath_to_data = filepath
    elif folder_name is not None:
        filepath_to_data = os.path.join(os.getcwd(), 'data', folder_name)
    else:
        raise Exception('Set one of the two parameters')

    # load the training dataset
    training_dataset = tf.keras.utils.image_dataset_from_directory(filepath_to_data, labels='inferred')

    # convert labels to one-hot encoded vectors
    # TODO: depth=<amount of labels>
    training_dataset = training_dataset.map(lambda x, y: (x, tf.one_hot(y, depth=10)))

    return training_dataset

def save_dataset(data, folder_name):
    # get path to 'training_data_big' folder
    project_dir = os.getcwd()

    # create data folder
    if not os.path.exists('data'):
        os.mkdir('data')

    # change current dir to 'data' folder
    os.chdir(os.path.join(project_dir, 'data'))

    # create folder
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # change current dir to <folder_name> folder
    os.chdir(os.path.join(project_dir, 'data', folder_name))

    # iterate over classes of our training data
    for label in data.keys():

        # create folder for class if it did not exist already
        if not os.path.exists(os.path.join(label)):
            os.mkdir(label)

        # iterate over images and filenames of these images
        for i in range(len(data[label][0])):
            # save the image in it using the filename of the image
            cv2.imwrite(os.path.join(label, data[label][1][i]), data[label][0][i])
