import os


def stats_of_dataset():
    mother_folder = "D:/Repositories/Datasets/LeafSnap30/leaf/test" # change this to your mother-folder path
    filenames = os.listdir(mother_folder) # get all files and folders in the mother-folder
    result = [] # create an empty list to store the folder names and file counts

    for filename in filenames: # loop through all the items
        if os.path.isdir(os.path.join(mother_folder, filename)): # check if the item is a folder
            file_count = sum(len(files) for _, _, files in os.walk(os.path.join(mother_folder, filename))) # count the number of files in the folder and its subdirectories
            result.append((filename, file_count)) # add a tuple of (folder name, file count) to the result list

    result.sort(key=lambda x: x[1]) # sort the result list by file count using a lambda function as the key
    result.reverse()
    print(result) # print the result list

def list_folders(path):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
