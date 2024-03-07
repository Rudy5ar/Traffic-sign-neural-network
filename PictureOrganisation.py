import os
import shutil
import random

def organiseFolders():
    os.chdir('data')
    print(os.listdir())

    for i in os.listdir('train/'):
        os.listdir('train/')
        insertValidTest(i, len(os.listdir(f'train/{i}')))

    os.chdir('../..')

def insertValidTest(folder_name, folder_size):
    os.mkdir(f'validate/{folder_name}')
    os.mkdir(f'test/{folder_name}')

    validate = random.sample(os.listdir(f'train/{folder_name}'),  int(folder_size * 20/100))
    for j in validate:
        shutil.move(f'train/{folder_name}/{j}', f'validate/{folder_name}')

    test = random.sample(os.listdir(f'train/{folder_name}'), int(folder_size * 10/100))
    for j in test:
        shutil.move(f'train/{folder_name}/{j}', f'test/{folder_name}')


def deletePictures():
    os.chdir('data')
    print(os.listdir())

    for i in os.listdir('train/'):
        os.listdir('train/')
        delete_part(i, len(os.listdir(f'train/{i}')))

    os.chdir('../..')

def delete_part(folder_name, folder_size):
    delete = random.sample(os.listdir(f'train/{folder_name}'), int(folder_size * 40/100))
    for j in delete:
        os.remove(f'train/{folder_name}/{j}')