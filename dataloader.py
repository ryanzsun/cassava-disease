import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
from aug import *
class CassavaDataset(Dataset):
    def __init__(self, transform, mode, image_size = 224, base_url = "/home/ryan/Documents/code/Playground/cassava-data"):
        self.transform = transform
        self.mode = mode
        self.image_size = image_size

        self.base_url = base_url
        self.set_mode(mode)

    def set_mode(self, mode, fold_index = 0):
        '''
        mode 1: "train" train with 80% labeled data
        mode 2: "full_train" train with 80% labeled data and unlabeld data
        mode 3: "validate" valdiate with 20% labeled data
        mode 4: "test" test with unlabeled data
        '''
        self.mode = mode

        if self.mode == 'train':
            image_list = glob.glob(self.base_url+"/train/*.jpg")

        elif self.mode == 'val':
            pass

        elif self.mode == 'test':
            pass

    def __getitem__(self, index):
        if self.mode == 'train':
            if self.is_non_empty:
                img_tmp = self.train_image_with_mask[index]
                image = cv2.imread(os.path.join(self.train_image_path, img_tmp), 1)
                label = cv2.imread(os.path.join(self.train_mask_path, img_tmp + '.png'), 1)

            else:
                if random.randint(0,1) == 0:
                    random_pos = random.randint(0, len(self.train_image_no_mask)-1)
                    img_tmp = self.train_image_no_mask[random_pos]

                    image = cv2.imread(os.path.join(self.train_image_path, img_tmp),1)
                    label = cv2.imread(os.path.join(self.train_mask_path, img_tmp + '.png'),1)
                else:
                    random_pos = random.randint(0, len(self.train_image_with_mask) - 1)
                    img_tmp = self.train_image_with_mask[random_pos]

                    image = cv2.imread(os.path.join(self.train_image_path, img_tmp), 1)
                    label = cv2.imread(os.path.join(self.train_mask_path, img_tmp + '.png'), 1)

        if self.mode == 'val':
            pass

        if self.mode == 'test':
            pass

        if self.mode == 'train':
            if random.randint(0, 1) == 0:
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)

            if random.randint(0, 1) == 0:
                image = cv2.flip(image, 0)
                label = cv2.flip(label, 0)

            if random.randint(0, 1) == 0:
                image = cv2.transpose(image)
                label = cv2.transpose(label)

            if random.randint(0, 1) == 0:
                image = randomHueSaturationValue(image,
                                               hue_shift_limit=(-30, 30),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))

                image, label = randomShiftScaleRotate(image, label,
                                                   shift_limit=(-0.1, 0.1),
                                                   scale_limit=(-0.1, 0.1),
                                                   aspect_limit=(-0.1, 0.1),
                                                   rotate_limit=(-0, 0))

        image = randomCrop(image)
        image = cv2.resize(image,(self.image_size, self.image_size))

        image = image.reshape([self.image_size, self.image_size, 3])

        image = np.transpose(image, (2,0,1))
        image = image.reshape([3, self.image_size, self.image_size])

        image = (np.asarray(image).astype(np.float32) - 127.5) / 127.5



        return torch.FloatTensor(image), torch.FloatTensor(label), is_empty

    def __len__(self):
        return len(self.image_list)



def get_loader(image_size, batch_size, mode='train'):
    """Build and return data loader."""
    dataset = SaltDataset5Fold(None, mode, image_size)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def get_5foldloader(image_size, batch_size, fold_index, mode='train', is_non_empty = False):
    """Build and return data loader."""
    dataset = SaltDataset5Fold(None, mode, image_size, fold_index, is_non_empty)

    shuffle = False
    if mode == 'train':
        shuffle = True

    if mode == 'test':
        shuffle = True
        print('test shuffle!!!!!!!!!')

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=32, shuffle=shuffle)
    return data_loader

def split_train_val():
    train_image_path = r'/data2/shentao/DATA/Kaggle/Salt/Kaggle_salt/train/images'

    image_list = os.listdir(train_image_path)
    random.shuffle(image_list)
    train_num = int(0.8*len(image_list))

    train_image_list = image_list[0: train_num]
    val_image_list = image_list[train_num:]


    print(len(train_image_list))
    print(len(val_image_list))

    f = open('train.txt','w')
    for image in train_image_list:
        f.write(image + '\n')
    f.close()

    f = open('val.txt','w')
    for image in val_image_list:
        f.write(image + '\n')
    f.close()

    print(len(image_list))
    return


def create_test_list():
    test_image_path = r'/data/shentao/Airbus/AirbusShipDetectionChallenge_384/test'

    image_list = os.listdir(test_image_path)
    random.shuffle(image_list)

    f = open('./image_list/test.txt','w')
    for image in image_list:
        f.write(image + '\n')
    f.close()

    print(len(image_list))