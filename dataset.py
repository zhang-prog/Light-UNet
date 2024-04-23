import cv2
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):
		image_path = self.image_name_list[idx]
		label_path = self.label_name_list[idx]

		image = cv2.imread(image_path)
		mask = []
		for i in range(1):
			try:
				mask.append(cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)[..., None])
			except:
				print(label_path)

		mask = np.dstack(mask)

		imidx = np.array([idx])

		if self.transform is not None:
			augmented = self.transform(image=image, mask=mask)
			image = augmented['image']
			mask = augmented['mask']

		image = image.astype('float32') / 255
		image = image.transpose(2, 0, 1)
		mask = mask.astype('float32') / 255
		mask = mask.transpose(2, 0, 1)

		data = {'imidx': imidx, 'image': image, 'label': mask}
		path = {'image_path': image_path, 'label_path': label_path}

		return {**data, **path}


class GlaSDataset():
	pass