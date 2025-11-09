import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  BraTS2021_00002_seg.nii.gz
                  where the last part before extension is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                # Filter out non-.nii/.nii.gz files
                files = [f for f in files if f.endswith('.nii.gz') or f.endswith('.nii')]
                
                if len(files) == 0:
                    continue
                    
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    # Fixed: changed from index 3 to index 2
                    try:
                        seqtype = f.split('_')[2].split('.')[0]
                        datapoint[seqtype] = os.path.join(root, f)
                    except IndexError:
                        print(f"Warning: Cannot parse filename {f}, skipping...")
                        continue
                
                # Debug: Print what we found
                if len(datapoint) > 0 and set(datapoint.keys()) != self.seqtypes_set:
                    print(f"Warning: Folder {root} has incomplete data")
                    print(f"  Expected: {self.seqtypes_set}")
                    print(f"  Found: {set(datapoint.keys())}")
                    print(f"  Files: {files}")
                    continue
                    
                if set(datapoint.keys()) == self.seqtypes_set:
                    self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path)

    def __len__(self):
        return len(self.database)

class BRATSDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  BraTS2021_00002_seg.nii.gz
                  where the last part before extension is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                # Filter out non-.nii.gz files
                files = [f for f in files if f.endswith('.nii.gz')]
                
                if len(files) == 0:
                    continue
                    
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    # Fixed: changed from index 3 to index 2
                    try:
                        seqtype = f.split('_')[2].split('.')[0]
                        datapoint[seqtype] = os.path.join(root, f)
                    except IndexError:
                        print(f"Warning: Cannot parse filename {f}, skipping...")
                        continue
                
                # Debug: Print what we found
                if len(datapoint) > 0 and set(datapoint.keys()) != self.seqtypes_set:
                    print(f"Warning: Folder {root} has incomplete data")
                    print(f"  Expected: {self.seqtypes_set}")
                    print(f"  Found: {set(datapoint.keys())}")
                    print(f"  Files: {files}")
                    continue
                    
                if set(datapoint.keys()) == self.seqtypes_set:
                    self.database.append(datapoint)
        return len(self.database) * 155

    def __getitem__(self, x):
        out = []
        n = x // 155
        slice = x % 155
        filedict = self.database[n]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            o = torch.tensor(nib_img.get_fdata())[:,:,slice]
            # if seqtype != 'seg':
            #     o = o / o.max()
            out.append(o)
        out = torch.stack(out)
        if self.test_flag:
            image=out
            # image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            # image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            # label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path
        
