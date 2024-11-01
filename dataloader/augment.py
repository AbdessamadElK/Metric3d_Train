import numpy as np
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Augmentor:
    def __init__(self, crop_size, spatial_aug = False, min_scale=-0.2, max_scale=0.4, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug = spatial_aug
        self.spatial_aug_prob = 0.8
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
    
    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=0.5]
        flow0 = flow[valid>=0.5]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img
    
    def spatial_transform(self, voxel1, voxel2, flow, valid, img, depth, normal):
        ht, wd = voxel2.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)
        
        if self.spatial_aug and  (np.random.rand() < self.spatial_aug_prob):
            # rescale the images
            voxel1 = cv2.resize(voxel1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            voxel2 = cv2.resize(voxel2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)
            # print('Resized:', voxel1.shape, voxel2.shape, flow.shape, valid.shape)

            img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            normal = cv2.resize(normal, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

        if self.crop_size is not None:
            margin_y = int(round(65 * scale_y))#downside
            margin_x = int(round(35 * scale_x))#leftside

            y0 = np.random.randint(0, voxel2.shape[0] - self.crop_size[0] - margin_y)
            x0 = np.random.randint(margin_x, voxel2.shape[1] - self.crop_size[1])

            y0 = np.clip(y0, 0, voxel2.shape[0] - self.crop_size[0])
            x0 = np.clip(x0, 0, voxel2.shape[1] - self.crop_size[1])
            
            voxel1 = voxel1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            voxel2 = voxel2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

            img = img[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

            depth = depth[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            normal = normal[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                voxel1 = voxel1[:, ::-1]
                voxel2 = voxel2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

                img = img[:, ::-1]

                depth = depth[:, ::-1]
                normal = normal[:, ::-1]


            if np.random.rand() < self.v_flip_prob: # v-flip
                voxel1 = voxel1[::-1, :]
                voxel2 = voxel2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                valid = valid[::-1, :]

                img = img[::-1, :]

                depth = depth[::-1, :]
                normal = normal[::-1, :]

        return voxel1, voxel2, flow, valid, img, depth, normal
    
    def __call__(self, voxel1, voxel2, flow, valid, img, depth, normal):
        voxel1, voxel2, flow, valid, img, depth, normal = self.spatial_transform(voxel1, voxel2, flow, valid, img, depth, normal)
        voxel1 = np.ascontiguousarray(voxel1)
        voxel2 = np.ascontiguousarray(voxel2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)  
        img = np.ascontiguousarray(img)
        depth = np.ascontiguousarray(depth)
        normal = np.ascontiguousarray(normal)
        
        return voxel1, voxel2, flow, valid, img, depth, normal   
                
                        