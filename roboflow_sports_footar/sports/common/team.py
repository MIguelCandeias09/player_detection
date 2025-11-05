from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
# import torch
# import umap
# from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids

# from sklearn.cluster import AgglomerativeClustering
# from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch

# from tqdm import tqdm
# from transformers import AutoProcessor, SiglipVisionModel
import matplotlib.pyplot as plt
# import colorsys
from skimage import color
import math
# from colorthief import ColorThief
# from sports.common import myColorThief
# import fast_colorthief

# V = TypeVar("V")

# SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'




DEBUG = False




# def create_batches(
#     sequence: Iterable[V], batch_size: int
# ) -> Generator[List[V], None, None]:
#     """
#     Generate batches from a sequence with a specified batch size.

#     Args:
#         sequence (Iterable[V]): The input sequence to be batched.
#         batch_size (int): The size of each batch.

#     Yields:
#         Generator[List[V], None, None]: A generator yielding batches of the input
#             sequence.
#     """
#     batch_size = max(batch_size, 1)
#     current_batch = []
#     for element in sequence:
#         if len(current_batch) == batch_size:
#             yield current_batch
#             current_batch = []
#         current_batch.append(element)
#     if current_batch:
#         yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 64):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
       """
        self.device = device
        self.batch_size = batch_size
        # self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        # self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        # self.reducer = umap.UMAP(n_components=5)
        # self.cluster_model = KMeans(n_clusters=2)
        # self.cluster_model = KMedoids(n_clusters=2)
        # self.cluster_model = MeanShift(n_clusters=2)
        # self.cluster_model = AgglomerativeClustering(n_clusters=2)  # doesn't have predict
        # self.cluster_model = SpectralClustering(n_clusters=2)  # doesn't have predict
        self.cluster_model = Birch(n_clusters=2, threshold=0.3, branching_factor=50)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:8
            np.ndarray: Extracted features as a numpy array.
        """
        # resizeX = 3
        # resizeY = 4

        # print('L0', len(crops))
        # print('L1', len(crops[0]))
        # print('L2', len(crops[0][0]))
        # print('L3', len(crops[0][0][0]))

        # for crop in crops:
        #     print(len(crop))

        # Use média de múltiplos pixels para mais robustez (não só 1 pixel)
        resizeX = 3
        resizeY = 4
            
        # if DEBUG:
            # print('original')
            # sv.plot_images_grid(images=crops, grid_size=(10, 10))

        # for crop in crops:
        #     for cropI in crop:
        #         for cropJ in cropI:
        #             cropJ[1] = 0    # remove green (bgr)
        
        
        if DEBUG:
            # print('no green')
            sv.plot_images_grid(images=crops, grid_size=(10, math.ceil(len(crops)/10)))   # debug 2025

        # subcrop the crops to get the midpart of the image
        tempCrops = []
        SUBCROP_PERCENT_X = 0.60
        SUBCROP_PERCENT_Y = 0.65
        for crop in crops:
            midWidth = len(crop[0]) / 2
            midHeight = len(crop) * 0.40
            x1 = max(round(midWidth - (midWidth * SUBCROP_PERCENT_X)), 0)
            y1 = max(round(midHeight - (midHeight * SUBCROP_PERCENT_Y)), 0)
            x2 = max(round(midWidth + (midWidth * SUBCROP_PERCENT_X)), 1)
            y2 = max(round(midHeight + (midHeight * SUBCROP_PERCENT_Y)), 1)

            # print('width', len(crop[0]))
            # print('height', len(crop))
            # print('midWidth', midWidth)
            # print('midHeight', midHeight)
            # print('x1y1x2y2', x1, y1, x2, y2)

            tempCrops.append(sv.crop_image(image=crop, xyxy=(x1, y1, x2, y2)))
        crops = tempCrops
        # end of subcrop the crops to get the midpart of the image

        # for crop in crops:
        #     print(len(crop))

        # crops = [sv.resize_image(crop, (64, 64)) for crop in crops]
        # print('crops',crops)
        
        if DEBUG:
            sv.plot_images_grid(images=crops, grid_size=(10, math.ceil(len(crops)/10)))   # debug 2025



        # mikeDominant_colorsIMG = [sv.resize_image(crop, (resizeX, resizeY)) for crop in crops]
        crops = [sv.resize_image(crop, (resizeX, resizeY)) for crop in crops]

        # fastThiefDominant_colors = []
        # for crop in crops:
        #     cropRGBA = np.ndarray((len(crop), len(crop[0]), 4), dtype=np.uint8)
        #     for i in range(len(cropRGBA)):
        #         for j in range(len(cropRGBA[i])):
        #             cropRGBA[i][j] = np.array([crop[i][j][0], crop[i][j][1], crop[i][j][2], 255], dtype=np.uint8)

            # cropRGBA = cropRGBA[np.newaxis]
            
            # cropRGBA = np.c_[cropRGBA, np.zeros(len(cropRGBA))]

            # for i in range(len(cropRGBA)):
                # cropRGBA[i] = np.c_[cropRGBA[i], np.zeros(len(cropRGBA[i]))]
                # cropRGBA[i] = np.append(cropRGBA[i])
                # np.concatenate(cropRGBA[i], axis=1)
                    # print('cropI[j]', cropI[j])
                    # print('np.append(cropI[j], 0)', np.append(cropI[j], 0))
                    # cropI[j] = np.append(cropI[j], 0)
            
            # fastThiefDominant_colors.append(ColorThief.get_dominant_color(image=cropRGBA, quality=1, use_gpu=False))
            # print('cropRGBA',cropRGBA)
            # fastThiefDominant_colors.append(fast_colorthief.get_dominant_color(cropRGBA, quality=1, use_gpu=True))

        # print('dominant_color', fastThiefDominant_colors)

        # fastThiefDominant_colorsIMG = [0] * len(crops)

        # print('fastThiefDominant_colorsIMG before attribution', fastThiefDominant_colorsIMG)
        # print('len fastThiefDominant_colorsIMG', len(fastThiefDominant_colorsIMG))

        # for dominant_color in fastThiefDominant_colorsIMG:
        # for cropI in range(len(crops)):
        #     fastThiefDominant_colorsIMG[cropI] = np.ndarray((1, 1, 3), dtype=np.uint8)
        #     for i in range(len(fastThiefDominant_colorsIMG[cropI])):
        #         for j in range(len(fastThiefDominant_colorsIMG[cropI][i])):
        #             fastThiefDominant_colorsIMG[cropI][i][j] = np.array([fastThiefDominant_colors[cropI][0], fastThiefDominant_colors[cropI][1], fastThiefDominant_colors[cropI][2]], dtype=np.uint8)


        # print('fastThiefDominant_colorsIMG', fastThiefDominant_colorsIMG)

        # crops = [np.reshape(crop, (2, 6144)) for crop in crops]
        # crops = [np.reshape(crop, (12288, 2)) for crop in crops]
        # crops = [np.reshape(crop, (12288)) for crop in crops]

        pixelToSelect = [max(math.floor(resizeY / 2) - 1, 0), math.floor(resizeX / 2)]
        # pixelToSelect = [1, 1]  # Linha x Coluna

        # print('pixelToSelect',pixelToSelect)

        # print('L0', len(crops))
        # print('L1', len(crops[0]))
        # print('L2', len(crops[0][0]))
        # print('L3', len(crops[0][0][0]))


        # for crop in crops:
            # print('crop[0][0][0]', crop[0][0][0])
            # print('crop[0][0][1]', crop[0][0][1])
            # print('crop[0][0][2]', crop[0][0][2])
            # crop = [crop[0][0][0], crop[0][0][1] * crop[0][0][2]]

        # print('print crops')
        # sv.plot_images_grid(images=crops, grid_size=(10, 10))
        # print('')

        # print('print fastThiefDominant_colorsIMG')
        # sv.plot_images_grid(images=fastThiefDominant_colorsIMG, grid_size=(10, 10))

        # print('print mikeDominant_colorsIMG')
        if DEBUG:
            # sv.plot_images_grid(images=mikeDominant_colorsIMG, grid_size=(10, 10))
            sv.plot_images_grid(images=crops, grid_size=(10, math.ceil(len(crops)/10)))   # debug 2025

        mikeDominant_colorsLAB = []

        # print('crops', crops)
        # Usar MÉDIA de todos os pixels para mais robustez (não apenas 1 pixel central)
        for i in range(len(crops)):
            crop_rgb = crops[i] / 255.0  # Normaliza para [0, 1]
            # Converte todos os pixels BGR -> RGB -> LAB e faz média
            crop_lab = color.rgb2lab(crop_rgb[..., ::-1])  # ::-1 inverte BGR para RGB
            # Média de todos os pixels em cada canal LAB
            l_mean = np.mean(crop_lab[:, :, 0])
            a_mean = np.mean(crop_lab[:, :, 1])
            b_mean = np.mean(crop_lab[:, :, 2])
            mikeDominant_colorsLAB.append([l_mean, a_mean, b_mean])

        # fastThiefDominant_colorsLAB = []

        # for i in range(len(fastThiefDominant_colors)):
        #     r = int(fastThiefDominant_colors[i][0])    # stored in RGB
        #     g = int(fastThiefDominant_colors[i][1])    # stored in RGB
        #     b = int(fastThiefDominant_colors[i][2])    # stored in RGB
        #     l,a,b = color.rgb2lab([r / 255, g / 255, b / 255])
        #     fastThiefDominant_colorsLAB.append([l, a, b])
        # print('crops2', crops)

        # crops = [np.array([int(crop[0][0][1]) / 255, int(crop[0][0][0]) * int(crop[0][0][2])]) for crop in mikeDominant_colorsLAB]



        # PLOTTING ######################################
        # fig = plt.figure()

        # # 3D FIGURE
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(xs=[crop[0] for crop in mikeDominant_colorsLAB], ys=[crop[1] for crop in mikeDominant_colorsLAB], zs=[crop[2] for crop in mikeDominant_colorsLAB])
        # plt.title('Team kit colors')


        # # plt.scatter(x=[crop[0] for crop in mikeDominant_colorsLAB], y=[0 for crop in mikeDominant_colorsLAB])
        # plt.title('Team kit colors')
        # plt.xlabel('X - HUE')


        # plt.show()
        # END OF PLOTTING ######################################


        # print('_L1', len(mikeDominant_colorsLAB[0]))

        # print('mikeDominant_colorsLAB', mikeDominant_colorsLAB) 


        
        # mikeDominant_colorsLAB = [sv.cv2_to_pillow(crop) for crop in crops]
        # batches = create_batches(crops, self.batch_size)
        # data = []
        # with torch.no_grad():
        #     for batch in tqdm(batches, desc='Embedding extraction'):
        #         inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
        #         outputs = self.features_model(**inputs)
        #         embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        #         # embeddings = torch.median(outputs.last_hidden_state).cpu().numpy()
        #         # embeddings = torch.prod(outputs.last_hidden_state, dim=1).cpu().numpy()
        #         data.append(embeddings)
        # return np.concatenate(data)

        # return np.concatenate(crops)
        return mikeDominant_colorsLAB
        # return fastThiefDominant_colorsLAB

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        print('fit called')
        data = self.extract_features(crops)
        # projections = self.reducer.fit_transform(data)
        projections = data
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        # projections = self.reducer.transform(data)
        projections = data
        preditcitons = self.cluster_model.predict(projections)

        # print('preditcitons', preditcitons)




        if DEBUG:
            # 3D FIGURE
            fig = plt.figure()
            xData1 = []
            xData2 = []
            yData1 = []
            yData2 = []
            zData1 = []
            zData2 = []

            # # draw centroids (not for Birch)
            # cluster_centers = self.cluster_model.subcluster_centers_
            # xCentroid1 = cluster_centers[0][0]
            # xCentroid2 = cluster_centers[1][0]
            # yCentroid1 = cluster_centers[0][1]
            # yCentroid2 = cluster_centers[1][1]
            # zCentroid1 = cluster_centers[0][2]
            # zCentroid2 = cluster_centers[1][2]
            c1 = []
            c2 = []

            for i in range(len(projections)):
                if preditcitons[i] == 0:
                    xData1.append(projections[i][0])
                    yData1.append(projections[i][1])
                    zData1.append(projections[i][2])
                    # c1.append(colorsys.hsv_to_rgb(projections[i][0], projections[i][1], projections[i][2]))
                    # c1.append([projections[i][0] / 255, projections[i][1] / 255, projections[i][2] / 255])
                    c1.append(color.lab2rgb([projections[i][0], projections[i][1], projections[i][2]]))
                else:
                    xData2.append(projections[i][0])
                    yData2.append(projections[i][1])
                    zData2.append(projections[i][2])
                    # c2.append(colorsys.hsv_to_rgb(projections[i][0], projections[i][1], projections[i][2]))
                    # c2.append([projections[i][0] / 255, projections[i][1] / 255, projections[i][2] / 255])
                    c2.append(color.lab2rgb([projections[i][0], projections[i][1], projections[i][2]]))

            ax = fig.add_subplot(projection='3d')
            ax.scatter(xs=[item for item in xData1], ys=[item for item in yData1], zs=[item for item in zData1], marker='*', c=c1, s=50)
            ax.scatter(xs=[item for item in xData2], ys=[item for item in yData2], zs=[item for item in zData2], marker='v', c=c2, s=50)

            # # draw centroids (not for Birch)
            # ax.scatter(xs=[xCentroid1], ys=[yCentroid1], zs=[zCentroid1], marker='*', c=[color.lab2rgb([xCentroid1, yCentroid1, zCentroid1])], s=300)
            # ax.scatter(xs=[xCentroid2], ys=[yCentroid2], zs=[zCentroid2], marker='v', c=[color.lab2rgb([xCentroid2, yCentroid2, zCentroid2])], s=300)

            plt.title('Team kit colors')
            ax.set_xlabel('L')
            ax.set_ylabel('A')
            ax.set_zlabel('B')

            plt.show()

        return preditcitons
