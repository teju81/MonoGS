import cv2
import numpy as np
import math

class ExtractorNode:
    def __init__(self):
        self.keypoints_list = []
        self.UL = None
        self.UR = None
        self.BL = None
        self.BR = None
        self.NoMore = False

    def DivideNode(self):
        """ Subdivide the node into four child nodes. """
        halfX = (self.UR[0] - self.UL[0]) // 2
        halfY = (self.BL[1] - self.UL[1]) // 2

        n1 = ExtractorNode()
        n2 = ExtractorNode()
        n3 = ExtractorNode()
        n4 = ExtractorNode()

        # Define the four child nodes' corners
        n1.UL = self.UL
        n1.UR = (self.UL[0] + halfX, self.UL[1])
        n1.BL = (self.UL[0], self.UL[1] + halfY)
        n1.BR = (self.UL[0] + halfX, self.UL[1] + halfY)

        n2.UL = n1.UR
        n2.UR = self.UR
        n2.BL = n1.BR
        n2.BR = (self.UR[0], n1.BR[1])

        n3.UL = n1.BL
        n3.UR = n1.BR
        n3.BL = self.BL
        n3.BR = (n1.BR[0], self.BL[1])

        n4.UL = n3.UR
        n4.UR = self.BR
        n4.BL = n3.BR
        n4.BR = self.BR

        # Distribute keypoints among child nodes
        for kp in self.keypoints_list:
            if kp.pt[0] < n1.BR[0]:
                if kp.pt[1] < n1.BR[1]:
                    n1.keypoints_list.append(kp)
                else:
                    n3.keypoints_list.append(kp)
            else:
                if kp.pt[1] < n1.BR[1]:
                    n2.keypoints_list.append(kp)
                else:
                    n4.keypoints_list.append(kp)

        return [n1, n2, n3, n4]

class ORBExtractor:
    def __init__(self):
        self.iniThFAST = 20
        self.minThFAST = 7

        self.nlevels = 8
        self.nfeatures = 1000
        self.features_per_level = [0]*self.nlevels
        self.umax_list = []

        self.scaleFactor = 1.2
        self.scaleFactor_list = []
        self.invScaleFactor_list = []
        self.levelSigma2_list = []
        self.invLevelSigma2_list = []

        self.ImagePyramid = []

        self.PATCH_SIZE = 31
        self.HALF_PATCH_SIZE = 15
        self.EDGE_THRESHOLD = 19

        self.pattern = None
        self.keypoints = []
        self.descriptors = []
        self.init_params()

    def init_params(self):
        #self.generate_pattern()
        self.scaleFactor_list = [1.0]*self.nlevels
        self.scaleFactor_list[1:] = [self.scaleFactor_list[i-1]*self.scaleFactor for i in range(1,self.nlevels)]
        self.levelSigma2_list = [self.scaleFactor_list[i]**2 for i in range(self.nlevels)]
        self.invScaleFactor_list = [1/self.scaleFactor_list[i] for i in range(self.nlevels)]
        self.invLevelSigma2_list = [1/self.levelSigma2_list[i] for i in range(self.nlevels)]

        factor = 1.0 / self.scaleFactor
        desired_features_per_scale = self.nfeatures*(1-factor)/(1-factor**self.nlevels)


        sum_features = 0.0
        for level in range(self.nlevels-1):
            self.features_per_level[level] = round(desired_features_per_scale)
            sum_features += self.features_per_level[level]
            desired_features_per_scale *= factor

        self.features_per_level[-1] = max(self.nfeatures - sum_features, 0)

        npoints = 512
        bit_pattern_31_ = self.get_bit_pattern()
        pattern0 = np.reshape(bit_pattern_31_, (-1,2))
        self.pattern = []
        self.pattern.extend(pattern0[:npoints])

        self.HALF_PATCH_SIZE = 15  # Assuming some value; replace with actual size
        self.umax_list = [0] * (self.HALF_PATCH_SIZE + 1)  # Equivalent to umax.resize

        # Precompute the end of a row in a circular patch
        vmax = math.floor(self.HALF_PATCH_SIZE * math.sqrt(2.0) / 2 + 1)
        vmin = math.ceil(self.HALF_PATCH_SIZE * math.sqrt(2.0) / 2)
        hp2 = self.HALF_PATCH_SIZE**2

        for v in range(vmax + 1):
            self.umax_list[v] = round(math.sqrt(hp2 - v * v))

        # Ensure symmetry
        v0 = 0
        for v in range(self.HALF_PATCH_SIZE, vmin - 1, -1):
            while self.umax_list[v0] == self.umax_list[v0 + 1]:
                v0 += 1
            self.umax_list[v] = v0
            v0 += 1

        return

    def generate_pattern(self):
        """Generate the ORB pattern used for descriptor sampling (placeholder)."""
        return [(x, y) for x in range(-15, 16) for y in range(-15, 16)]  # Example pattern

    def ComputePyramid(self, image):
        image_np = image.permute(1,2,0).cpu().numpy()
        image_np = (image_np*255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        self.ImagePyramid = []
        for level in range(self.nlevels):
            scale = self.invScaleFactor_list[level]
            sz = (int(round(image_np.shape[1] * scale)), int(round(image_np.shape[0] * scale)))  # Width, Height (cols, rows)
            whole_size = (sz[0] + self.EDGE_THRESHOLD * 2, sz[1] + self.EDGE_THRESHOLD * 2)  # Add padding (whole size)

            # Create a temporary image with padding
            temp = np.zeros((whole_size[1], whole_size[0]), dtype=image_np.dtype)

            # Crop out the center image where there is no padding
            current_pyramid_level = temp[self.EDGE_THRESHOLD:self.EDGE_THRESHOLD + sz[1], self.EDGE_THRESHOLD:self.EDGE_THRESHOLD + sz[0]]
            
            # Resize the image for this pyramid level
            if level != 0:
                # Resize the previous level's image to the current size
                resized_image = cv2.resize(self.ImagePyramid[level - 1], sz, interpolation=cv2.INTER_LINEAR)
                current_pyramid_level[:, :] = resized_image  # Copy resized image to the pyramid level
            else:
                # The first level is just the original image
                current_pyramid_level[:, :] = cv2.resize(image_np, sz, interpolation=cv2.INTER_LINEAR)

            # Apply the border to the temp image
            if level == 0:
                # For the first level, we add padding to the original image
                temp_with_border = cv2.copyMakeBorder(image_np, self.EDGE_THRESHOLD, self.EDGE_THRESHOLD, self.EDGE_THRESHOLD, self.EDGE_THRESHOLD,
                                                      cv2.BORDER_REFLECT_101)
            else:
                # For other levels, we add padding to the resized image
                temp_with_border = cv2.copyMakeBorder(current_pyramid_level, self.EDGE_THRESHOLD, self.EDGE_THRESHOLD, self.EDGE_THRESHOLD, self.EDGE_THRESHOLD,
                                                      cv2.BORDER_REFLECT_101 + cv2.BORDER_ISOLATED)
            
            # Store the padded image in the pyramid
            self.ImagePyramid.append(current_pyramid_level)

        return


    def DistributeOctTree(self, ToDistributeKeys_List, minX, maxX, minY, maxY, level):

        N = self.features_per_level[level]

        # Compute how many initial nodes
        nIni = round(float(maxX - minX) / (maxY - minY))
        hX = float(maxX - minX) / nIni

        # Create initial nodes
        nodes_list = []
        IniNodes_list = []

        for i in range(nIni):
            ni = ExtractorNode()
            ni.UL = (minX + hX * i, minY)
            ni.UR = (minX + hX * (i + 1), minY)
            ni.BL = (ni.UL[0], maxY)
            ni.BR = (ni.UR[0], maxY)
            nodes_list.append(ni)
            IniNodes_list.append(ni)

        # Associate keypoints to child nodes
        for kp in ToDistributeKeys_List:
            idx = int((kp.pt[0] - minX) // hX)
            if idx < len(IniNodes_list):
                IniNodes_list[idx].keypoints_list.append(kp)

        # Remove empty nodes or nodes with only one keypoint
        nodes_list = [n for n in nodes_list if len(n.keypoints_list) > 0]
        for node in nodes_list:
            if len(node.keypoints_list) == 1:
                node.NoMore = True

        bFinish = False
        iteration = 0

        SizeAndPointerToNode_List = []

        while not bFinish:
            iteration += 1

            prevSize = len(nodes_list)

            SizeAndPointerToNode_List.clear()

            # Subdivide nodes
            for lit in nodes_list[:]:
                if not lit.NoMore:
                    child_nodes = lit.DivideNode()
                    nodes_list.remove(lit)
                    for child in child_nodes:
                        if len(child.keypoints_list) > 0:
                            nodes_list.append(child)
                            if len(child.keypoints_list) > 1:
                                SizeAndPointerToNode_List.append((len(child.keypoints_list), child))

            if len(nodes_list) >= N or len(nodes_list) == prevSize:
                bFinish = True
            elif len(nodes_list) + len(SizeAndPointerToNode_List) * 3 > N:
                # Further subdivide nodes if needed
                SizeAndPointerToNode_List.sort(reverse=True, key=lambda x: x[0])
                for i in range(len(SizeAndPointerToNode_List) - 1, -1, -1):
                    node = SizeAndPointerToNode_List[i][1]
                    child_nodes = node.DivideNode()
                    nodes_list.remove(node)
                    for child in child_nodes:
                        if len(child.keypoints_list) > 0:
                            nodes_list.append(child)
                            if len(child.keypoints_list) > 1:
                                SizeAndPointerToNode_List.append((len(child.keypoints_list), child))
                    if len(nodes_list) >= N:
                        break

                if len(nodes_list) >= N or len(nodes_list) == prevSize:
                    bFinish = True

        # Retain the best keypoint from each node
        ResultKeys_List = []
        for node in nodes_list:
            if node.keypoints_list:
                best_kp = max(node.keypoints_list, key=lambda kp: kp.response)
                ResultKeys_List.append(best_kp)

        return ResultKeys_List


    def IC_Angle(self, image, pt):
        m_01 = 0
        m_10 = 0

        centerX = int(round(pt[0]))
        centerY = int(round(pt[1]))

        patch_size = 31
        half_patch_size = patch_size // 2

        # Ensure the patch is within image bounds
        if centerX - half_patch_size < 0 or centerX + half_patch_size >= image.shape[1] or centerY - half_patch_size < 0 or centerY + half_patch_size >= image.shape[0]:
            return 0  # If the patch is outside the image bounds, return 0

        # Loop over the circular patch around the keypoint
        for u in range(-half_patch_size, half_patch_size + 1):
            m_10 += u * image[centerY, centerX + u]

        for v in range(1, half_patch_size + 1):
            v_sum = 0
            d = self.umax_list[v]
            for u in range(-d, d + 1):
                val_plus = image[centerY + v, centerX + u]
                val_minus = image[centerY - v, centerX + u]
                m_10 += u * (val_plus + val_minus)
                v_sum += (val_plus - val_minus)
            m_01 += v * v_sum

        angle = np.arctan2(m_01, m_10) * 180 / np.pi  # Convert to degrees
        if angle < 0:
            angle += 360

        return angle   


    def ComputeOrientation(self, level):
        for keypoint in self.allKeypoints[level]:
            keypoint.angle = self.IC_Angle(self.ImagePyramid[level], keypoint.pt)

        return

    def ComputeKeyPointsOctTree(self):

        self.allKeypoints = []
        W = 35.0  # Cell width for dividing the image into regions

        for level in range(self.nlevels):
            minBorderX = self.EDGE_THRESHOLD - 3
            minBorderY = minBorderX
            maxBorderX = self.ImagePyramid[level].shape[1] - self.EDGE_THRESHOLD + 3  # width - border
            maxBorderY = self.ImagePyramid[level].shape[0] - self.EDGE_THRESHOLD + 3  # height - border

            ToDistributeKeys_List = []

            # Image region size
            width = maxBorderX - minBorderX
            height = maxBorderY - minBorderY

            # Calculate number of rows and columns based on the region size and W (cell width)
            nCols = int(width / W)
            nRows = int(height / W)

            wCell = math.ceil(width / nCols)
            hCell = math.ceil(height / nRows)

            # Process each cell in the image region
            for i in range(nRows):
                iniY = minBorderY + i * hCell
                maxY = iniY + hCell + 6

                if iniY >= maxBorderY - 3:
                    continue
                if maxY > maxBorderY:
                    maxY = maxBorderY

                for j in range(nCols):
                    iniX = minBorderX + j * wCell
                    maxX = iniX + wCell + 6

                    if iniX >= maxBorderX - 6:
                        continue
                    if maxX > maxBorderX:
                        maxX = maxBorderX

                    # Apply FAST detector
                    vKeysCell = []
                    cell_region = self.ImagePyramid[level][iniY:maxY, iniX:maxX]
                    fast = cv2.FastFeatureDetector_create(self.iniThFAST, True)
                    vKeysCell = fast.detect(cell_region, None)

                    # If no keypoints are found with iniThFAST, try with minThFAST
                    if len(vKeysCell) == 0:
                        fast = cv2.FastFeatureDetector_create(self.minThFAST, True)
                        vKeysCell = fast.detect(cell_region, None)

                    # Adjust keypoint coordinates relative to the full image
                    for kp in vKeysCell:
                        kp.pt = (kp.pt[0] + j * wCell, kp.pt[1] + i * hCell)
                        ToDistributeKeys_List.append(kp)

            # Distribute keypoints across the image using an octree
            keypoints_list = self.DistributeOctTree(ToDistributeKeys_List, minBorderX, maxBorderX, minBorderY, maxBorderY, level)
            
            # Set keypoint size and scale
            scaledPatchSize = self.PATCH_SIZE * self.scaleFactor_list[level]
            for kp in keypoints_list:
                kp.pt = (kp.pt[0] + minBorderX, kp.pt[1] + minBorderY)
                kp.octave = level
                kp.size = scaledPatchSize

            # Store keypoints for this level
            self.allKeypoints.append(keypoints_list)

        # Compute keypoint orientations for each level
        for level in range(self.nlevels):
            self.ComputeOrientation(level)

        return

    def ComputeDescriptors(self, image, keypoints):

        # Initialize descriptors as a zero matrix of shape (number of keypoints, 32 bytes per descriptor)
        descriptors = np.zeros((len(keypoints), 32), dtype=np.uint8)

        # Loop through each keypoint and compute its descriptor
        for i, keypoint in enumerate(keypoints):
            self.compute_orb_descriptor(keypoint, image, descriptors[i])

        return descriptors


    def compute_orb_descriptor(self, keypoint, image, descriptor):
        # Get the keypoint's coordinates
        angle = keypoint.angle * np.pi / 180.0  # Convert angle to radians
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Precompute the rotation matrix elements for the keypoint
        center_x = int(round(keypoint.pt[0]))
        center_y = int(round(keypoint.pt[1]))

        # Descriptor is 32 bytes (256 bits)
        for k in range(32):
            val = 0
            for j in range(8):  # Each byte consists of 8 comparisons
                idx = 8 * k + j
                p1_x = int(center_x + cos_angle * self.pattern[idx][0] - sin_angle * self.pattern[idx][1])
                p1_y = int(center_y + sin_angle * self.pattern[idx][0] + cos_angle * self.pattern[idx][1])
                p2_x = int(center_x + cos_angle * self.pattern[idx + 1][0] - sin_angle * self.pattern[idx + 1][1])
                p2_y = int(center_y + sin_angle * self.pattern[idx + 1][0] + cos_angle * self.pattern[idx + 1][1])

                # Ensure the points are within image bounds
                if (0 <= p1_x < image.shape[1] and 0 <= p1_y < image.shape[0] and
                        0 <= p2_x < image.shape[1] and 0 <= p2_y < image.shape[0]):
                    if image[p1_y, p1_x] < image[p2_y, p2_x]:
                        val |= 1 << j

            descriptor[k] = val

        return

    def ORBExtract(self, image):

        # Compute Image Pyramid
        self.ComputePyramid(image)

        # Find Keypoints
        self.ComputeKeyPointsOctTree()


        # Initialize descriptors
        nkeypoints = sum(len(kp) for kp in self.allKeypoints)
        if nkeypoints == 0:
            return 0

        descriptors = np.zeros((nkeypoints, 32), dtype=np.int32)
        keypoints = [cv2.KeyPoint() for _ in range(nkeypoints)]

        # Loop through pyramid levels
        for level, keypoints_level in enumerate(self.allKeypoints):
            nkeypoints_level = len(keypoints_level)
            if nkeypoints_level == 0:
                continue

            # Preprocess image for this pyramid level
            workingMat = cv2.GaussianBlur(self.ImagePyramid[level], (7, 7), 2, borderType=cv2.BORDER_REFLECT_101)

            # Compute descriptors for the current level
            desc = self.ComputeDescriptors(workingMat, keypoints_level)

            # Scale keypoints and add them to the keypoint list
            scale = self.scaleFactor_list[level]
            for i, keypoint in enumerate(keypoints_level):
                if level != 0:
                    keypoint.pt = (keypoint.pt[0] * scale, keypoint.pt[1] * scale)

                keypoints[i] = keypoint
                descriptors[i] = desc[i]
        return keypoints, descriptors


    def get_bit_pattern(self):
        bit_pattern_31_ = np.array([
            8,-3, 9,5,
            4,2, 7,-12,
            -11,9, -8,2,
            7,-12, 12,-13,
            2,-13, 2,12,
            1,-7, 1,6,
            -2,-10, -2,-4,
            -13,-13, -11,-8,
            -13,-3, -12,-9,
            10,4, 11,9,
            -13,-8, -8,-9,
            -11,7, -9,12,
            7,7, 12,6,
            -4,-5, -3,0,
            -13,2, -12,-3,
            -9,0, -7,5,
            12,-6, 12,-1,
            -3,6, -2,12,
            -6,-13, -4,-8,
            11,-13, 12,-8,
            4,7, 5,1,
            5,-3, 10,-3,
            3,-7, 6,12,
            -8,-7, -6,-2,
            -2,11, -1,-10,
            -13,12, -8,10,
            -7,3, -5,-3,
            -4,2, -3,7,
            -10,-12, -6,11,
            5,-12, 6,-7,
            5,-6, 7,-1,
            1,0, 4,-5,
            9,11, 11,-13,
            4,7, 4,12,
            2,-1, 4,4,
            -4,-12, -2,7,
            -8,-5, -7,-10,
            4,11, 9,12,
            0,-8, 1,-13,
            -13,-2, -8,2,
            -3,-2, -2,3,
            -6,9, -4,-9,
            8,12, 10,7,
            0,9, 1,3,
            7,-5, 11,-10,
            -13,-6, -11,0,
            10,7, 12,1,
            -6,-3, -6,12,
            10,-9, 12,-4,
            -13,8, -8,-12,
            -13,0, -8,-4,
            3,3, 7,8,
            5,7, 10,-7,
            -1,7, 1,-12,
            3,-10, 5,6,
            2,-4, 3,-10,
            -13,0, -13,5,
            -13,-7, -12,12,
            -13,3, -11,8,
            -7,12, -4,7,
            6,-10, 12,8,
            -9,-1, -7,-6,
            -2,-5, 0,12,
            -12,5, -7,5,
            3,-10, 8,-13,
            -7,-7, -4,5,
            -3,-2, -1,-7,
            2,9, 5,-11,
            -11,-13, -5,-13,
            -1,6, 0,-1,
            5,-3, 5,2,
            -4,-13, -4,12,
            -9,-6, -9,6,
            -12,-10, -8,-4,
            10,2, 12,-3,
            7,12, 12,12,
            -7,-13, -6,5,
            -4,9, -3,4,
            7,-1, 12,2,
            -7,6, -5,1,
            -13,11, -12,5,
            -3,7, -2,-6,
            7,-8, 12,-7,
            -13,-7, -11,-12,
            1,-3, 12,12,
            2,-6, 3,0,
            -4,3, -2,-13,
            -1,-13, 1,9,
            7,1, 8,-6,
            1,-1, 3,12,
            9,1, 12,6,
            -1,-9, -1,3,
            -13,-13, -10,5,
            7,7, 10,12,
            12,-5, 12,9,
            6,3, 7,11,
            5,-13, 6,10,
            2,-12, 2,3,
            3,8, 4,-6,
            2,6, 12,-13,
            9,-12, 10,3,
            -8,4, -7,9,
            -11,12, -4,-6,
            1,12, 2,-8,
            6,-9, 7,-4,
            2,3, 3,-2,
            6,3, 11,0,
            3,-3, 8,-8,
            7,8, 9,3,
            -11,-5, -6,-4,
            -10,11, -5,10,
            -5,-8, -3,12,
            -10,5, -9,0,
            8,-1, 12,-6,
            4,-6, 6,-11,
            -10,12, -8,7,
            4,-2, 6,7,
            -2,0, -2,12,
            -5,-8, -5,2,
            7,-6, 10,12,
            -9,-13, -8,-8,
            -5,-13, -5,-2,
            8,-8, 9,-13,
            -9,-11, -9,0,
            1,-8, 1,-2,
            7,-4, 9,1,
            -2,1, -1,-4,
            11,-6, 12,-11,
            -12,-9, -6,4,
            3,7, 7,12,
            5,5, 10,8,
            0,-4, 2,8,
            -9,12, -5,-13,
            0,7, 2,12,
            -1,2, 1,7,
            5,11, 7,-9,
            3,5, 6,-8,
            -13,-4, -8,9,
            -5,9, -3,-3,
            -4,-7, -3,-12,
            6,5, 8,0,
            -7,6, -6,12,
            -13,6, -5,-2,
            1,-10, 3,10,
            4,1, 8,-4,
            -2,-2, 2,-13,
            2,-12, 12,12,
            -2,-13, 0,-6,
            4,1, 9,3,
            -6,-10, -3,-5,
            -3,-13, -1,1,
            7,5, 12,-11,
            4,-2, 5,-7,
            -13,9, -9,-5,
            7,1, 8,6,
            7,-8, 7,6,
            -7,-4, -7,1,
            -8,11, -7,-8,
            -13,6, -12,-8,
            2,4, 3,9,
            10,-5, 12,3,
            -6,-5, -6,7,
            8,-3, 9,-8,
            2,-12, 2,8,
            -11,-2, -10,3,
            -12,-13, -7,-9,
            -11,0, -10,-5,
            5,-3, 11,8,
            -2,-13, -1,12,
            -1,-8, 0,9,
            -13,-11, -12,-5,
            -10,-2, -10,11,
            -3,9, -2,-13,
            2,-3, 3,2,
            -9,-13, -4,0,
            -4,6, -3,-10,
            -4,12, -2,-7,
            -6,-11, -4,9,
            6,-3, 6,11,
            -13,11, -5,5,
            11,11, 12,6,
            7,-5, 12,-2,
            -1,12, 0,7,
            -4,-8, -3,-2,
            -7,1, -6,7,
            -13,-12, -8,-13,
            -7,-2, -6,-8,
            -8,5, -6,-9,
            -5,-1, -4,5,
            -13,7, -8,10,
            1,5, 5,-13,
            1,0, 10,-13,
            9,12, 10,-1,
            5,-8, 10,-9,
            -1,11, 1,-13,
            -9,-3, -6,2,
            -1,-10, 1,12,
            -13,1, -8,-10,
            8,-11, 10,-6,
            2,-13, 3,-6,
            7,-13, 12,-9,
            -10,-10, -5,-7,
            -10,-8, -8,-13,
            4,-6, 8,5,
            3,12, 8,-13,
            -4,2, -3,-3,
            5,-13, 10,-12,
            4,-13, 5,-1,
            -9,9, -4,3,
            0,3, 3,-9,
            -12,1, -6,1,
            3,2, 4,-8,
            -10,-10, -10,9,
            8,-13, 12,12,
            -8,-12, -6,-5,
            2,2, 3,7,
            10,6, 11,-8,
            6,8, 8,-12,
            -7,10, -6,5,
            -3,-9, -3,9,
            -1,-13, -1,5,
            -3,-7, -3,4,
            -8,-2, -8,3,
            4,2, 12,12,
            2,-5, 3,11,
            6,-9, 11,-13,
            3,-1, 7,12,
            11,-1, 12,4,
            -3,0, -3,6,
            4,-11, 4,12,
            2,-4, 2,1,
            -10,-6, -8,1,
            -13,7, -11,1,
            -13,12, -11,-13,
            6,0, 11,-13,
            0,-1, 1,4,
            -13,3, -9,-2,
            -9,8, -6,-3,
            -13,-6, -8,-2,
            5,-9, 8,10,
            2,7, 3,-9,
            -1,-6, -1,-1,
            9,5, 11,-2,
            11,-3, 12,-8,
            3,0, 3,5,
            -1,4, 0,10,
            3,-6, 4,5,
            -13,0, -10,5,
            5,8, 12,11,
            8,9, 9,-6,
            7,-4, 8,-12,
            -10,4, -10,9,
            7,3, 12,4,
            9,-7, 10,-2,
            7,0, 12,-2,
            -1,-6, 0,-11
        ], dtype=np.int32)

        return bit_pattern_31_
