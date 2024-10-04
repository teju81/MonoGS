
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
        self.iniThFAST = None
        self.minThFAST = None

        self.nlevels = None
        self.nfeatures = None
        self.features_per_level = None
        self.umax_list = []

        self.scaleFactor = None
        self.scaleFactor_list = []
        self.invScaleFactor_list = []
        self.levelSigma2_list = []
        self.invLevelSigma2_list = []

        self.ImagePyramid = []

        self.PATCH_SIZE = 31
        self.HALF_PATCH_SIZE = 15
        self.EDGE_THRESHOLD = 19

    def GaussianBlur(self):
        pass

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


    def ComputeOrientation(self, level):
        for i, keypoint in enumerate(self.allKeypoints[level]):
            keypoint.angle = self.IC_Angle(self.ImagePyramid[level], self.allKeypoints[level].pt)
            self.allKeypoints[level][i] = keypoint

        return

    def IC_Angle(self, image, pt)
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


    def ComputePyramid(self, image):
        self.ImagePyramid = []
        for level in range(self.nlevels):
            scale = self.invScaleFactor[level]
            sz = (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale)))  # Width, Height (cols, rows)
            whole_size = (sz[0] + EDGE_THRESHOLD * 2, sz[1] + EDGE_THRESHOLD * 2)  # Add padding (whole size)

            # Create a temporary image with padding
            temp = np.zeros((whole_size[1], whole_size[0], image.shape[2]) if len(image.shape) == 3 else (whole_size[1], whole_size[0]), dtype=image.dtype)

            # Crop out the center image where there is no padding
            current_pyramid_level = temp[EDGE_THRESHOLD:EDGE_THRESHOLD + sz[1], EDGE_THRESHOLD:EDGE_THRESHOLD + sz[0]]
            
            # Resize the image for this pyramid level
            if level != 0:
                # Resize the previous level's image to the current size
                resized_image = cv2.resize(self.ImagePyramid[level - 1], sz, interpolation=cv2.INTER_LINEAR)
                current_pyramid_level[:, :] = resized_image  # Copy resized image to the pyramid level
            else:
                # The first level is just the original image
                current_pyramid_level[:, :] = cv2.resize(image, sz, interpolation=cv2.INTER_LINEAR)

            # Apply the border to the temp image
            if level == 0:
                # For the first level, we add padding to the original image
                temp_with_border = cv2.copyMakeBorder(image, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                                                      cv2.BORDER_REFLECT_101)
            else:
                # For other levels, we add padding to the resized image
                temp_with_border = cv2.copyMakeBorder(current_pyramid_level, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                                                      cv2.BORDER_REFLECT_101 + cv2.BORDER_ISOLATED)
            
            # Store the padded image in the pyramid
            self.ImagePyramid.append(current_pyramid_level)

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

            #Distribute keypoints across the image using an octree
            keypoints_list = self.DistributeOctTree(ToDistributeKeys_List, minBorderX, maxBorderX, minBorderY, maxBorderY, level)
            
            # Set keypoint size and scale
            scaledPatchSize = self.PATCH_SIZE * self.scaleFactor_list[level]
            for kp in keypoints_list:
                kp.pt = (kp.pt[0] + minBorderX, kp.pt[1] + minBorderY)
                kp.octave = level
                kp.size = scaledPatchSize

            # Store keypoints for this level
            self.allKeypoints.append(keypoints_list)

        # TO DO: Compute keypoint orientations for each level
        for level in range(self.nlevels):
            self.ComputeOrientation(level)

        return

    def ComputeDescriptors(self):
        pass


    def ORBExtract(self, image):

        self.ComputePyramid(image)

        self.ComputeKeyPointsOctTree()

        self.GaussianBlur()

        self.ComputeDescriptors()
