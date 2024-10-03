class KeyFrameDataBase:
	def __init__():

		self.word2kfs_dict = {} # InvertedIndex

	def AddKF2DB(self, kf):
		# Parse the BowVec and update the invertedIndex
		for word in kf.BowList:
			if word in list(self.word2kfs_dict.keys()):
				self.word2kfs_dict[word].append(kf)
			else:
				self.word2kfs_dict[word] = [kf]

	def DetectNBestCandidates(self, QueryKF, numCandidates=3):

		LoopBow_List = []
		KFSharingWords_Dict = []

		# First retrieve the connected Key Frames
		ConnectedKeyFrames_list = QueryKF.GetConnectedKeyFrames()

		# Search all keyframes that share a word with current frame
		for word in QueryKF.BowList:

			KF_list = self.word2kfs_dict[word]

			for kf in KF_list:

				if kf.PlaceRecognitionQueryUID != QueryKF.kf_uid:
					kf.PlaceRecognitionWords = 0

					if kf not in ConnectedKeyFrames_list:
						KFSharingWords_Dict[kf] = 0

				KFSharingWords_Dict[kf] += 1

		if len(KFSharingWords_Dict[kf]) == 0:
			return LoopBow_List

		maxCommonWords = max(list(KFSharingWords_Dict[kf].values()))

		minCommonWords = 0.8*maxCommonWords

		# Compute Similarity Score
		ScoreAndMatch_dict = {}
		for kf,numCommonWords in KFSharingWords_Dict.items():
			if numCommonWords > minCommonWords:
				kf.PlaceRecognitionScore = self.voc.score(QueryKF, kf)
				ScoreAndMatch_dict[kf] = kf.PlaceRecognitionScore
				
		if len(ScoreAndMatch_dict) == 0:
			return LoopBow_List

		# Compute Accumulated Scores
		AccScoreAndMatch_dict = {}
		bestAccScore = 0

		for kf,score in ScoreAndMatch_dict.items():
			best_neigh_kf_list = kf.GetBestCovisibilityKeyFrames(10) # Again here kf needs to be a Camera object and not an ID
	
			for kf in best_neigh_kf_list:
				if kf.PlaceRecognitionQueryUID != QueryKF.kf_uid:
					continue

				accScore += kf.PlaceRecognitionScore

			score_list = [kf.PlaceRecognitionScore for kf in best_neigh_kf_list]
			best_score = max(score_list)
			best_kf = score_list.index(best_score)
			AccScoreAndMatch_dict[best_kf] = accScore


		# TO DO - Sort the AccScoreMatch list before processing it

		for kf in list(AccScoreAndMatch_dict.keys()):

			if len(LoopBow_List) == numCandidates:
				break

			if kf not in LoopBow_List:
				LoopBow_List.append(kf)
