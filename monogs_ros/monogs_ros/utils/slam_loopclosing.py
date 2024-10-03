class LoopClosing:
	def __init__(viewpoint, kfdb):
		self.mQueryKF = viewpoint
		self.mKFDB = kfdb
		self.mLoopBow_List = []
		self.mbLoopClosureDetected = False

	def DetectCommonRegionsFromBoW(self):
		pass


	def DetectLoopClosure(self):
		numCandidates = 3
		self.mKFDB.DetectNBestCandidates(self.mLoopBow_List, numCandidates)

		# Check the BoW candidates if the geometric candidate list is empty
		if len(mLoopBow_List) == 0:
			self.DetectCommonRegionsFromBoW()

		self.mKFDB.AddKF2DB(self.mQueryKF)


    def f2b_listener_callback(self, b2lc_msg):
        self.get_logger().info('I heard from backend: "%s"' % b2lc_msg.msg)
        _ = self.convert_from_ros_msg(b2lc_msg)

		self.DetectLoopClosure()
		if self.mbLoopClosureDetected:
			self.LoopCorrection()
