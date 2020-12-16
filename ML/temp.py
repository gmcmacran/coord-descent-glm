X = np.ones([100, 1])
Y = 2 * X + 3
X = np.hstack([X, np.ones([X.shape[0], 1])])

B = np.ndarray([2, 1])
B[0, 0] = 2
B[1, 0] = 3.1


X = np.random.rand(100, 1)
Y = 2 * X + 3

B = np.zeros([2, 1])
step = np.array([.1])
shuffle = True
step = np.array([.1])
