import scipy.io as sio
filename = 'mnist_test.mat'
test = sio.loadmat(filename)
test_s = test["mnist_test"]

filename = 'mnist_test_labels.mat'
testlabel = sio.loadmat(filename)
test_l = testlabel["mnist_test_labels"]