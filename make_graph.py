import matplotlib.pyplot as plt

# define the lists
#DeUnet_VGG16_CE_3epoch = [91.0,63.77097729516288, 46.6436327739388, 39.04244817374136, 35.9822309970385, 33.41559723593287, 31.786771964461995, 30.454096742349456, 29.66436327739388, 28.874629812438304, 27.541954590325766]
DeUnet_VGG16_CE_2epoch = [91.5,63.96841066140178, 47.18657453109576, 39.04244817374136, 35.68608094768016, 31.934846989141164, 30.454096742349456, 30.108588351431393, 28.97334649555775, 28.529121421520237, 28.677196446199407]
#DeUnet_VGG16_CE_1epoch = [90.8,61.45113524185587, 44.52122408687068, 37.660414610069104, 34.40276406712734, 30.65153010858835, 30.306021717670287, 28.529121421520237, 28.232971372161895, 27.690029615004935, 28.381046396841064]
VGG16 = [94.0,51.38203356367226, 27.541954590325766, 16.04146100691017, 12.438302073050345, 8.736426456071076, 7.897334649555775, 6.367226061204343, 5.62685093780849, 5.873642645607108, 5.429417571569595]
#DeUnet_VGG16_1epoch = [89.5,60.85883514313919, 44.37314906219151, 36.62388943731491, 33.81046396841066, 31.83613030602172, 29.41757156959526, 28.479763079960513, 29.022704837117473, 28.035538005923, 26.801579466929912]
DeUnet_VGG16_2epoch = [89.8,63.228035538005926, 45.06416584402764, 38.943731490621914, 34.35340572556762, 32.67522211253702, 30.898321816386968, 29.615004935834158, 28.627838104639682, 27.49259624876604, 29.41757156959526]
#DeUnet_VGG16_3epoch = [88.8,63.91905231984205, 46.89042448173741, 39.04244817374136, 34.35340572556762, 32.6258637709773, 32.03356367226061, 29.911154985192496, 29.51628825271471, 28.57847976307996, 28.381046396841064]
ep = [0.00,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

# plot the lists
#plt.plot(DeUnet_VGG16_CE_3epoch, color='blue', label='DeUnet_VGG16_CE_3epoch',)
plt.plot(DeUnet_VGG16_CE_2epoch, color='red', label='DeUnet_VGG16_CE_2epoch')
#plt.plot(DeUnet_VGG16_CE_1epoch, color='green', label='DeUnet_VGG16_CE_1epoch')

#plt.plot(DeUnet_VGG16_1epoch, color='orange', label='DeUnet_VGG16_1epoch')
plt.plot(DeUnet_VGG16_2epoch, color='purple', label='DeUnet_VGG16_2epoch')
#plt.plot(DeUnet_VGG16_3epoch, color='cyan', label='DeUnet_VGG16_3epoch')

#plt.plot(VGG16, color='pink', label='VGG16')
#plt.plot(ep, color='green', label='ep')

# set the title and axis labels
plt.title('Experiment')
plt.xlabel(ep)
plt.ylabel('test_accuracy')

# add a legend
plt.legend()
plt.grid()
# display the graph
plt.show()
