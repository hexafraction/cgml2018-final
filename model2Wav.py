[lp,gt,mx]=sess.run([labels_predicted, labels, cutmix])
bigmix = [lp,gt,mx]
bigmix[0] = bigmix[0].reshape(245810,2)
bigmix[1] = bigmix[1].reshape(245810,2)
bigmix[2] = bigmix[2].reshape(245810,2)


for i in range(30): 
	[lp,gt,mx]=sess.run([labels_predicted, labels, cutmix])
	bigmix[0] = np.concatenate((bigmix[0],lp.reshape(245810,2)),0)
	bigmix[1] = np.concatenate((bigmix[1],gt.reshape(245810,2)),0)
	bigmix[2] = np.concatenate((bigmix[2],mx.reshape(245810,2)),0)


wavio.write("demo20181129_long/mx.wav", bigmix[0], 44100, sampwidth=3)
wavio.write("demo20181129_long/gt.wav", bigmix[1], 44100, sampwidth=3)
wavio.write("demo20181129_long/lp.wav", bigmix[2], 44100, sampwidth=3)
