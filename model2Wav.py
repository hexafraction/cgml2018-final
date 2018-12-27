[lp,gt,mx]=sess.run([labels_predicted, labels, mc])
bigmix = [lp,gt,mx]
bigmix[0] = bigmix[0].reshape(286820//2,2)
bigmix[1] = bigmix[1].reshape(286820//2,2)
bigmix[2] = bigmix[2].reshape(286820//2,2)


for i in range(30): 
	print(i)
	[lp,gt,mx]=sess.run([labels_predicted, labels, mc])
	bigmix[0] = np.concatenate((bigmix[0],lp.reshape(286820//2,2)),0)
	bigmix[1] = np.concatenate((bigmix[1],gt.reshape(286820//2,2)),0)
	bigmix[2] = np.concatenate((bigmix[2],mx.reshape(286820//2,2)),0)


wavio.write("/tmp/phlp.wav", bigmix[0], 22050, sampwidth=3)
wavio.write("/tmp/phgt.wav", bigmix[1], 22050, sampwidth=3)
wavio.write("/tmp/phmx.wav", bigmix[2], 22050, sampwidth=3)
