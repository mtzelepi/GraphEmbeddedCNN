import caffe
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (10, 10)





caffe.set_mode_gpu() 
net = caffe.Net( '/path/deploy.prototxt',
                 '/path/model.caffemodel',
               caffe.TEST)



imgtxt = '/path/test.txt' #image paths
q = open(imgtxt)
linelist = [line for line in q.readlines()]

x=0
for img in linelist:
    x=x+1
    img=img.strip()
    im = caffe.io.load_image(img)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #transformer.set_mean('data', np.load('/path/mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))  
    plt.subplot(1, 2, 1)
    plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(out['prob'][0,1], cmap='jet')
    plt.axis('off')
    plt.savefig('heatmapExample_' +str(x)+'.jpg', bbox_inches='tight')
    plt.subplots_adjust(wspace=0, hspace=0)

 
