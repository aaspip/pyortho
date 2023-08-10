from orthocfun import *
import numpy as np

def focus(din,rect,dim=3,niter=100,verb=1):
	'''
	FOCUSC: Focusing indicator. 
	
	
	dim: 2 (processing up to second dimension)
		 3 (processing up to third dimension)
	'''

	if din.ndim==2:	#for 2D problems
		din=np.expand_dims(din, axis=2)
	
	[n1,n2,n3]=din.shape
	
	r1=rect[0];
	r2=rect[1];
	r3=rect[2];
	
	din=np.float32(din.flatten(order='F'));
	
	print(n1,n2,n3,r1,r2,r3,niter,dim,verb);
	dout=Cfocus(din,n1,n2,n3,r1,r2,r3,niter,dim,verb);
	
	dout=dout.reshape(n1,n2,n3,order='F')
	
	if n3==1:	#for 1D/2D problems
		dout=np.squeeze(simi)
		
	return dout

