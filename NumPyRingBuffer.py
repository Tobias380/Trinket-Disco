class NumPyRingBuffer(object):
    """
    NumPyRingBuffer is a circular buffer implemented using a one dimensional 
    numpy array on the backend. The algorithm used to implement the ring buffer
    behavour does not require any array copies to occur while the ring buffer is
    maintained, while at the same time allowing sequential element access into the 
    numpy array using a subset of standard slice notation.
    
    When the circular buffer is created, a maximum size , or maximum
    number of elements,  that the buffer can hold *must* be specified. When 
    the buffer becomes full, each element added to the buffer removes the oldest
    element from the buffer so that max_size is never exceeded. 
    
    The class supports simple slice type access to the buffer contents
    with the following restrictions / considerations:
    
    #. Negative indexing is not supported.
 
    Items area dded to the ring buffer using the classes append method.
    
    The current number of elements in the buffer can be retrieved using the 
    getLength() method of the class. 
    
    The isFull() method can be used to determine if
    the ring buffer has reached its maximum size, at which point each new element
    added will disregard the oldest element in the array.
    
    The getElements() method is used to retrieve the actual numpy array containing
    the elements in the ring buffer. The element in index 0 is the oldest remaining 
    element added to the buffer, and index n (which can be up to max_size-1)
    is the the most recent element added to the buffer.

    Methods that can be called from a standard numpy array can also be called using the 
    NumPyRingBuffer instance created. However Numpy module level functions will not accept
    a NumPyRingBuffer as a valid arguement.
    
    To clear the ring buffer and start with no data in the buffer, without
    needing to create a new NumPyRingBuffer object, call the clear() method
    of the class.
    
    Example::
    
        ring_buffer=NumPyRingBuffer(10)
        
        for i in xrange(25):
            ring_buffer.append(i)
            print '-------'
            print 'Ring Buffer Stats:'
            print '\tWindow size: ',len(ring_buffer)
            print '\tMin Value: ',ring_buffer.min()
            print '\tMax Value: ',ring_buffer.max()
            print '\tMean Value: ',ring_buffer.mean()
            print '\tStandard Deviation: ',ring_buffer.std()
            print '\tFirst 3 Elements: ',ring_buffer[:3]
            print '\tLast 3 Elements: ',ring_buffer[-3:]
        
        
        
    """
    def __init__(self, max_size, dtype=numpy.float32):
        self._dtype=dtype
        self._npa=numpy.empty(max_size*2,dtype=dtype)
        self.max_size=max_size
        self._index=0
        
    def append(self, element):
        """
        Add element e to the end of the RingBuffer. The element must match the 
        numpy data type specified when the NumPyRingBuffer was created. By default,
        the RingBuffer uses float32 values.
        
        If the Ring Buffer is full, adding the element to the end of the array 
        removes the currently oldest element from the start of the array.
        
        :param numpy.dtype element: An element to add to the RingBuffer.
        :returns None:
        """
        i=self._index
        self._npa[i%self.max_size]=element
        self._npa[(i%self.max_size)+self.max_size]=element
        self._index+=1

    def getElements(self):
        """
        Return the numpy array being used by the RingBuffer, the length of 
        which will be equal to the number of elements added to the list, or
        the last max_size elements added to the list. Elements are in order
        of addition to the ring buffer.
        
        :param None:
        :returns numpy.array: The array of data elements that make up the Ring Buffer.
        """
        return self._npa[self._index%self.max_size:(self._index%self.max_size)+self.max_size]

    def isFull(self):
        """
        Indicates if the RingBuffer is at it's max_size yet.
        
        :param None:
        :returns bool: True if max_size or more elements have been added to the RingBuffer; False otherwise.
        """
        return self._index >= self.max_size
        
    def clear(self):
        """
        Clears the RingBuffer. The next time an element is added to the buffer, it will have a size of one.
        
        :param None:
        :returns None: 
        """
        self._index=0
        
    def __setitem__(self, indexs,v):
        if isinstance(indexs,(list,tuple)):
            for i in indexs:
                if isinstance(i, (int,long)):
                    i=i+self._index
                    self._npa[i%self.max_size]=v
                    self._npa[(i%self.max_size)+self.max_size]=v
                elif isinstance(i,slice):
                    istart=indexs.start
                    if istart is None:
                        istart=0
                    istop=indexs.stop
                    if indexs.stop is None:
                        istop=0
                    start=istart+self._index
                    stop=istop+self._index            
                    self._npa[slice(start%self.max_size,stop%self.max_size,i.step)]=v
                    self._npa[slice((start%self.max_size)+self.max_size,(stop%self.max_size)+self.max_size,i.step)]=v
        elif isinstance(indexs, (int,long)):
            i=indexs+self._index
            self._npa[i%self.max_size]=v
            self._npa[(i%self.max_size)+self.max_size]=v
        elif isinstance(indexs,slice):
            istart=indexs.start
            if istart is None:
                istart=0
            istop=indexs.stop
            if indexs.stop is None:
                istop=0
            start=istart+self._index
            stop=istop+self._index  
            self._npa[slice(start%self.max_size,stop%self.max_size,indexs.step)]=v
            self._npa[slice((start%self.max_size)+self.max_size,(stop%self.max_size)+self.max_size,indexs.step)]=v
        else:
            raise TypeError()

    def __getitem__(self, indexs):
        current_array=self.getElements()
        if isinstance(indexs,(list,tuple)):
            rarray=[]
            for i in indexs:
                if isinstance(i, (int,long)):
                    rarray.append(current_array[i])
                elif isinstance(i,slice):          
                    rarray.extend(current_array[i])
            return numpy.asarray(rarray,dtype=self._dtype)
        elif isinstance(indexs, (int,long,slice)):
            return current_array[indexs]
        else:
            raise TypeError()
    
    def __getattr__(self,a):
        if self._index<self.max_size:
            return getattr(self._npa[:self._index],a)
        return getattr(self._npa[self._index%self.max_size:(self._index%self.max_size)+self.max_size],a)
    
    def __len__(self):
        if self.isFull():
            return self.max_size
        return self._index
        
###############################################################################
#
