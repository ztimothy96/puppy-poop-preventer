import numpy as np
from collections import OrderedDict
from scipy.spatial.distance import cdist

class CentroidTracker:
    def __init__(self, max_time_disappeared):
        self.next_id = 0
        self.objects = OrderedDict() # id -> centroid
        self.time_disappeared = OrderedDict() # id -> time disappeared
        self.max_time_disappeared = max_time_disappeared

    def register(self, centroid):
        print('registering new object ID {}'.format(self.next_id))
        self.objects[self.next_id] = centroid
        self.time_disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, id):
        print('deregistering object ID {}'.format(id))
        del self.objects[id]
        del self.time_disappeared[id]

    def update(self, rects):
        print(rects)
        if not rects:
            for id in self.time_disappeared.keys():
                self.time_disappeared[id] += 1
                if self.time_disappeared[id] > self.max_time_disappeared:
                    self.deregister(id)
            return self.objects
        
        new_centroids = np.zeros((len(rects), 2), dtype=int)
        for i, (x1, y1, x2, y2) in enumerate(rects):
            new_centroids[i] = ((x1 + x2)/2, (y1+y2)/2)
            
        if not self.objects:
            for centroid in new_centroids:
                self.register(centroid)
            return self.objects
        
        '''
        old centroids: m_a x 2, 
        new centroids: m_b x 2, 
        distances: m_a X m_b 
        for each existing object (row), find the closest centroid (col).
        '''
        ids = list(self.objects.keys())
        old_centroids = np.array(list(self.objects.values()))
        distances = cdist(old_centroids, new_centroids, 'sqeuclidean')
        used_columns = set()
        used_rows = set()
        '''
        rows ordered by increasing distance to closest point
        cols closest to the corresponding row
        '''
        rows = distances.min().argsort() 
        cols = distances.argmin(axis=1)[rows] 
        R, C = distances.shape

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_columns:
                continue
            id = ids[r]
            self.objects[id] = new_centroids[c]
            self.time_disappeared[id] = 0
            used_rows.add(r)
            used_columns.add(c)

        print(distances.shape)
        if R >= C:
            for r in range(R):
                id = ids[r]
                if not r in used_rows:
                    self.time_disappeared[id] += 1
                if self.time_disappeared[id] > self.max_time_disappeared:
                    self.deregister(id)
        else:
            for c in range(C):
                if not c in used_columns:
                    self.register(new_centroids[c])
        return self.objects






