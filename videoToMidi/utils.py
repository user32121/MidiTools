import numpy as np

#assumes range in [0,1]
def radialDistance(a,b):
    return np.minimum(np.minimum(np.abs(a-b), np.abs(a-b+1)), np.abs(a-b-1))

#project a in the range [b-0.5,b+0.5]
def projectCloseTo(a, b):
    if(np.shape(a) == ()):
        return np.array([a, a+1, a-1])[np.argmin([np.abs(a-b), np.abs(a-b+1), np.abs(a-b-1)], axis=0)]
    else:
        return np.array([a, a+1, a-1])[np.argmin([np.abs(a-b), np.abs(a-b+1), np.abs(a-b-1)], axis=0), np.arange(len(a))]

def clusterByHue(notes, hues):
    extractedHues = hues[notes>0]

    #circular mean-shift clustering
    RADIUS = 0.05
    ITERATIONS = 10
    print("mean-shift clustering with radius: {}, iterations: {}".format(RADIUS, ITERATIONS))
    centroids = list(np.arange(0, 1, RADIUS))
    for _ in range(ITERATIONS):
        i = 0
        while i < len(centroids):
            nearPoints = extractedHues[radialDistance(extractedHues, centroids[i]) <= RADIUS]
            if(len(nearPoints) == 0):
                centroids.pop(i)
                continue
            centroids[i] = np.sum(projectCloseTo(nearPoints, centroids[i]))/len(nearPoints)
            i += 1

    #merge close centroids
    i = 1
    while i < len(centroids):
        if(radialDistance(centroids[i], centroids[i-1]) <= RADIUS/2):
            centroids[i] = projectCloseTo((centroids[i] + projectCloseTo(centroids[i-1], centroids[i]))/2, 0.5)
            centroids.pop(i-1)
            continue
        i += 1
    print("detected {} colors".format(len(centroids)))

    #color notes
    for t in range(len(notes)):
        for i in range(len(notes[t])):
            if(notes[t,i]):
                notes[t,i] *= np.argmin(radialDistance(centroids, hues[t,i]))+1
    
    return notes, len(centroids)
