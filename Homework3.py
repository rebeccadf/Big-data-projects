# Import Packages
#import findspark
#findspark.init()
#findspark.find()

from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    L = int(sys.argv[4])
    start = 0
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(lambda x : strToVector(x)).repartition(L).cache()
    N = inputPoints.count()
    end = time.time()
    
    
    # Pring input parameters
    print("File : " + filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", str((end-start)*1000), " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)

    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()
    print("Objective function = ", objective)
    print("Time to compute objective function: ", str((end-start)*1000), " ms")
     



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def strToVector(str):
    out = tuple(map(float, str.split(',')))
    return out



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method squaredEuclidean: squared euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def squaredEuclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return res



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method euclidean:  euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points, k, z, L):

    
    #------------- ROUND 1 ---------------------------
    start_time1= time.time()
    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k+z+1)).cache()
    coreset.count()
    end_time1= (time.time()-start_time1)*1000
    
    
    # END OF ROUND 1

    
    #------------- ROUND 2 ---------------------------
    start_time2 = time.time()
    elems = coreset.collect()
    print("End")
    coresetPoints = list() 
    coresetWeights = list()
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])
    
    global dist_matrix
    dist_matrix = compute_dist(coresetPoints)
    dist_matrix = dist_matrix + dist_matrix.T - np.diag(np.diag(dist_matrix))
    # ****** ADD YOUR CODE
    # ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
    solution = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 2)
    
    end_time2 = time.time() 
    # ****** Measure and print times taken by Round 1 and Round 2, separately
    # ****** Return the final solution
    print("Time for round 1: ", end_time1)
    print("Time for round 2: ", end_time2)
    print(solution)
    return solution
   

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, points):
    partition = list(iter)
    centers = kCenterFFT(partition, points)
    weights = computeWeights(partition, centers)
    c_w = list()
    for i in range(0, len(centers)):
        entry = (centers[i], weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w
    
    
    
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    random.seed(123)
    idx_rnd = random.randint(0, len(points)-1)
    centers = [points[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(points))]
    dist_near_center = [squaredEuclidean(points[i], centers[0]) for i in range(len(points))]

    for i in range(k-1):
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0] # argmax operation
        centers.append(points[new_center_idx])
        for j in range(len(points)):
            if j != new_center_idx:
                dist = squaredEuclidean(points[j], centers[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return centers



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        mindist = squaredEuclidean(point,centers[0])
        for i in range(1, len(centers)):
            dist = squaredEuclidean(point,centers[i])
            if dist < mindist:
                mindist = dist
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights


def compute_dist(P): #to precompute the distances between points, since the matrix will be symmetric we only compute the "lower triangle"
    dist_matrix = np.zeros((len(P),len(P)), dtype = float)
    for i in range(len(P)):
        for j in range(i, len(P)):
            dist_matrix[j][i] = euclidean(P[i],P[j])
    return dist_matrix

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


def SeqWeightedOutliers(P,W,k,z,alpha):
    init_t =time.time() 
    min_dist = np.min(dist_matrix[:k+z+1, :k+z+1][np.nonzero(dist_matrix[:k+z+1, :k+z+1])]) # min dist betweem the first k+z+1 points
    r = min_dist / 2
    print("Initial guess =  ", r)
    n_guesses = 1
    
    while True:
        S = [] 
        Z = P[:] 
        W_z = sum(W) 

        while (len(S) < k) and (W_z >0):
            max_d = 0
            for x in P: #only for the y belonging to Z
                x_index = P.index(x) #find the index of the y in P 
                ball1 = np.where(dist_matrix[x_index]<(1+2 * alpha) * r)[0] #returns the indexes of the y points belonging to B_Z centered                                                                             in the current x with ray (1+2alpha)r
                ball1_new = [i for i in ball1 if P[i] in Z]                
                        
                ball_weight = np.sum([W[i] for i in ball1_new])
                
                if ball_weight > max_d:
                    max_d = ball_weight
                    idx_center = x_index
                    
            S.append(P[idx_center])
            
            ball2 = np.where(dist_matrix[idx_center] <(( 3 + 4 * alpha) * r))[0] #returns the indexes of the y points belonging to the B_z                                                                  centered in the center with idx_center as index and ray with (3+4alpha)*r
            
            for el in ball2:
                if P[el] in Z:
                   Z.remove(P[el])
                   W_z -= W[el]   
      
        if W_z <= z:
            print("Final guess: ", r)
            print("Number of guesses: ", n_guesses)
            return S
      
        else:
            r *= 2 
            n_guesses +=1

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(points, centers, z):
    random.seed(123)
    #print(points.count())
    l = math.floor(math.sqrt(points.count()))
    #objective = (points.map(lambda x: (x, random.randint(0,5))))
    objective = (points.map(lambda x: (x, random.randint(0,l-1)))             
    .groupByKey()
    .map(lambda x: (list(x[1])[0], x[0]))
    .map(lambda x: (min([euclidean(x[1],point2) for point2 in centers]), x[0]))
    .sortByKey(ascending=False)
    .take(z+1)[-1])
    
    return objective[0]

# Just start the main program
if __name__ == "__main__":
    main()

