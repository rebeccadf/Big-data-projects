import time
import sys
import math
import numpy as np


def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result
    
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)

def compute_dist(P): #to precompute the distances between points, since the matrix will be symmetric we only compute the "lower triangle"
    dist_matrix = np.zeros((len(P),len(P)), dtype = float)
    for i in range(len(P)):
        for j in range(i, len(P)):
            dist_matrix[j][i] = euclidean(P[i],P[j])
    print(dist_matrix)
    return dist_matrix

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

def ComputeObjective(P, S, z):
    distances = np.zeros(len(P)) 
    idx_centers = [P.index(c) for c in S] #indexes of the centers
    
    for i in range(len(P)): 
        min_d = dist_matrix[i][idx_centers[0]]
        for c in idx_centers[1:]:
            new_d = dist_matrix[i][c]
            if new_d < min_d:
                min_d = new_d
        distances[i] = min_d
    
    sorted_d = np.sort(distances) 
 
 
    if z > 0:
        return sorted_d[-(z+1)]
    else:
        return sorted_d[-1]
     
    
    
def main():
    filename = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    inputPoints = readVectorsSeq(filename)
    
    global dist_matrix
    dist_matrix = compute_dist(inputPoints)
    dist_matrix = dist_matrix + dist_matrix.T - np.diag(np.diag(dist_matrix)) #to fill the upper part of the matrix (we precomputed only the lower triangle)
  
    print("Input size n = ", len(inputPoints))
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    
    weights = np.ones(len(inputPoints))
    
    t_initial = time.time()
    solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0)
    t_final = time.time()
    objective = ComputeObjective(inputPoints, solution, z)

    print("Objective function = ", objective)
    print("Time of SeqWeightOutliers = ", (t_final-t_initial)*1000)


if __name__ == "__main__":
        main()
 

