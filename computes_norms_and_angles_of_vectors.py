import numpy as np
from numpy import linalg as LA

def calculate_norm_and_angles(ordered_array):
  coeff_norms = LA.norm(ordered_array,axis = 1) #Computes each vector norm, taking each row of the ordered array as a vector. Returns a 1D array with all the norms
  angles = np.array([])
  for i in range (0,len(ordered_array)-1,1): #From here it computes the angles
    for j in range (i+1,len(ordered_array),1):
      if coeff_norms[j] < 10**-6 or coeff_norms[i] < 10**-6:
        alpha = 0
        angles = np.append(angles, alpha)
      else:
        producto_escalar = np.dot(ordered_array[i,:],ordered_array[j,:])
        denominador = coeff_norms[i]*coeff_norms[j]
        cociente = producto_escalar/denominador
        alpha = np.arccos(cociente)
        print(producto_escalar, denominador, cociente, alpha)
        angles = np.append(angles, alpha)
  norm_and_angles = np.concatenate((coeff_norms,angles)) #Concatenates the two variables
  return norm_and_angles #Returns a 1D array with the norms and the angles

#TEST CODE TO CHECK THE NORMS AND ANGLES ARE CORRECT

test_vectors = list()
for i in range(0,3,1):
  test_vector = list((0,0,0))
  test_vector[i] = 1
  test_vectors.append(test_vector)

test_vectors = np.array(test_vectors).reshape((3,3))
norms_and_angles_test = calculate_norm_and_angles(test_vectors)
