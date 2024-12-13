def calculate_norm_and_angles(ordered_array):
  coeff_norms = LA.norm(ordered_array,axis = 1) #Calcula la norma cada vector, tomando como vector a cada una de las filas del array ordenado. Devuelve un array 1D con todas las normas.
  angulos = np.array([])
  for i in range (0,len(ordered_array)-1,1): #A partir de aca calcula los angulos
    for j in range (i+1,len(ordered_array),1):
      if coeff_norms[j] < 10**-6 or coeff_norms[i] < 10**-6:
        alfa = 0
        angulos = np.append(angulos, alfa)
      else:
        alfa = np.arccos(np.dot(ordered_array[i,:],ordered_array[j,:])/(coeff_norms[i]*coeff_norms[j]))
        angulos = np.append(angulos, alfa)
  norm_and_angles = np.concatenate((coeff_norms,angulos)) #Concatena las dos variables
  return norm_and_angles #Devuelve un array 1D con las normas y los angulos.
