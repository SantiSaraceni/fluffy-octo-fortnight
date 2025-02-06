import h5py
import numpy as np
import pandas as pd
from numpy import linalg as LA

def main():
  ds = h5py.File("diatomics-exc.h5")
  g002 = ds["002"]
  coeficientes =g002["coefficients"]
  especies = g002["species"]
  coordenadas = g002["coordinates"]
  vector_atomos = [[1,1],[9,9],[17,17],[1,9],[1,17]] #Defino variables que voy a necesitar para escribir los archivos .csv
  nombres = ["H2","F2","Cl2","HF","HCl"]
  tipos = ["s","p","d","pd"]
  devuelve_archivos_csv_coefs_SIN_transf_molecula(vector_atomos, nombres,g002)
  devuelve_archivos_csv_coefs_transf_molecula(vector_atomos,nombres,g002)
  devuelve_archivos_csv_coefs_transf_tipo(vector_atomos,nombres,tipos,g002)
  devuelve_archivos_csv_coefs_SIN_transf_tipo(vector_atomos,nombres,tipos,g002)

#Returns the indices where "iteration_idx" is 0
def get_iteration_indices(grupo):
  return np.where(np.asarray(grupo["iteration_idx"]) == 0)[0] #First transforms the dataset into an array, then, when it applies the condition "== 0" we get a boolean array that acts as input for the np.where method. Because we are only passing the condition, this function will return a tuple with arrays as elements. Each array has the indices of the elements that meet the condition

#Returns a subset of the coefficients dataset, with all the coefficients being part of a 0 iteration, the specific type of molecule and the specific type of function

def devuelve_array_coeficientes(nros_atomicos, tipo,grupo):
  coeficientes = grupo["coefficients"]
  especies = grupo["species"]
  indices_iteraciones_0 = get_iteration_indices(grupo)
  mask = np.where(np.all(especies[indices_iteraciones_0] == nros_atomicos, axis=1))
  selected_indices = indices_iteraciones_0[mask]

  if tipo == "s":
    column_slice = slice(0, 9)
  elif tipo == "p":
    column_slice = slice(9, 21)
  elif tipo == "d":
    column_slice = slice(21, 45)
  elif tipo == "none":
    column_slice = slice(0, 45)
  elif tipo == "pd":
    column_slice = slice(9,45)
  else:
    raise ValueError("Invalid 'tipo' value. Must be 's', 'p', 'd','pd' or 'none'.")

  return coeficientes[selected_indices, :, column_slice]

#This function returns a 2d array with all the coefficents order in a specific manner

def ordena_coeficientes_para_transformar(tipo, array_coeficientes_desordenado):
    if tipo == "p":
        coeficientes_reordenados = array_coeficientes_desordenado.reshape((-1, 2, 4, 3))
    elif tipo == "d":
        coeficientes_reshepeados = array_coeficientes_desordenado.reshape((-1, 2, 4, 6))

        ii_indices = [0, 2, 5]
        ij_indices = [1, 3, 4]

        array_ii = coeficientes_reshepeados[:, :, :, ii_indices]
        array_ij = coeficientes_reshepeados[:, :, :, ij_indices]

        # Concatenate along axis=2 to get (1000, 2, 8, 3)
        coeficientes_reordenados = np.concatenate((array_ii, array_ij), axis=2)

    elif tipo == "pd":
      array_p = array_coeficientes_desordenado[:,:,0:12].reshape((-1, 2, 4, 3))
      array_d = array_coeficientes_desordenado[:,:,12:36].reshape((-1, 2, 4, 6))

      ii_indices = [0, 2, 5]
      ij_indices = [1, 3, 4]

      array_ii = array_d[:, :, :, ii_indices]
      array_ij = array_d[:, :, :, ij_indices]

      coeficientes_d_reordenados = np.concatenate((array_ii, array_ij), axis=2)

      coeficientes_reordenados = np.concatenate((array_p,coeficientes_d_reordenados),axis = 2)
    else:
        raise ValueError("Invalid 'tipo' value. Must be 'p', 'd' or 'pd'.")

    return coeficientes_reordenados

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
        angles = np.append(angles, alpha)
  norm_and_angles = np.concatenate((coeff_norms,angles)) #Concatenates the two variables
  return norm_and_angles #Returns a 1D array with the norms and the angles

#Ahora calculamos los coeficientes transformados
def calcula_coefs_transformados(array_coeficientes_reordenados):

    _,n_atoms,_,_ = array_coeficientes_reordenados.shape
    coeficientes_transformados = []
    for atom in range(n_atoms):
      coeficientes_transformados_atom = []
      for molecule in array_coeficientes_reordenados:
        coeficientes_transformados_atom.append(calculate_norm_and_angles(molecule[atom]))
      coeficientes_transformados.append(coeficientes_transformados_atom) #Lista que tiene 2 elementos, donde el primer elemento es una lista de arrays con otods los coeficicentes de un cierto tipo de cada atomo de H de cada molecula y el 2do elemento de la lista general corresponde a una lista de arrays con los coeficientes de los atomos de F de cada molecula
    return coeficientes_transformados

#Este codigo me arma los archivos .csv con los coeficientes transformados para cada tipo de molecula, donde cada columna corresponde a los coeficientes sin transformar de cada una de las estructuras.

def devuelve_archivos_csv_coefs_SIN_transf_molecula(vector_atomos,nombres,grupo):
  for i in range(0,len(vector_atomos),1):
    array_coeficientes = devuelve_array_coeficientes(vector_atomos[i],"none",grupo)
    all_molecules = []  # Create an empty list to store the Series
    for j in range(0, len(array_coeficientes), 1):
      molecule = array_coeficientes[j]
      molecula_reshaped = np.reshape(molecule, molecule.size)
      all_molecules.append(pd.Series(molecula_reshaped, name=f"{j}"))  # Create a Series and add it to the list
    df = pd.concat(all_molecules, axis=1)  # Concatenate all Series into a DataFrame
    df.to_csv(f"coefs_SIN_transf_{nombres[i]}_completos.csv")

#Este codigo me arma los archivos .csv con los coeficientes transformados para cada tipo de molecula, donde cada columna corresponde a los coeficientes sin transformar de cada una de las estructuras.

def devuelve_archivos_csv_coefs_transf_molecula(vector_atomos,nombres,grupo):
  for i in range(0,len(vector_atomos),1):
    array_coeficientes_mlc_total = devuelve_array_coeficientes(vector_atomos[i],"none",grupo)
    array_coeficientes_mlc_total_s = array_coeficientes_mlc_total[:,:,0:9]
    array_coeficientes_mlc_total_pd = array_coeficientes_mlc_total[:,:,9:45]
    coefs_transformados_totales = calcula_coefs_transformados(ordena_coeficientes_para_transformar("pd",array_coeficientes_mlc_total_pd))
    all_molecules = []  # Create an empty list to store the Series
    array_0 = coefs_transformados_totales[0]
    array_1 = coefs_transformados_totales[1]
    for j in range(0, len(array_coeficientes_mlc_total), 1):
      coefs_transformados_mlc = np.concatenate((array_0[j], array_1[j]))
      molecule_coefs_s = array_coeficientes_mlc_total_s[j]
      molecula__coefs_s_reshaped = np.reshape(molecule_coefs_s, molecule_coefs_s.size)
      coeficientes_finales_molecula = np.concatenate((molecula__coefs_s_reshaped, coefs_transformados_mlc))
      all_molecules.append(pd.Series(coeficientes_finales_molecula, name=f"{j}"))  # Create a Series and add it to the list
    df = pd.concat(all_molecules, axis=1)  # Concatenate all Series into a DataFrame
    df.to_csv(f"coefs_transf_{nombres[i]}_completos.csv")

#Este codigo me arma los archivos .csv con los coeficientes transformados para cada tipo de molécula y tipo de funcion (p,d,pd) con los coeficientes de un mismo tipo de todos los atomos de todas las moleculas (no entiendo si tiene que quedar como el de abajo cada estructura en una columna o directamente todas las estructuras en una misma columna)

def devuelve_archivos_csv_coefs_transf_tipo(vector_atomos,nombres,tipo,grupo):
  tipo_1 = tipo[1:]
  for i in range(0,len(vector_atomos),1):
    for j in tipo_1:
      array_coeficientes = devuelve_array_coeficientes(vector_atomos[i],j,grupo)
      array_coeficientes = ordena_coeficientes_para_transformar(j,array_coeficientes)
      coefs_transformados = calcula_coefs_transformados(array_coeficientes)
      array_0 = coefs_transformados[0]
      array_1 = coefs_transformados[1]
      all_molecules = []
      for k in range(0,len(array_0),1):
          coefs_transformados_mlc = np.concatenate((array_0[k], array_1[k]))
          all_molecules.append(pd.Series(coefs_transformados_mlc, name=f"{k}"))  # Create a Series and add it to the list
      df = pd.concat(all_molecules, axis=1)  # Concatenate all Series into a DataFrame
      df.to_csv(f"coefs_transf_{nombres[i]}_{j}_pormlc.csv")

#Este codigo me arma los archivos .csv para cada tipo de molécula y tipo de funcion (s,p,d) con los coeficientes de un mismo tipo de todos los atomos de todas las moleculas (no entiendo si tiene que quedar como el de abajo cada estructura en una columna o directamente todas las estructuras en una misma columna)

def devuelve_archivos_csv_coefs_SIN_transf_tipo(vector_atomos,nombres,tipo,grupo):
  tipo_corregido = tipo[0:len(tipo)-1]
  for i in range(0,len(vector_atomos),1):
    for j in tipo:
      array_coeficientes = devuelve_array_coeficientes(vector_atomos[i],j,grupo)
      all_molecules = []
      for k in range(0, len(array_coeficientes), 1):
        molecule = array_coeficientes[k]
        molecula_reshaped = np.reshape(molecule, molecule.size)
        all_molecules.append(pd.Series(molecula_reshaped, name=f"{k}"))  # Create a Series and add it to the list
      df = pd.concat(all_molecules, axis=1)  # Concatenate all Series into a DataFrame
      df.to_csv(f"coefs_SIN_transf_{nombres[i]}_{j}_pormlc.csv")

if __name__ == "__main__":
  main()

