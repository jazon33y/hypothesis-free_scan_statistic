import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import beta

'''
Python implementation of the hypothesis-free multiple scan statistic for
detecting clusters in point processes.  The algorithm uses a Monte
Carlo procedure to compute p-values.  This task becomes computationally
inefficient as a squared function of the number of unique points.  This
limitation can be avoided by pre-computing the null statistic distri-
butions, and querying the resultant table at the pre-specified alpha level.

Example useage:

    data = [70586162,70586165,70586190,70586196,70586199,70586201,70586367,70586574,70586631,
            70586652,70586660,70586664,70586670,70586691,70586996,70587002,70587020,70587032,
            70587053,70587066,70587117,70587200,70587422,70587498,70587561,70587641,70587663,
            70587675,70587704,70587771,70587804,70588077,70588319,70588386,70588500,70589169,
            70589274,70589498,70590081,70590360,70590401,70590420,70590447,70590825,70590864,
            70590869,70590953,70591142,70591185,70591215,70591248,70591278,70591308,70591337,
            70591434,70591441,70591442,70591445,70591530,70591656,70591720,70591720,70591823,
            70587032,70587066,70588319,70590864,70590953,70591530,70586196,70586367,70586652,
            70587032,70587053,70587117,70587675,70587704,70591720]

    initial_cluster = get_clusters(data, type='min')
    all_clusters = get_additional_clusters(data, initial_cluster, type='max', permutations=1000)

    all_clusters
'''

def correct_intervals(data): # just returns elements of all possible correct intervals, assumes input is sorted, small to large
  a = data
  n = len(data)
  a = range(0,n)
  b = range(0,n)
  c = []
  d = []
  for i in range(0,len(a)):
    b.pop(0)
    a.pop()
    c.extend(a)
    d.extend(b)
  X_i = c
  X_j = d
  return c,d

def normalize(a): # normalize to [0, 1]
  a = sorted(map(float,a))
  normalized_a = []
  y = min(a)
  z = max(a)
  normalized_a = map(lambda x: (x - y) / (z - y), a)
  return normalized_a

def ibf(x, a, b): # incomplete beta function, *VECTORIZED*
  return beta.cdf(x = x, a = a,  b = b) * special.beta(a, b)

def transform(cluster_location, data, verbose = False): # transform data around significant cluster
  i = cluster_location[0]
  j = cluster_location[1]
  n = len(data)
  t = 1 - data[j] + data[i]
  new_data = []
  for k in range(n):
    if ((0 <= k) and (k <= i)): new_data.append( data[k] / t )
    elif (( (i + 1) <= k) and (k <= ((n-1) - j + i) )): new_data.append((data[k+j-i] - data[j] + data[i]) / t) # n-1 important here because len(data) is greater than the numeral of the last element in data, i.e., if len(data) = 10, data[10] will fail!!!
    elif verbose == True: print 'repositioning data: {', k, ':', data[k], '}\n'
  return new_data

def get_clusters(data, type="min", stat_dist=False): # cluster detection
  if len(data) < 3:
    print 'Need more than two data points'
  if stat_dist==True: data = list( np.random.uniform(0,1,len(data)) ) #
  data = list( set(data) ) # remove duplicated values
  n = len(data)
  data = sorted(data)
  data = normalize(data)

  #get all correct intervals
  i,j = correct_intervals(data)
    
  #compute the statistic
  I_H_F = float(1) / ibf((np.array(data)[j] - np.array(data)[i]), np.array(j) - np.array(i), (n) + 1 - np.array(j) + np.array(i))
  I_H_F = list(I_H_F)
  I_H_F_max_loc = I_H_F.index(max(I_H_F))
  I_H_F_min_loc = I_H_F.index(min(I_H_F))
  
  full_data_max = {'X_i':[data[i[I_H_F_max_loc]]],
           'X_j':[data[j[I_H_F_max_loc]]],
           'i':[i[I_H_F_max_loc]],
           'j':[j[I_H_F_max_loc]],
           'I_H_F':[I_H_F[I_H_F_max_loc]]}

  full_data_min = {'X_i':[data[i[I_H_F_min_loc]]],
           'X_j':[data[j[I_H_F_min_loc]]],
           'i':[i[I_H_F_min_loc]],
           'j':[j[I_H_F_min_loc]],
           'I_H_F':[I_H_F[I_H_F_min_loc]]}
  
  if type=='max': return full_data_max 
  if type=='min': return full_data_min

def get_additional_clusters(data, clust_table, type='min', permutations=1000,p=.05):
  def get_ps(data,clust_table,permutations):
    if len(data) < 3 : return clust_table
    else:
      L = []
      for i in range(permutations):
        table = get_clusters(data, stat_dist=True)
        L.append(table['I_H_F'][-1])

      L = sorted(L)
      if type=='max': A_H_F_p = 1. - sum(((np.array(L)<clust_table["I_H_F"][-1]))/len(L))
      if type=='min': A_H_F_p = sum(1.*((np.array(L)<clust_table["I_H_F"][-1]))/len(L))
      clust_table['ps'][-1] = A_H_F_p
      return clust_table
  
  data = list(set(data))
  data = sorted(data)
  data = normalize(data)
  clust_table['ps'] = [float(0.0)]
  clust_table = get_ps(data,clust_table,permutations)
  while (clust_table['ps'][-1] < p):
    i = clust_table['i'][-1]
    j = clust_table['j'][-1]
    data = transform([i,j],data)
    new_clust_table = get_clusters(data)
    new_clust_table['ps'] = [float(0.0)]
    new_clust_table_with_ps = get_ps(data,new_clust_table,permutations)
    for ii in new_clust_table_with_ps.keys():
      clust_table[ii].extend(new_clust_table_with_ps[ii])            
  return( pd.DataFrame(clust_table) )