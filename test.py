# Python3 program for use   
# of insert() method  
import numpy as np

start1 = ["ATACGTGACA","ATGACGTGAC","ATGCTGACCT"]
start2 = ["AGACGTGTCA","ATGACTCGAT","CGTACGACCT"]

seq1 =start1[0]
seq2 =start2[0]
count=0



def rm_lcs(str1,str2):
  output =[]
  #make the bitmap array of strings
  for i in range (len(str1)):
     if(str1[i]==str2[i]):
        output.append(1)
     else: output.append(0)
  print(output)

  #now we have a bitmap of string match, get the index and length of LCS
  count = 0
  result = 0
  curr_count=0
  curr_idx=-1
  last_count=0
  last_idx=-1
  store = True
  for i in range (len(output)):
    
    if(output[i]==0):
      count=0
      store=True
      if(curr_count>=last_count):
        last_count=curr_count
        last_idx=curr_idx
      curr_count=0
      curr_idx=-1
      #reset the idx when you find zero
      
    else:
      if(store):
        curr_idx=i
        store = False
      curr_count+=1
      count+=1
      result = max(result, count)  
  
  # recheck. if the update is done
  if(curr_count>=last_count):
        last_count=curr_count
        last_idx=curr_idx
  
  res =[result,last_idx]
  # now the result has the max length
  print(result,last_idx)
  #now return the length and index
  return res



for i in range(len(start1)):
  print(rm_lcs(start1[i],start2[i]))


var1=list(start1[0])
var2 =list(start2[0])
indice = rm_lcs(start1[0],start2[0])
ar1=["".join(var1[0:indice[1]]),"".join(var1[(indice[1]):(indice[1]+indice[0])]),"".join(var1[(indice[1]+indice[0]):])]
print(indice)
print(ar1)

# making the list from string
name ="yash"
list1 = list(name)

surname ="patel"
list2 =list(surname)
#list1 = [ 1, 2, 3, 4, 5, 6, 7 ]  
  
# insert 10 at 4th index 
list2="".join(list2)

list1.insert(4, list2)  
print(list1)


lis ="".join(list1)
print(lis)

