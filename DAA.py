#import all the libraries
import string
import random
import numpy as np
import pandas as pd
import matplotlib as plt


#generates the sequence of some input length with delimiter as space
def dna_seq_generator(length,filename):
    fh = open(filename, "a+")
    for i in range (length):
        letter = random.choice('ATCG')
        fh.write(letter+" ")#put in quotes your own delimiter
    fh.close()


#print the array by taking low as starting index
def print_arr_range(arr,low,length):
    for i in range(length):
        print(arr[i + low])


# a simple brute force search algo to find match
def brute_force_search(filename1,filename2):
    seq1 = pd.read_csv(filename1, delimiter = " " , header = None)
    seq2 = pd.read_csv(filename2, delimiter = " " , header = None)
    seq1=seq1.iloc[0]# now the data can be accesed by putting index after seq1 such as seq1[i]
    seq2=seq2.iloc[0]#now the data can be accesed by putting index after seq2 such as seq2[i]
    for i in range (max(len(seq1),len(seq2))):
        j=0
        while(j<len(seq2) and (seq2[j]==seq1[i+j])):
            j = j+1
        if(j==len(seq2)):
            print_arr_range(seq1,i,len(seq2))
            return i
    return -1


def dot_plot(filename1,filename2):
    str1 = pd.read_csv(filename1, delimiter = " " , header = None)
    str2 = pd.read_csv(filename2, delimiter = " " , header = None)
    str1=str1.iloc[0]# now the data can be accesed by putting index after seq1 such as seq1[i]
    str2=str2.iloc[0]#now the data can be accesed by putting index after seq2 such as seq2[i]

    D = np.zeros((len(str1),(len(str2))))
    for i in range(0,len(str1)):
        for j in range (0,len(str2)):
            if(str1[i]==str2[j]):
                D[i,j]=1
                plot =plt.scatter(i,j)
            else:
                D[i,j]=0
    plt.show()
    return D


def LCS(str1,str2):
    #initialization
    D = np.zeros((len(str1),len(str2)))
    D[0][0]=0
    for i in range(0,len(str1)):
        D[i][0]=0
    for j in range(0,len(str2)):
        D[0][j]=0
    # recursion
    for i in range (0,len(str1)):
        for j in range(0,len(str2)):
            if(str1[i]==str2[j]):
                D[i][j]=D[i-1][j-1]+1
            else:
                D[i][j]=0
    return D

def get_score(n1, n2, penalty = -1, reward = 1):

    if n1 == n2:
        return reward
    else:
        return penalty

    # score matrix (initialization)
    score_matrix = np.ndarray((len(X) + 1, len(Y) + 1))

    for i in range(len(X) + 1):
        score_matrix[i, 0] = penalty * i
    for j in range(len(Y)+1):
        score_matrix[0,j] =panalty*j

    # get each box in matric by its max score
    for i in range(1, len(X) + 1):
        for j in range(1, len(Y) + 1):
            match = score_matrix[i - 1, j - 1] + get_score(X[i - 1], Y[j - 1], penalty, reward)
            delete = score_matrix[i -1, j] + penalty
            insert = score_matrix[i, j - 1] + penalty
            score_matrix[i, j] = max([match, delete, insert])

def get_score(n1, n2, penalty = -1, reward = 1):

    if n1 == n2:
        return reward
    else:
        return penalty

def get_string_alignment(X, Y, penalty = -1, reward = 1):

    # initialize score matrix
    score_matrix = np.ndarray((len(X) + 1, len(Y) + 1))

    for i in range(len(X) + 1):
        score_matrix[i, 0] = penalty * i

    for j in range(len(Y) + 1):
        score_matrix[0, j] = penalty * j


    # define each cell in the matrix by as the max score possible in that stage
    for i in range(1, len(X) + 1):
        for j in range(1, len(Y) + 1):
            match = score_matrix[i - 1, j - 1] + get_score(X[i - 1], Y[j - 1], penalty, reward)
            delete = score_matrix[i -1, j] + penalty
            insert = score_matrix[i, j - 1] + penalty

            score_matrix[i, j] = max([match, delete, insert])


    i = len(X)
    j = len(Y)

    align_X = ""
    align_Y = ""

    while i > 0 or j > 0:

        current_score = score_matrix[i, j]
        left_score = score_matrix[i - 1, j]


        if i > 0 and j > 0 and X[i - 1] == Y[j - 1]:
            align_X = X[i - 1] + align_X
            align_Y = Y[j - 1] + align_Y
            i = i - 1
            j = j - 1

        elif i > 0 and current_score == left_score + penalty:
            align_X = X[i - 1] + align_X
            align_Y = "-" + align_Y
            i = i - 1

        else:
            align_X = "-" + align_X
            align_Y = Y[j - 1] + align_Y
            j = j - 1


    return [align_X, align_Y]

def get_string_alignment_filename(filename1, filename2, penalty = -1, reward = 1):
    X = pd.read_csv(filename1, delimiter = " " , header = None)
    Y = pd.read_csv(filename2, delimiter = " " , header = None)
    X=X.iloc[0]# now the data can be accesed by putting index after seq1 such as seq1[i]
    Y=Y.iloc[0]#now the data can be accesed by putting index after seq2 such as seq2[i]
    X="".join(X)
    Y="".join(Y)


    # initialize score matrix
    score_matrix = np.ndarray((len(X) + 1, len(Y) + 1))

    for i in range(len(X) + 1):
        score_matrix[i, 0] = penalty * i

    for j in range(len(Y) + 1):
        score_matrix[0, j] = penalty * j


    # define each cell in the matrix by as the max score possible in that stage
    for i in range(1, len(X) + 1):
        for j in range(1, len(Y) + 1):
            match = score_matrix[i - 1, j - 1] + get_score(X[i - 1], Y[j - 1], penalty, reward)
            delete = score_matrix[i -1, j] + penalty
            insert = score_matrix[i, j - 1] + penalty

            score_matrix[i, j] = max([match, delete, insert])


    i = len(X)
    j = len(Y)

    align_X = ""
    align_Y = ""

    while i > 0 or j > 0:

        current_score = score_matrix[i, j]
        left_score = score_matrix[i - 1, j]


        if i > 0 and j > 0 and X[i - 1] == Y[j - 1]:
            align_X = X[i - 1] + align_X
            align_Y = Y[j - 1] + align_Y
            i = i - 1
            j = j - 1

        elif i > 0 and current_score == left_score + penalty:
            align_X = X[i - 1] + align_X
            align_Y = "-" + align_Y
            i = i - 1

        else:
            align_X = "-" + align_X
            align_Y = Y[j - 1] + align_Y
            j = j - 1


    return [align_X, align_Y]

# a recursive method for splitting the sequence in some number of parts

def split_str(seq, chunk, skip_tail=False):
    lst = []
    if chunk <= len(seq):
        lst.extend([seq[:chunk]])
        lst.extend(split_str(seq[chunk:], chunk, skip_tail))
    elif not skip_tail and seq:
        lst.extend([seq])
    return lst
#use example
#seq = "ATCGGTACCTGAATGC"
#print(split_str(seq, 3))



# defination of kmp algo: -> O(n) for perfect string match
def kmp(filename1, filename2):

    string1 = pd.read_csv(filename2, delimiter = " " , header = None)
    string2 = pd.read_csv(filename1, delimiter = " " , header = None)
    string1=string1.iloc[0]# now the data can be accesed by putting index after seq1 such as seq1[i]
    string2=string2.iloc[0]#now the data can be accesed by putting index after seq2 such as seq2[i]
    i = len(string1)
    j = len(string2)

    lm = [0]*i
    q = 0


    get_lps(string1, i, lm)

    p = 0
    while p < j:
        if string1[q] == string2[p]:
            p += 1
            q += 1

        if q == i:
            print ("Match at index "+ str(p-q))
            q = lm[q-1]

        elif p < j and string1[q] != string2[p]:
            if q != 0:
                q = lm[q-1]
            else:
                p += 1

def get_lps(string1, i, lm):
    length = 0
    lm[0]
    p = 1
    while p < i:
        if string1[p]== string1[length]:
            length += 1
            lm[p] = length
            p += 1
        else:
            if length != 0:
                length = lm[length-1]
            else:
                lm[p] = 0
                p += 1
# testing the algorithm
# string2 = "ABABDABACDABABCABAB"
# string1 = "ABABCABAB"
#kmp("seq1.txt", "seq2.txt")


def get_lcs_index(str1,str2):
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

  res =[last_idx,result]
  # now the result has the max length
  #now return the length and index
  return res

def divideonlcs(array,indice):
    var1= list(array)
    if(indice[0] is not (-1)):
        result = ["".join(var1[0:indice[0]]),"".join(var1[(indice[0]):(indice[0]+indice[1])]),"".join(var1[(indice[0]+indice[1]):])]
    else:
        result = ["".join(var1),"",""]
    return result

def hybrid(filename1,filename2,chunks):
    #splitting the sequence
    seq1 = pd.read_csv(filename1,delimiter=" ",header=None)
    seq1 = seq1.iloc[0]
    seq1 =np.array(seq1)
    seq1=''.join(seq1)
    seq2 = pd.read_csv(filename2,delimiter=" ",header=None)
    seq2 = seq2.iloc[0]
    seq2 =np.array(seq2)
    seq2=''.join(seq2)
    #splitstring divide the code recursively
    split_string1=split_str(seq1,chunks)
    split_string2=split_str(seq2,chunks)
    print(split_string1)
    print(split_string2)
    # call the remove element function and then merge it over there only
    lcs_indices=[]
    for i in range (len(split_string1)):
        lcs_indices.append(get_lcs_index(split_string1[i],split_string2[i]))
    print (lcs_indices)

    #divide strings based on index
    splitted_arr1=[]
    splitted_arr2=[]
    for j in range (len(lcs_indices)):
        splitted_arr1.append(divideonlcs(split_string1[j],lcs_indices[j]))
        splitted_arr2.append(divideonlcs(split_string2[j],lcs_indices[j]))
    print(splitted_arr1)
    print (splitted_arr2)

    #now perform needleman wunch on the first and last index
    aligned_arr1=[]
    aligned_arr2=[]
    for k in range (len(splitted_arr1)):
        s1=splitted_arr1[k][0]
        s2=splitted_arr2[k][0]
        s3=splitted_arr1[k][2]
        s4=splitted_arr2[k][2]
        if(len(s1) is not 0):
             temp =get_string_alignment(s1,s2)
             aligned_arr1.append(temp[0])
             aligned_arr1.append(splitted_arr1[k][1])
             aligned_arr2.append(temp[1])
             aligned_arr2.append(splitted_arr2[k][1])
        if(len(s3) is not 0):
            temp1=get_string_alignment(s3,s4)
            aligned_arr1.append(temp1[0])
            aligned_arr2.append(temp1[1])
    print(aligned_arr1)
    print (aligned_arr2)
    final_str1="".join(aligned_arr1)
    final_str2="".join(aligned_arr2)
    print(final_str1)
    print (final_str2)
    return 0




#dna_seq_generator(100000,"seq1.txt")
#dna_seq_generator(100000,"seq2.txt")

#===========================================================================================================#
#print(brute_force_search("seq1.txt","seq2.txt"))
#print("Brute force done\n")
#kmp("seq1.txt","seq2.txt")
#print("kmp done \n")
#print(get_string_alignment("ATAGCTATAGCAT","ACCTACGGATCGT"))
print(get_string_alignment_filename("ht1.txt","ht2.txt"))
hybrid("ht1.txt","ht2.txt",20)
#print(split_str("ATACGTGACGTGACGATAAACGATGACGGATACGATGACAGTACCCAGATCAGATACCCGATGAACGTGTGACGGATCVGTA", 8))


# #splitting the sequence
# seq = pd.read_csv("ht1.txt",delimiter=" ",header=None)
# seq = seq.iloc[0]
# seq =np.array(seq)
# seq=''.join(seq)
# print(split_str(seq,20))

