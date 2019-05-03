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


def dot_plot(str1,str2):
    D = np.zeros((len(str1),(len(str2))))
    for i in range(0,len(str1)):
        for j in range (0,len(str2)):
            if(str1[i]==str2[j]):
                D[i,j]=1
                plot =plt.scatter(i,j)
            else:
                D[i,j]=0
    #axes.invert_yaxis()   
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
 
 
    return align_X, align_Y


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
# kmp(filename2, filename2) 



#dna_seq_generator(10000,"seq1.txt")
#dna_seq_generator(10000,"seq2.txt")


#print(brute_force_search("seq1.txt","seq2.txt"))
kmp("seq1.txt","seq2.txt")
print(get_string_alignment("ATAGCTATAGCAT","ACCTACGGATCGT"))
