# This is the code for the LSH project of TDT4305

import configparser  # for reading the parameters file
import sys  # for system errors and printouts
from pathlib import Path  # for paths of files
import os  # for reading the input data
import time  # for timing
import numpy as np  # for creating matrices or arrays
import random  # for randomly generating a and b for hash functions
from itertools import combinations  # for creating candidate pairs in lsh

# Global parameters
parameter_file = "default_parameters.ini"  # the main parameters file
data_main_directory = Path("data")  # the main path were all the data directories are
parameters_dictionary = (
    dict()
)  # dictionary that holds the input parameters, key = parameter name, value = value
document_list = (
    dict()
)  # dictionary of the input documents, key = document id, value = the document


# DO NOT CHANGE THIS METHOD
# Reads the parameters of the project from the parameter file 'file'
# and stores them to the parameter dictionary 'parameters_dictionary'
def read_parameters():
    config = configparser.ConfigParser()
    config.read(parameter_file)
    for section in config.sections():
        for key in config[section]:
            if key == "data":
                parameters_dictionary[key] = config[section][key]
            elif key == "naive":
                parameters_dictionary[key] = bool(config[section][key])
            elif key == "t":
                parameters_dictionary[key] = float(config[section][key])
            else:
                parameters_dictionary[key] = int(config[section][key])


# DO NOT CHANGE THIS METHOD
# Reads all the documents in the 'data_path' and stores them in the dictionary 'document_list'
def read_data(data_path):
    for root, dirs, file in os.walk(data_path):
        for f in file:
            file_path = data_path / f
            doc = open(file_path).read().strip().replace("\n", " ")
            file_id = int(file_path.stem)
            document_list[file_id] = doc


# DO NOT CHANGE THIS METHOD
# Calculates the Jaccard Similarity between two documents represented as sets
def jaccard(doc1, doc2):
    return len(doc1.intersection(doc2)) / float(len(doc1.union(doc2)))


# DO NOT CHANGE THIS METHOD
# Define a function to map a 2D matrix coordinate into a 1D index.
def get_triangle_index(i, j, length):
    if i == j:  # that's an error.
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    if j < i:  # just swap the values.
        temp = i
        i = j
        j = temp
    # Calculate the index within the triangular array. Taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # adapted for a 0-based index.
    k = int(i * (length - (i + 1) / 2.0) + j - i) - 1

    return k


# DO NOT CHANGE THIS METHOD
# Calculates the similarities of all the combinations of documents and returns the similarity triangular matrix
def naive():
    docs_Sets = []  # holds the set of words of each document

    for doc in document_list.values():
        docs_Sets.append(set(doc.split()))

    # Using triangular array to store the similarities, avoiding half size and similarities of i==j
    num_elems = int(len(docs_Sets) * (len(docs_Sets) - 1) / 2)
    similarity_matrix = [0 for x in range(num_elems)]
    for i in range(len(docs_Sets)):
        for j in range(i + 1, len(docs_Sets)):
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(
                docs_Sets[i], docs_Sets[j]
            )

    return similarity_matrix


# METHOD FOR TASK 1
# Creates the k-Shingles of each document and returns a list of them
def k_shingles():
    docs_k_shingles = []  # holds the k-shingles of each document
    k = parameters_dictionary["k"]

    for doc in document_list.values():
        words = doc.split()
        shingles = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
        docs_k_shingles.append(set(shingles))
    return docs_k_shingles


# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list

def signature_set(k_shingles):  # Optimized by using dictionaries

    # Find unique shingles and create a mapping from shingles to indices
    unique_shingles = sorted(set().union(*k_shingles))
    shingle_index_map = {
        shingle: index for index, shingle in enumerate(unique_shingles)
    }

    # Create the signature set
    docs_sig_sets = np.zeros((len(unique_shingles), len(k_shingles)))

    # Fill in the signature set using the mapping
    for j, doc in enumerate(k_shingles):
        for shingle in doc:
            docs_sig_sets[shingle_index_map[shingle], j] = 1

    return docs_sig_sets


# METHOD FOR TASK 3
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    for i in range(5, int(np.sqrt(n)) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True


# A function for generating hash functions
def generate_hash_functions(num_perm, N):
    hash_funcs = []
    
    # Find a prime number greater than N
    p = N + 1
    while not is_prime(p):
        p += 1

    for _ in range(num_perm):
        # Generate random coefficients 'a' and 'b'
        a = random.randint(1, N - 1)
        b = random.randint(1, N - 1)
        hash_func = lambda x, a=a, b=b, p=p: (a * x + b) % p
        hash_funcs.append(hash_func)
    return hash_funcs


# Creates the minHash signatures after generating hash functions
def minHash(docs_signature_sets, hash_fn):
    min_hash_signatures = np.full(
        (len(hash_fn), len(docs_signature_sets.T)), np.inf
    )  # Initialize the signature matrix with infinity

    for i, row in enumerate(docs_signature_sets):
        for j, col in enumerate(row):
            if col == 1:
                for k, hash_func in enumerate(hash_fn):
                    hashed_value = hash_func(i+1)
                    if hashed_value < min_hash_signatures[k][j]:
                        min_hash_signatures[k][j] = hashed_value
    return min_hash_signatures


# METHOD FOR TASK 4
# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents
def lsh(m_matrix):
    candidates = set()  # list of candidate sets of documents for checking similarity

    bands = parameters_dictionary["b"]
    rows = len(m_matrix) // bands

    # Print some information (for debugging)
    print(f"Number of bands: {bands}")
    print(f"Number of rows per band: {rows}")
    print(f"Length of signature matrix: {len(m_matrix)}")

    assert len(m_matrix) % rows == 0  # Number of rows must be divided equally

    # Initialize a dictionary to store buckets for each band
    buckets = {band_num: {} for band_num in range(bands)}
    for band_num in range(bands):
        start_row = band_num * rows
        end_row = start_row + rows
        band = m_matrix[start_row:end_row]
        # Generate hash keys for the band
        hash_keys = [hash(tuple(row)) for row in band.T]
        # Store documents in corresponding buckets
        for doc_index, hash_key in enumerate(hash_keys):
            if hash_key not in buckets[band_num]:
                buckets[band_num][hash_key] = []
            buckets[band_num][hash_key].append(doc_index)

    # Identify candidate pairs of documents
    for band_num in range(bands):
        for hash_key, docs in buckets[band_num].items():
            if len(docs) > 1:
                # Add pairs of documents as candidate pairs
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        doc_pair = (docs[i], docs[j])
                        candidates.add(doc_pair)
    return list(candidates)


# METHOD FOR TASK 5
def calculate_similarity(doc_tuple, min_hash_matrix):
    doc1_index, doc2_index = doc_tuple
    doc1_signature = min_hash_matrix[:, doc1_index]
    doc2_signature = min_hash_matrix[:, doc2_index]
    similarity = np.sum(doc1_signature == doc2_signature) / len(doc1_signature)
    return similarity


# Calculates the similarities of the candidate documents
def candidates_similarities(candidate_docs, min_hash_matrix):
    similarity_dict = []
    threshold = parameters_dictionary["t"]
    for doc_tuple in candidate_docs:
        similarity = calculate_similarity(doc_tuple, min_hash_matrix)
        if similarity >= threshold:
            similarity_dict.append({doc_tuple: similarity})
    return similarity_dict


# DO NOT CHANGE THIS METHOD
# The main method where all code starts
if __name__ == "__main__":
    # Reading the parameters
    read_parameters()

    # Reading the data
    print("Data reading...")
    print(data_main_directory / parameters_dictionary["data"])
    data_folder = data_main_directory / parameters_dictionary["data"]
    t0 = time.time()
    read_data(data_folder)
    document_list = {k: document_list[k] for k in sorted(document_list)}
    t1 = time.time()
    print(len(document_list), "documents were read in", t1 - t0, "sec\n")

    # Naive
    naive_similarity_matrix = []
    if parameters_dictionary["naive"]:
        print("Starting to calculate the similarities of documents...")
        t2 = time.time()
        naive_similarity_matrix = naive()
        t3 = time.time()
        print(
            "Calculating the similarities of",
            len(naive_similarity_matrix),
            "combinations of documents took",
            t3 - t2,
            "sec\n",
        )

    # k-Shingles
    print("Starting to create all k-shingles of the documents...")
    t4 = time.time()
    all_docs_k_shingles = k_shingles()
    t5 = time.time()
    print("Representing documents with k-shingles took", t5 - t4, "sec\n")

    # signatures sets
    print("Starting to create the signatures of the documents...")
    t6 = time.time()
    signature_sets = signature_set(all_docs_k_shingles)
    t7 = time.time()
    print("Signatures representation took", t7 - t6, "sec\n")

    # Permutations
    print("Starting to simulate the MinHash Signature Matrix...")
    t8 = time.time()
    hash_fn = generate_hash_functions(
        parameters_dictionary["permutations"], len(signature_sets)
    )
    min_hash_signatures = minHash(signature_sets, hash_fn)
    t9 = time.time()
    print("Simulation of MinHash Signature Matrix took", t9 - t8, "sec\n")

    # LSH
    print("Starting the Locality-Sensitive Hashing...")
    t10 = time.time()
    candidate_docs = lsh(min_hash_signatures)
    t11 = time.time()
    print("LSH took", t11 - t10, "sec\n")

    # Return the over t similar pairs
    print(
        "Starting to get the pairs of documents with over ",
        parameters_dictionary["t"],
        "% similarity...",
    )
    t14 = time.time()
    true_pairs = candidates_similarities(candidate_docs, min_hash_signatures)
    t15 = time.time()
    print(f"The total number of candidate pairs from LSH: {len(candidate_docs)}")
    print(f"The total number of true pairs from LSH: {len(true_pairs)}")
    print(
        f"The total number of false positives from LSH: {len(candidate_docs) - len(true_pairs)}"
    )

    if parameters_dictionary["naive"]:
        print("Naive similarity calculation took", t3 - t2, "sec")

    print("LSH process took in total", t14 - t15, "sec")

    print("The pairs of documents are:\n")
    for p in true_pairs:
        print(
            f"LSH algorithm reveals that the BBC article {list(p.keys())[0][0]+1}.txt and {list(p.keys())[0][1]+1}.txt \
              are {round(list(p.values())[0],2)*100}% similar"
        )

        print("\n")
