#! /usr/bin/env python

import torch
import cpuinfo
import random

def inc_errors(funct):
    def wrapper(*args, **kwargs):
        global errors
        try:
            funct(*args, **kwargs)
        except Exception as e:
            print("ERROR: " + str(e), end = "\n\n")
            errors += 1
    return wrapper

@inc_errors
def test_tensors(device):
    global errors
    
    # Complex Double; Complex Single; Double; Single
    data_types = [torch.cdouble, torch.cfloat, torch.double, torch.float]
    
    for i in range(len(data_types)):
        matrix_size = random.randint(1, 8)
        matrix_a = torch.rand(matrix_size, matrix_size, dtype = data_types[i]).to(device)
        matrix_b = torch.rand(matrix_size, matrix_size, dtype = data_types[i]).to(device)
        
        print("RANDOM NUMBERS\n" + str(matrix_a), end = "\n\n")
        
        print("DOT PRODUCT")
        print(torch.matmul(matrix_a, matrix_b), end = "\n\n")
        
        print("TENSOR PRODUCT")
        print(torch.tensordot(matrix_a, matrix_b, dims = ([0, 1], [1, 0])), end = "\n\n")
        
        print("INVERSION") # Square matrices only
        print(torch.inverse(matrix_a), end = "\n\n")
        
        print("DETERMINANT")
        print(torch.det(matrix_a), end = "\n\n")
        
        # Linear algebra operations
        print("VECTOR NORM")
        print(torch.linalg.vector_norm(matrix_a), end = "\n\n")
        
        print("MATRIX NORM")
        print(torch.linalg.matrix_norm(matrix_a), end = "\n\n")
        
        print("SINGULAR VALUE DECOMPOSITION")
        U, S, V = torch.linalg.svd(matrix_a)
        print("U = " + str(U))
        print("S = " + str(S))
        print("V = " + str(V), end = "\n\n")
        
        print("EIGENVALUES AND EIGENVECTORS")
        L, V = torch.linalg.eig(matrix_a)
        print("L = " + str(L))
        print("V = " + str(V), end = "\n\n")

dev_names = []
dev_ids = []
dev_count = 0
errors = 0

if torch.cuda.is_available():
    dev_count = torch.cuda.device_count()
    print(f"Found {dev_count} compatible GPU", end = "")
    
    ## Add an "s" at the end if more than one GPU
    if dev_count != 1:
        print("s", end = "")
    print(".")
    
    ## Get device names and ID for each
    for i in range(dev_count):
        dev_names.append(torch.cuda.get_device_name(i))
        dev_ids.append(i)
        print(f"Device {dev_ids[i]}: {dev_names[i]}")
    
    ## Try to do matrix operations with different floating point types
    ## Some cards cannot perform half-precision, so we try to test all of them
    print("\nTesting tensors and placing them on GPUs...")
    for i in range(dev_count):
        test_tensors(dev_ids[i])
else:
    print("No compatible GPUs found, but PyTorch can still use the CPU, albeit slower.")
    print("CPU: " + cpuinfo.get_cpu_info()['brand_raw'] + "\n")
    test_tensors("cpu")

print(f"Diagnosis complete; {errors} runtime error" + ("s" if errors != 1 else "") + " occurred.")
