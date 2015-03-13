##Sparse Matrix-Vector Multiplication in parallel with MPI

###Design
Parallelizing the sparse matrix-vector multiplication using MPI: 

1. Reading in the files and distributing the data to all processors in Step 1 using a 1D rows decomposition, this takes O(n) and then O(nnz) where n is the number of rows, nnz is the number of non-zeros in the matrix. Matrix A data is read in and stored in the CSR format, which includes three arrays: row pointers, column index of non-zeros and matrix values. Step 1 also involves data distribution to p processors with MPI Bcast.

2. Each process prepares to get the non-local Vector elements it needs using the prepareRemoteVec function. Here it goes through the local Column Index of the Matrix and checks what the remote Vector entries needed are, then resize the local Vector Data array (vSize + numRemoteVec) to hold these additional remote vector entries from other processors at the end of the array. Finally we need re-index the local Column Index array (which previously points to the global Vector Data index). We go through the local Column Index array and have them pointed to the right local Vector Data indexes (either local elements, at the front or remote elements, at the back of the Vector Data array, with appropriate offsets). This ensures that accessing the received elements of Vector Data costs the same as accessing the local vector elements, O(1). Additionally, this is memory scalable as we only use exactly as much memory as required to store the needed Vector elements for multiplication. (vSize + numRemoteVec). This steps only takes O(n/p).

3. Knowing the local vector size each processor received, all processors can find out which section of the global Vector B everyone handle, by just looking at the ranks and vector size. Therefore, it can easily figure out which processors hold which needed remote vector index. So 2 MPI Alltoallv operations are used for all processors to tell every others the number of vector index they need and then which particular set of vector index they need. Now each processor has an array of vector index that they need to send out (tosendVecIndex), they look at their locally stored vector array to get these data ready to send.

4. Now all processors perform the actual transfer of the vector data with another MPI Alltoallv operation. Each processor send personalized data from their sendVecData array to all others, who store it in their reallocated local vector array.

5. At this point, the non-local elements of Vector B that each processor needs, are all stored in the local vector array and the adjusted (in Step 2) local Column Index is pointing to the right index there. So accessing the non-local received vector elements are straightforward. Each process now computes the results, using a normal CSR matrix-vector multiplication function, costing O(2nnz) for nnz multiplications and nnz-1 additions. Partial
results are stored. O(2n)

6. All processors gather their results to the root processor using MPI Gatherv operation. The root processor store this in a totalResult array and print them out to the output file in O(n) time.
