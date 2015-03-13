#include <mpi.h>
#include "util.h"

// *** Original Implementation of Matvect using Sort + BSearch for Remote Vector Data *** //

int sortGroupVec(int *colInd, int numE, int myid, int numprocs, int nrows, 
		int *remoteVecCount, int *remoteVecPtr, int **remoteVecIndex); 
int compare(const void * a, const void * b);
void computeMatVec(int n, int *rowptr, int *colind, double *matval, double *vec, double *res);

int main(int argc, char *argv[]) {
	int num_procs, myid, name_len, p=0, i=0, nrows, vecSize, myNumE, vSize=0, vSize2=0; 
	double *vecData, *myMatVal, *result, *totalResult, *sendVecData;
	int *myColInd, *myRowptr;
	double timer56, timerTotal;

	char proc_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Get_processor_name(proc_name, &name_len);

	int remoteVecCount[num_procs], remoteVecPtr[num_procs];
	int tosendVecCount[num_procs], tosendVecPtr[num_procs];
	int tosendTotal = 0, numVecTotal = 0;
	int *remoteVecIndex, *tosendVecIndex;

	// First Process - Input & Output / Distribute Data & Receive Results / Compute
	if ( myid == 0 ) {
		size_t vecLen;
		char *vec=NULL; char *colI=NULL,*rowI=NULL;

		FILE *inputMat; FILE *inputVec;
		inputMat = fopen(argv[1],"r"); inputVec = fopen(argv[2],"r");

		nrows = 0; //Find number of Vector lines
		while (!feof(inputVec)) {
			getline(&vec,&vecLen,inputVec);
			nrows++;
		} nrows--;
		int totalE = 0; //Find number of Matrix lines
		while (!feof(inputMat)) {
			getline(&vec,&vecLen,inputMat);
			totalE++;
		} totalE--;
		fclose(inputMat); fclose(inputVec); 

		inputMat = fopen(argv[1],"r"); inputVec = fopen(argv[2],"r");
		vSize = ceil((double)nrows / num_procs); //partial vector size for each
		vSize2 = nrows - (num_procs-1)*vSize;    //vector size for last process
		if (vSize2 < 0) { //Error case
			printf("Invalid num processors, exiting..\n");
			exit(0);
		}

		//// Step 1: Read Files, Distribute Data
		//Reading Vector file
		double *allVecData = (double *)malloc(sizeof(double) * nrows);
		for (i = 0; i < nrows; i++) {
			getline(&vec,&vecLen,inputVec);
			allVecData[i] = strtod(vec,NULL);
		} //stored all vector entries 
		int vecDataSize[num_procs], vecDataDispls[num_procs];
		for (p = 0; p < num_procs-1; p++) { 
			vecDataSize[p] = vSize;
			vecDataDispls[p] = p*vSize;
		} //vector size and displacement for each processor
		vecDataSize[num_procs-1] = vSize2;
		vecDataDispls[num_procs-1] = (num_procs-1)*vSize;

		//Reading Matrix file into CSR format
		int *rowptr = (int *)malloc(sizeof(int) * (nrows+1));
		int *colInd = (int *)malloc(sizeof(int) * totalE);
		double *matVal = (double *)malloc(sizeof(double) * totalE);
		int prevRowI=-1; int j = 0;
		for (i = 0; i < totalE; i++) {
			getMatline(&vec,&rowI,&colI,&vecLen,inputMat);
			matVal[i] = strtod(vec,NULL);
			colInd[i] = strtol(colI,NULL,10);
			if (strtol(rowI,NULL,10) > prevRowI)
				rowptr[j++] = i;  
			prevRowI = strtol(rowI,NULL,10) ;
		} rowptr[j] = totalE;
		// stored matrix as arrays of row pointers, column index and values
		free(vec); free(rowI); free(colI);
		// matrix entries count and displacement for each processor
		int eCount[num_procs]; int eDispls[num_procs];
		for (p = 0; p < num_procs; p++) {
			eCount[p] = rowptr[p*vSize+vecDataSize[p]] - rowptr[p*vSize];
			eDispls[p] = rowptr[p*vSize];
		}
		fclose(inputMat); fclose(inputVec); 

		// store local vector entries
		vecData = (double *)malloc(sizeof(double) * vSize);

		// Scatter vector entries to each processor
		MPI_Bcast(&nrows,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Scatterv(allVecData,vecDataSize,vecDataDispls,MPI_DOUBLE,
				vecData,vSize,MPI_DOUBLE,0,MPI_COMM_WORLD);

		// Scatter matrix entries to each processor
		// by sending partial Row pointers, Column Index and Values
		MPI_Scatter(eCount,1,MPI_INT,&myNumE,1,MPI_INT,0,MPI_COMM_WORLD);

		myRowptr = (int *)malloc(sizeof(int) * (vSize+1));
		MPI_Scatterv(rowptr,vecDataSize,vecDataDispls,MPI_INT,
				myRowptr,vSize,MPI_INT,0,MPI_COMM_WORLD);
		myRowptr[vSize] = myNumE;

		myColInd = (int *)malloc(sizeof(int) * myNumE);
		MPI_Scatterv(colInd,eCount,eDispls,MPI_INT,
				myColInd,myNumE,MPI_INT,0,MPI_COMM_WORLD);
		
		myMatVal = (double *)malloc(sizeof(double) * myNumE);
		MPI_Scatterv(matVal,eCount,eDispls,MPI_DOUBLE,
				myMatVal,myNumE,MPI_DOUBLE,0,MPI_COMM_WORLD);
		free(rowptr); free(colInd); free(matVal);

		//// Step 2 - Analyze elements
		clearwctimer(timerTotal);
		startwctimer(timerTotal);

		// Sort column index of local matrix values to determine remote vector entries
		numVecTotal = sortGroupVec(myColInd,myNumE,myid,num_procs,nrows,
					remoteVecCount,remoteVecPtr,&remoteVecIndex);

		//// Step 3 & 4 - determine which processors to send what
		// Each process send to all others number of remote vector entries needed 
		MPI_Alltoall(remoteVecCount,1,MPI_INT,tosendVecCount,1,MPI_INT,MPI_COMM_WORLD);

		for (p = 0; p < num_procs; p++) 
			tosendTotal = tosendTotal + tosendVecCount[p];
		tosendVecIndex = (int *)malloc(sizeof(int) * tosendTotal);
		tosendVecPtr[0] = 0;
		for (p = 1; p < num_procs; p++)
			tosendVecPtr[p] = tosendVecPtr[p-1] + tosendVecCount[p-1];  

		// Each process send to all others index of remote vector entries needed
		MPI_Alltoallv(remoteVecIndex,remoteVecCount,remoteVecPtr,MPI_INT,tosendVecIndex,
				tosendVecCount,tosendVecPtr,MPI_INT,MPI_COMM_WORLD);
		
		// Get the vector entries other processors needed, ready to send
		sendVecData = (double *)malloc(sizeof(double) * tosendTotal);
		for (i = 0; i < tosendTotal; i++) 
			sendVecData[i] = vecData[tosendVecIndex[i]];

		// Realloc local vector array to store received remote vector entries
		vecData = (double *)realloc(vecData,sizeof(double) * (numVecTotal+vSize));
		// local entries at the front, remote entries are concatenated starting from: 
		double *recvVecData = vecData + vSize; 

		clearwctimer(timer56);
		startwctimer(timer56);

		//// Step 5 - All processors send the actual vector entries to each other 
		MPI_Alltoallv(sendVecData,tosendVecCount,tosendVecPtr,MPI_DOUBLE,recvVecData,
				remoteVecCount,remoteVecPtr,MPI_DOUBLE,MPI_COMM_WORLD);

		// Re-indexing the Column Index array
		int *searchInd;
		for (i = 0; i < myNumE; i++) { 
			searchInd = bsearch(&myColInd[i],remoteVecIndex,numVecTotal,
					sizeof(int),compare);
			if (searchInd != NULL) 
				myColInd[i] = vSize + searchInd - remoteVecIndex;
		}

		//// Step 6 - Compute the results
		result = (double *)malloc(sizeof(double) * vSize);
		computeMatVec(vSize,myRowptr,myColInd,myMatVal,vecData,result);

		stopwctimer(timer56);
		stopwctimer(timerTotal);
		printf("Total Time Taken: %.2lf sec; Time Taken (Steps 5&6): %.2lf sec.\n",
			       getwctimer(timerTotal), getwctimer(timer56));	

		//// Step 7 - Gather results from all processors
		totalResult = (double *)malloc(sizeof(double) * nrows);
		MPI_Gatherv(result,vSize,MPI_DOUBLE,totalResult,vecDataSize,vecDataDispls,
				MPI_DOUBLE,0,MPI_COMM_WORLD);

		//creating output filename
		int c = 0, temp = nrows;
		while (temp != 0) {
			temp = temp / 10;
			c++;
		}
		char fname[c+5];
		sprintf(fname,"o%d.vec",nrows);

		//Writing results to output file
		FILE *fpout;
		fpout = fopen(fname,"w");
		for (i = 0; i < nrows; i++)
			fprintf(fpout,"%.15lg\n",totalResult[i]);

		fclose(fpout);
		free(totalResult);

	// Other processes: Receive Data / Compute / Send Back
	} else {
		int *colInd, *eCount, *eDispls, *rowptr, *vecDataSize, *vecDataDispls;
		double *matVal, *allVecData;
		int size;

		MPI_Bcast(&nrows,1,MPI_INT,0,MPI_COMM_WORLD);
		vSize = ceil((double)nrows / num_procs);
		size = vSize;
		if (myid == num_procs-1)
			vSize = nrows - myid*vSize;
		// store local vector entries
		vecData = (double *)malloc(sizeof(double) * vSize);

		// Receive vector entries from root processor
		MPI_Scatterv(allVecData,vecDataSize,vecDataDispls,MPI_DOUBLE,
				vecData,vSize,MPI_DOUBLE,0,MPI_COMM_WORLD);

		MPI_Scatter(eCount,1,MPI_INT,&myNumE,1,MPI_INT,0,MPI_COMM_WORLD);

		// Receive matrix entries from root processor
		myRowptr = (int *)malloc(sizeof(int) * (vSize+1));
		MPI_Scatterv(rowptr,vecDataSize,vecDataDispls,MPI_INT,
				myRowptr,vSize,MPI_INT,0,MPI_COMM_WORLD);
		int rowptrOffset = myRowptr[0];
		for (i = 0; i < vSize; i++)
			myRowptr[i] = myRowptr[i] - rowptrOffset; //offset
		myRowptr[vSize] = myNumE;
		
		myColInd = (int *)malloc(sizeof(int) * myNumE);
		MPI_Scatterv(colInd,eCount,eDispls,MPI_INT,
				myColInd,myNumE,MPI_INT,0,MPI_COMM_WORLD);

		myMatVal = (double *)malloc(sizeof(double) * myNumE);
		MPI_Scatterv(matVal,eCount,eDispls,MPI_DOUBLE,
				myMatVal,myNumE,MPI_DOUBLE,0,MPI_COMM_WORLD);

		// Sort column index of local matrix values to determine remote vector entries
		numVecTotal = sortGroupVec(myColInd,myNumE,myid,num_procs,nrows,
					remoteVecCount,remoteVecPtr,&remoteVecIndex);

		// Each process send to all others number of remote vector entries needed 
		MPI_Alltoall(remoteVecCount,1,MPI_INT,tosendVecCount,1,MPI_INT,MPI_COMM_WORLD);

		for (p = 0; p < num_procs; p++) 
			tosendTotal = tosendTotal + tosendVecCount[p];
		tosendVecIndex = (int *)malloc(sizeof(int) * tosendTotal);
		tosendVecPtr[0] = 0;
		for (p = 1; p < num_procs; p++)
			tosendVecPtr[p] = tosendVecPtr[p-1] + tosendVecCount[p-1];  

		// Each process send to all others index of remote vector entries needed
		MPI_Alltoallv(remoteVecIndex,remoteVecCount,remoteVecPtr,MPI_INT,
				tosendVecIndex,tosendVecCount,tosendVecPtr,MPI_INT,MPI_COMM_WORLD);
		// Get the vector entries other processors needed, ready to send
		sendVecData = (double *)malloc(sizeof(double) * tosendTotal);
		for (i = 0; i < tosendTotal; i++) 
			sendVecData[i] = vecData[tosendVecIndex[i]-myid*size];
		
		// Realloc local vector array to store received remote vector entries
		vecData = (double *)realloc(vecData,sizeof(double) * (numVecTotal+vSize));
		// local entries at the front, remote entries are concatenated starting from: 
		double *recvVecData = vecData + vSize;

		// All processors send the actual vector entries to each other 
		MPI_Alltoallv(sendVecData,tosendVecCount,tosendVecPtr,MPI_DOUBLE,recvVecData,
				remoteVecCount,remoteVecPtr,MPI_DOUBLE,MPI_COMM_WORLD);


		// Re-indexing the Column Index array
		int *searchInd;
		for (i = 0; i < myNumE; i++) { 
			searchInd = bsearch(&myColInd[i],remoteVecIndex,numVecTotal,
					sizeof(int),compare);
			if (searchInd == NULL) 
				myColInd[i] -= (myid * size); 
			else
				myColInd[i] = vSize + searchInd-remoteVecIndex;
		}

		// Compute the local results
		result = (double *)malloc(sizeof(double) * vSize);
		computeMatVec(vSize,myRowptr,myColInd,myMatVal,vecData,result);

		// Send the results to the root processor
		MPI_Gatherv(result,vSize,MPI_DOUBLE,totalResult,vecDataSize,vecDataDispls,
				MPI_DOUBLE,0,MPI_COMM_WORLD);
	}
	free(myRowptr); free(myColInd); free(myMatVal); free(tosendVecIndex);
	free(sendVecData); free(result); free(vecData); free(remoteVecIndex);

	MPI_Finalize();
	return 0;
}

//CRS Matrix-Vector multiplication
void computeMatVec(int nrows, int *rowptr, int *colind, double *matval, double *vec, double *res) {
	int i, j, count;
	count = 0;
	for (i = 0; i < nrows; i++) {
		res[i] = 0.0;
		for (j = rowptr[i]; j < rowptr[i+1]; j++) {
			res[i] += matval[count] * vec[colind[j]];
			count++;
		}
	}
}

int compare(const void * a, const void * b) {
	return ( *(int*)a - *(int*)b );
}

int sortGroupVec(int *colInd, int numE, int myid, int numprocs, int nrows,
	       	int *remoteVecCount, int *remoteVecPtr, int **remoteVecIndex) {
	int i;
	int size = ceil((double)nrows / numprocs);
	int *sortedInd = (int *)malloc(sizeof(int) * numE);
	int localVecIndStart = myid * size;
	int localVecIndEnd = (myid + 1) * size;
	memcpy(sortedInd,colInd,sizeof(int)*numE);
	// Sort the temp Column Index in increasing order
	qsort(sortedInd,numE,sizeof(int),compare);

	for (i = 0; i < numprocs; i++)
		remoteVecCount[i] = 0;
	*remoteVecIndex = (int *)malloc(sizeof(int) * numE);

	int prevInd = -1 ;
	int j = 0;
	// Only take distinct index values that are also non-local
	// Keep track of these remote vector entries count and index for each processor
	for (i = 0; i < numE; i++) {
		if ((sortedInd[i] > prevInd) &&
		((sortedInd[i] < localVecIndStart) || (sortedInd[i] >= localVecIndEnd))) {
			remoteVecCount[sortedInd[i]/size]++;
			(*remoteVecIndex)[j] = sortedInd[i];
			j++;
		}
		prevInd = sortedInd[i];
	}
	remoteVecPtr[0] = 0;
	for (i = 1; i < numprocs; i++)
		remoteVecPtr[i] = remoteVecPtr[i-1] + remoteVecCount[i-1];  

	free(sortedInd);
	//return total number of remote vector entries needed
	return j;
}

double wClockSeconds(void)
{
#ifdef __GNUC__
  struct timeval ctime;

  gettimeofday(&ctime, NULL);

  return (double)ctime.tv_sec + (double).000001*ctime.tv_usec;
#else
  return (double)time(NULL);
#endif
}

int getMatline(char **lineptr, char **rowInd, char **colInd, size_t *n, FILE *stream)
{
  size_t i;
  int ch;

  if (feof(stream))
    return -1;  

  if (*rowInd == NULL || *n == 0) {
    *n = 1024;
    *rowInd = malloc((*n)*sizeof(char));
  }
  i = 0;
  while ((ch = getc(stream)) != ' ') {
    (*rowInd)[i++] = (char)ch;
    /* reallocate memory if reached at the end of the buffer. The +1 is for '\0' */
    if (i+1 == *n) { 
      *n = 2*(*n);
      *rowInd = realloc(*rowInd, (*n)*sizeof(char));
    }
  }
  (*rowInd)[i] = '\0';

  if (*colInd == NULL || *n == 0) {
    *n = 1024;
    *colInd = malloc((*n)*sizeof(char));
  }
  i = 0;
  while ((ch = getc(stream)) != ' ') {
    (*colInd)[i++] = (char)ch;
    /* reallocate memory if reached at the end of the buffer. The +1 is for '\0' */
    if (i+1 == *n) { 
      *n = 2*(*n);
      *colInd = realloc(*colInd, (*n)*sizeof(char));
    }
  }
  (*colInd)[i] = '\0';

  /* Initial memory allocation if *lineptr is NULL */
  if (*lineptr == NULL || *n == 0) {
    *n = 1024;
    *lineptr = malloc((*n)*sizeof(char));
  }


  /* get into the main loop */
  i = 0;
  while ((ch = getc(stream)) != EOF) {

    if (ch == '\n') 
      break;

    (*lineptr)[i++] = (char)ch;

    /* reallocate memory if reached at the end of the buffer. The +1 is for '\0' */
    if (i+1 == *n) { 
      *n = 2*(*n);
      *lineptr = realloc(*lineptr, (*n)*sizeof(char));
    }
      
  }
  (*lineptr)[i] = '\0';

  return (i == 0 ? -1 : i);
}
