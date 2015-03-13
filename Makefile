CC = gcc
OPTFLAGS = -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99 -Wall\
           -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
           -Wno-parentheses -Wsequence-point

CFLAGS = $(COPTIONS)  $(OPTFLAGS)

LIBS = -lm

MPISRC = matvec_mpi.c
MPISRC2 = matvec_mpiSortSearch.c

################################################################################
default:
	( mpicc -o matvec_mpi $(MPISRC) ) 
	( mpicc -o matvec_mpiSortSearch $(MPISRC2) )
	
clean:
	rm -f *.o \
	rm -f matvec_mpi \
	rm -f matvec_mpiSortSearch ;
