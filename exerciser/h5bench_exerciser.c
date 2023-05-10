/*
 * Parallel HDF5 Exerciser Benchmark
 *
 * Programmers: Richard Zamora <rzamora@anl.gov>
 *              last modified: December 13th, 2018
 *
 *              Paul Coffman <pcoffman@anl.gov>
 *              last modified: April 2018
 *
 * Purpose:     The HDF5 Exerciser Benchmark creates an HDF5 use case with
 *              some ideas/code borrowed from other benchmarks (namely `IOR`,
 *              `VPICIO` and `FLASHIO`). The benchmark "exercises" both
 *              metadata and bandwidth-limited operations. See README.md for
 *              more information.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <hdf5.h>
#include <mpi.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#ifdef __linux__
#include <sys/sysinfo.h>
#include <sys/resource.h>
#endif

/* Define Constants */
#define NUM_ATTRIBUTES 64
#define ATTRIBUTE_SIZE 1024
#define NAME_LENGTH    1024
#define NUM_ITERATIONS 10
#define MAX_DIM        4
#define HYPER_VERBOSE  1
#define MIN(a, b)      (((a) < (b)) ? (a) : (b))
#define MAX(a, b)      (((a) > (b)) ? (a) : (b))
#define NUM_MOM        5
#define INIT_MOM                                                                                             \
    {                                                                                                        \
        0.0, 0.0, 0.0, 0.0, 0.0                                                                              \
    }
#define AVG_ind 0
#define STD_ind 1
#define MIN_ind 2
#define MED_ind 3
#define MAX_ind 4

/* Helper function to calc statistical quantities */
void getmoments(double *ain, int n, double *momarr);

/* data structure from flashio */
typedef struct sim_params_t {
    int    total_blocks;
    int    nsteps;
    int    nxb;
    int    nyb;
    int    nzb;
    double time;
    double timestep;
} sim_params_t;

/* Main Method of the HDF5 Exerciser */
int
main(int argc, char *argv[])
{

    /* Local Vars */
    int nprocs, rank, i, j;

    /* Following defaults are 0 */
    int useMetaDataCollectives = 0;
    int addDerivedTypeDataset  = 0;
    int addAttributes          = 0;
    int useIndependentIO       = 0;
    int keepFile               = 0;
    int numDims                = 0;
    int useChunked             = 0;
    int rankshift              = 0;
    int maxcheck_set           = 0;

    /* Following defaults are 1 */
    int numBufLoops = 1;

    /* Following defaults are "other" */
    long maxcheck = 17179869184;

    /* Multi-dim Buffers */
    int bufMult[MAX_DIM];
    int minNEls[MAX_DIM];
    int curNEls[MAX_DIM];
    int dimRanks[MAX_DIM];
    int rankLocation[MAX_DIM];
    int rankLocationShiftedWr[MAX_DIM];
    int rankLocationShiftedRd[MAX_DIM];

    /* Mem/File space settings */
    hsize_t fileBlock_def[MAX_DIM];
    hsize_t fileBlock_dbl[MAX_DIM];
    hsize_t fileBlock_spe[MAX_DIM];
    hsize_t fileCount_dbl[MAX_DIM];
    hsize_t fileCount_spe[MAX_DIM];
    hsize_t fileStride_def[MAX_DIM];
    hsize_t fileStride_dbl[MAX_DIM];
    hsize_t fileStride_spe[MAX_DIM];
    for (i = 0; i < MAX_DIM; i++) {
        bufMult[i]               = 2;
        minNEls[i]               = 8;
        curNEls[i]               = 8;
        dimRanks[i]              = 1;
        rankLocation[i]          = 1;
        rankLocationShiftedWr[i] = 1;
        rankLocationShiftedRd[i] = 1;
        fileBlock_def[i]         = 0;
        fileBlock_dbl[i]         = 0;
        fileBlock_spe[i]         = 0;
        fileStride_def[i]        = 1;
        fileStride_dbl[i]        = 1;
        fileStride_spe[i]        = 1;
        fileCount_dbl[i]         = 1;
        fileCount_spe[i]         = 1;
    }
    hsize_t memBlock_dbl  = 0;
    hsize_t memStride_dbl = 1;
    hsize_t memCount_dbl  = 1;

    /* Parse Input Args */
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--metacoll") == 0)
            useMetaDataCollectives = 1;
        else if (strcmp(argv[i], "--derivedtype") == 0)
            addDerivedTypeDataset = 1;
        else if (strcmp(argv[i], "--addattr") == 0)
            addAttributes = 1;
        else if (strcmp(argv[i], "--indepio") == 0)
            useIndependentIO = 1;
        else if (strcmp(argv[i], "--keepfile") == 0)
            keepFile = 1;
        else if (strcmp(argv[i], "--usechunked") == 0)
            useChunked = 1;
        else if (strcmp(argv[i], "--maxcheck") == 0) {
            i++;
            maxcheck_set = 1;
            maxcheck     = (hsize_t)atol(argv[i]);
        }
        else if (strcmp(argv[i], "--nsizes") == 0) {
            i++;
            numBufLoops = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--bufmult") == 0) {
            if (numDims < 1) {
                printf("ERROR: '--numdims' flag must come before '--bufmult'\n");
                exit(-1);
            }
            for (j = 0; j < numDims; j++) {
                i++;
                bufMult[j] = atoi(argv[i]);
            }
        }
        else if (strcmp(argv[i], "--memblock") == 0) {
            i++;
            memBlock_dbl = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--memstride") == 0) {
            i++;
            memStride_dbl = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--fileblocks") == 0) {
            if (numDims < 1) {
                printf("ERROR: '--numdims' flag must come before '--fileblocks'\n");
                exit(-1);
            }
            for (j = 0; j < numDims; j++) {
                i++;
                fileBlock_def[j] = atoi(argv[i]);
            }
        }
        else if (strcmp(argv[i], "--filestrides") == 0) {
            if (numDims < 1) {
                printf("ERROR: '--numdims' flag must come before '--filestrides'\n");
                exit(-1);
            }
            for (j = 0; j < numDims; j++) {
                i++;
                fileStride_def[j] = atoi(argv[i]);
            }
        }
        else if (strcmp(argv[i], "--numdims") == 0) {
            i++;
            numDims = atoi(argv[i]);
            if (numDims > 4) {
                printf("WARNING: Maximum of 4 dimensions is suppported -- ");
                printf("Using numDims = 4 \n");
                numDims = 4;
            }
        }
        else if (strcmp(argv[i], "--minels") == 0) {
            if (numDims < 1) {
                printf("ERROR: '--numdims' flag must come before '--minels'\n");
                exit(-1);
            }
            for (j = 0; j < numDims; j++) {
                i++;
                minNEls[j] = atoi(argv[i]);
            }
        }
        else if (strcmp(argv[i], "--dimranks") == 0) {
            if (numDims < 1) {
                printf("ERROR: '--numdims' flag must come before '--dimranks'\n");
                exit(-1);
            }
            for (j = 0; j < numDims; j++) {
                i++;
                dimRanks[j] = atoi(argv[i]);
            }
        }
        else if (strcmp(argv[i], "--rshift") == 0) {
            i++;
            rankshift = atoi(argv[i]);
        }
        else {
            printf("ERROR - unrecognized parameter: %s.  Exitting.\n", argv[i]);
            exit(-1);
        }
    }

    double startTime;
    herr_t rc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm comm = MPI_COMM_WORLD;

    /* Make sure the array decomposition makes sense. For now, we can just divide
     * the 0th domain by nprocs if the given decomposition is no good.
     * (could also call a function to calculate a good decomp automatically)
     */
    if (rank == 0) {
        int ndomains = 1;
        for (i = 0; i < numDims; ++i)
            ndomains *= dimRanks[i];
        if (ndomains != nprocs) {
            printf("WARNING!! - ndomains: %d must be: %d. Only dividing dim-0\n", ndomains, nprocs);
            for (j = 0; j <= numDims; ++j)
                dimRanks[j] = 1;
            dimRanks[0] = nprocs;
        }
    }
    MPI_Bcast(dimRanks, numDims, MPI_INT, 0, MPI_COMM_WORLD);

    int cary = rank;
    for (i = numDims - 1; i >= 0; --i) {
        rankLocation[i] = cary % dimRanks[i];
        cary            = cary / dimRanks[i];
    }
    /* If we are shifting the ranks... */
    if (rankshift > 0) {
        /* Shifting the location of the WRITE location in file */
        cary = rank + rankshift;
        if (cary >= nprocs)
            cary -= nprocs;
        for (i = numDims - 1; i >= 0; --i) {
            rankLocationShiftedWr[i] = cary % dimRanks[i];
            cary                     = cary / dimRanks[i];
        }
        /* Shifting the location of the READ location in file */
        cary = rank + rankshift * 2;
        if (cary >= nprocs)
            cary -= nprocs;
        for (i = numDims - 1; i >= 0; --i) {
            rankLocationShiftedRd[i] = cary % dimRanks[i];
            cary                     = cary / dimRanks[i];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("WARNING: Bufsize is the size of the double-type dataset (The derived-type dataset is 8x "
               "larger!).\n");
        printf("useMetaDataCollectives: %d addDerivedTypeDataset: %d addAttributes: %d useIndependentIO: %d "
               "numDims: %d useChunked: %d rankShift: %d\n",
               useMetaDataCollectives, addDerivedTypeDataset, addAttributes, useIndependentIO, numDims,
               useChunked, rankshift);
        printf("Metric      Bufsize   H5DWrite    RawWrBDWTH    H5Dread    RawRdBDWTH    Dataset      Group  "
               "Attribute    H5Fopen   H5Fclose   H5Fflush OtherClose\n");
    }

    int bufloopIter = 0;
    int loopIter    = 0;

    /* randomize the testfilename based on time so multiple exercisers can run concurrently */
    char           testFileName[NAME_LENGTH];
    struct timeval time;
    gettimeofday(&time, NULL);
    srand(((unsigned int)time.tv_sec * 1000) + ((unsigned int)time.tv_usec / 1000));
    sprintf(testFileName, "hdf5TestFile-%d", rand());
    MPI_Bcast(testFileName, NAME_LENGTH, MPI_CHAR, 0, comm);
    char dataSetName1[NAME_LENGTH] = "hdf5DataSet1";

    /* Create the file initially outside of the timing loop */
    hid_t fd, accessPropList, createPropList;
    createPropList = H5Pcreate(H5P_FILE_CREATE);

    /* Set the parallel driver.
     * Don't want to do this repetatively, as overhead is quite large for some
     * mpi implementations and most apps will just do this once anyway. */
    accessPropList = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(accessPropList, comm, MPI_INFO_NULL);
    fd = H5Fcreate(testFileName, H5F_ACC_TRUNC, createPropList, accessPropList);
    rc = H5Fclose(fd);

#ifdef METACOLOK
    /* Set collective metadata if available */
    if (useMetaDataCollectives) {
        hbool_t metaDataCollWrite = true;
        rc                        = H5Pset_coll_metadata_write(accessPropList, metaDataCollWrite);
        if (rc < 0) {
            printf("H5Pset_coll_metadata_write failed with rc %d\n", rc);
            exit(1);
        }
    };
#else
    if (rank == 0) {
        printf("Warning: Metadata Collectives are DISABLED.\n");
    }
#endif

    /* Initialize curNEls to minNEls */
    for (j = 0; j < numDims; j++)
        curNEls[j] = minNEls[j];

    /* Loop through different buffer-size iterations */
    for (bufloopIter = 0; bufloopIter < numBufLoops; bufloopIter++) {

        /* Determine array sizes (Store numDims-D array as 1D array in memory) */
        /* Total size of the local multidimentional buffer: */
        long BufSizeTotalDouble  = sizeof(H5T_NATIVE_DOUBLE);
        long BufSizeTotalDerived = sizeof(sim_params_t);
        /* Total number of double elements needed to fill the local buffer: */
        int NumDoubleElements = 1;
        /* Total number of derived-type elements needed to fill the local buffer: */
        int NumSimParamElements = 1;
        for (j = 0; j < numDims; j++) {

            BufSizeTotalDouble *= curNEls[j];
            BufSizeTotalDerived *= curNEls[j];
            assert(NumDoubleElements * curNEls[j] < INT_MAX);
            NumDoubleElements *= curNEls[j];
            assert(NumSimParamElements * curNEls[j] < INT_MAX);
            NumSimParamElements *= curNEls[j];

            /* Lets reset the block setting to default */
            fileBlock_dbl[j]  = fileBlock_def[j];
            fileStride_dbl[j] = fileStride_def[j];

            /* Check that the FILE space settings make sense */
            if ((fileBlock_dbl[j] < 1) || (curNEls[j] % fileBlock_dbl[j] != 0)) {
                /* Setting doesn't make sense, fall back to a simple setting
                 * and print an error/warning... */
                fileBlock_dbl[j]  = curNEls[j];
                fileStride_dbl[j] = curNEls[j];
                fileCount_dbl[j]  = 1;
            }
            else {
                fileCount_dbl[j] = curNEls[j] / fileBlock_dbl[j];
                if (fileStride_dbl[j] < fileBlock_dbl[j]) {
                    fileStride_dbl[j] = fileBlock_dbl[j];
                }
            }
            if (rank == 0 && HYPER_VERBOSE)
                printf("FileSpace: Dim= %d Block= %llu Stride= %llu Count= %llu\n", j, fileBlock_dbl[j],
                       fileStride_dbl[j], fileCount_dbl[j]);

            /* Check that the FILE space settings make sense */
            if ((fileBlock_spe[j] < 1) || (curNEls[j] % fileBlock_spe[j] != 0)) {
                /* Setting doesn't make sense, fall back to a simple setting
                 * and print an error/warning... */
                fileBlock_spe[j]  = curNEls[j];
                fileStride_spe[j] = curNEls[j];
                fileCount_spe[j]  = 1;
            }
            else {
                fileCount_spe[j] = curNEls[j] / fileBlock_spe[j];
                if (fileStride_spe[j] < fileBlock_spe[j]) {
                    fileStride_spe[j] = fileBlock_spe[j];
                }
            }
        }

        /* Know how many double elements we need -- now add memory block/stride */
        if ((memBlock_dbl < 1) || (NumDoubleElements % memBlock_dbl != 0)) {
            memBlock_dbl  = 1;
            memStride_dbl = 1;
            memCount_dbl  = NumDoubleElements;
        }
        else {
            memCount_dbl = NumDoubleElements / memBlock_dbl;
            if (memStride_dbl < memBlock_dbl) {
                memStride_dbl = memBlock_dbl;
            }
            NumDoubleElements = memStride_dbl * memCount_dbl;
        }
        if (rank == 0 && HYPER_VERBOSE)
            printf("MemSpace:  Dim= %d Block= %llu Stride= %llu Count= %llu\n", 0, memBlock_dbl,
                   memStride_dbl, memCount_dbl);

        /* data buffers for writing and reading */
        double *dataBuffer  = (double *)malloc(sizeof(double) * NumDoubleElements);
        double *checkBuffer = (double *)malloc(sizeof(double) * NumDoubleElements);
        double *dataBufferShift;
        if (rankshift > 0) {
            dataBufferShift = (double *)malloc(sizeof(double) * NumDoubleElements);
        }
        sim_params_t *sim_params = (sim_params_t *)malloc(sizeof(sim_params_t) * NumSimParamElements);
        for (i = 0; i < NumSimParamElements; i++) {
            sim_params[i].total_blocks = i + rank * 100;
            sim_params[i].time         = 12345.6789;
            sim_params[i].timestep     = 12345.6789;
            sim_params[i].nsteps       = i + rank * 100;
            sim_params[i].nxb          = i + rank * 100;
            sim_params[i].nyb          = i + rank * 100;
            sim_params[i].nzb          = i + rank * 100;
        }

        /* Init buffers with some data */
        for (i = 0; i < memCount_dbl; i++) {
            for (j = 0; j < memStride_dbl; j++) {
                int ii = i * memStride_dbl + j;
                if (j < memBlock_dbl) {
                    // dataBuffer[ii] = (double)(rank+1) * ((double)(ii+1) * (double)0.355 * (double)(ii+1)) /
                    // (double) 23355.53235;
                    dataBuffer[ii] = (double)(rank + 1);
                }
                else {
                    dataBuffer[ii] = 0.0;
                }
                checkBuffer[ii] = 0.0;
                /* Populate a buffer with expected data for "shifted" read... */
                if (rankshift > 0) {
                    if (j < memBlock_dbl) {
                        cary = rank + rankshift; // * 2;
                        if (cary >= nprocs)
                            cary -= nprocs;
                        dataBufferShift[ii] = (double)(cary + 1);
                    }
                    else {
                        dataBufferShift[ii] = 0.0;
                    }
                }
            }
        }

        double minRawWriteBDWTH[NUM_ITERATIONS];
        double minRawReadBDWTH[NUM_ITERATIONS];
        double maxWriteDataTime[NUM_ITERATIONS];
        double maxReadDataTime[NUM_ITERATIONS];
        double maxDataSetTime[NUM_ITERATIONS];
        double maxGroupTime[NUM_ITERATIONS];
        double maxAttrTime[NUM_ITERATIONS];
        double maxFcloseTime[NUM_ITERATIONS];
        double maxFopenTime[NUM_ITERATIONS];
        double maxFflushTime[NUM_ITERATIONS];
        double maxOtherCloseTime[NUM_ITERATIONS];
        double MOMWriteDataTime[NUM_MOM]  = INIT_MOM;
        double MOMReadDataTime[NUM_MOM]   = INIT_MOM;
        double MOMDataSetTime[NUM_MOM]    = INIT_MOM;
        double MOMGroupTime[NUM_MOM]      = INIT_MOM;
        double MOMAttrTime[NUM_MOM]       = INIT_MOM;
        double MOMFcloseTime[NUM_MOM]     = INIT_MOM;
        double MOMFopenTime[NUM_MOM]      = INIT_MOM;
        double MOMFflushTime[NUM_MOM]     = INIT_MOM;
        double MOMOtherCloseTime[NUM_MOM] = INIT_MOM;
        double MOMRawWriteBDWTH[NUM_MOM]  = INIT_MOM;
        double MOMRawReadBDWTH[NUM_MOM]   = INIT_MOM;

        for (loopIter = 0; loopIter < NUM_ITERATIONS; loopIter++) {

            double timenow      = 0.0;
            double rawReadBDWTH = 0.0, rawWriteBDWTH = 0.0;
            double writeDataTime = 0.0, readDataTime = 0.0;
            double dataSetTime = 0.0, groupTime = 0.0, attrTime = 0.0, fopenTime = 0.0;
            double fcloseTime = 0.0, fflushTime = 0.0, otherCloseTime = 0.0;

            /* Init memory -
             * Keep this dynamic because numDims is customizable from command line */
            hsize_t *dataSetDims = (hsize_t *)malloc(sizeof(hsize_t) * numDims);
            hsize_t *fileStart   = (hsize_t *)malloc(sizeof(hsize_t) * numDims);
            hsize_t *fileStride  = (hsize_t *)malloc(sizeof(hsize_t) * numDims);
            hsize_t *fileBlock   = (hsize_t *)malloc(sizeof(hsize_t) * numDims);
            hsize_t *fileCount   = (hsize_t *)malloc(sizeof(hsize_t) * numDims);
            /* Lets keep things simple by treating memory space as 1d */
            hsize_t memStart[]         = {0};
            hsize_t memStride[]        = {memStride_dbl};
            hsize_t memBlock[]         = {memBlock_dbl};
            hsize_t memCount[]         = {memCount_dbl};
            hsize_t memDataSpaceDims[] = {NumDoubleElements};
            for (i = 0; i < numDims; i++) {
                fileStride[i]  = fileStride_dbl[i];
                fileBlock[i]   = fileBlock_dbl[i];
                fileCount[i]   = fileCount_dbl[i];
                fileStart[i]   = fileStride[i] * fileCount[i] * rankLocation[i];
                dataSetDims[i] = fileStride[i] * fileCount[i] * dimRanks[i];
            }

            /* create the file -
             * Since we have already created it will just open and overwrite */
            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            fd        = H5Fcreate(testFileName, H5F_ACC_TRUNC, createPropList, accessPropList);
            fopenTime += (MPI_Wtime() - startTime);

            hid_t xferPropList;  /* xfer property list */
            hid_t dataSet;       /* data set id */
            hid_t fileDataSpace; /* file data space id */
            hid_t memDataSpace;  /* memory data space id */

            MPI_Barrier(comm);
            startTime = MPI_Wtime();

            hid_t topgroupid = H5Gcreate(fd, "/Data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            groupTime += (MPI_Wtime() - startTime);

            hid_t   attr1id, attr1DS;
            hsize_t attr1Dims[]              = {ATTRIBUTE_SIZE};
            char    attr1Buf[ATTRIBUTE_SIZE] = {"This is my attribute string."};
            char    attributeName[NAME_LENGTH];

            startTime = MPI_Wtime();
            if (addAttributes) {
                MPI_Barrier(comm);
                startTime = MPI_Wtime();
                for (i = 0; i < NUM_ATTRIBUTES; i++) {
                    sprintf(attr1Buf, "This is my attribute string number %d", i);
                    attr1DS = H5Screate_simple(1, attr1Dims, NULL);
                    sprintf(attributeName, "Attribute %d Name", i);
                    attr1id = H5Acreate(topgroupid, attributeName, H5T_NATIVE_CHAR, attr1DS, H5P_DEFAULT,
                                        H5P_DEFAULT);
                    rc      = H5Awrite(attr1id, H5T_NATIVE_CHAR, attr1Buf);
                    H5Aclose(attr1id);
                    H5Sclose(attr1DS);
                }
            }
            attrTime += (MPI_Wtime() - startTime);

            /* create prop list for parallel io transfer */
            xferPropList = H5Pcreate(H5P_DATASET_XFER);

            if (useIndependentIO)
                H5Pset_dxpl_mpio(xferPropList, H5FD_MPIO_INDEPENDENT);
            else
                H5Pset_dxpl_mpio(xferPropList, H5FD_MPIO_COLLECTIVE);

            hid_t dataSetPropList = H5Pcreate(H5P_DATASET_CREATE);

            /* If we are chunking the data.. */
            if (useChunked) {
                H5Pset_chunk(dataSetPropList, numDims, fileBlock);
                H5Pset_dxpl_mpio_chunk_opt(xferPropList, H5FD_MPIO_CHUNK_MULTI_IO);
                // H5Pset_dxpl_mpio_chunk_opt(xferPropList, H5FD_MPIO_CHUNK_ONE_IO); //Not implemented in CCIO
                // yet!
            }

            H5Pset_fill_time(dataSetPropList, H5D_FILL_TIME_NEVER);

            /* Create data spaces */
            memDataSpace  = H5Screate_simple(1, memDataSpaceDims, NULL);
            fileDataSpace = H5Screate_simple(numDims, dataSetDims, NULL);

            /* (Write) Change rankLocation list if we are shifting the ranks... */
            if (rankshift > 0) {
                /* Shifting the write location of the ranks in the file... */
                for (i = 0; i < numDims; i++) {
                    fileStart[i] = fileStride[i] * fileCount[i] * rankLocationShiftedWr[i];
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }

            /* define hyperslab for memory data space */
            H5Sselect_hyperslab(memDataSpace, H5S_SELECT_SET, memStart, memStride, memCount, memBlock);
            /* define hyperslab for file data space */
            H5Sselect_hyperslab(fileDataSpace, H5S_SELECT_SET, fileStart, fileStride, fileCount, fileBlock);

            /* create the file dataset */
            sprintf(dataSetName1, "hdf5DataSet1-iter%d-size%d", loopIter, NumDoubleElements);

            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            dataSet   = H5Dcreate(topgroupid, dataSetName1, H5T_NATIVE_DOUBLE, fileDataSpace, H5P_DEFAULT,
                                dataSetPropList, H5P_DEFAULT);
            dataSetTime += (MPI_Wtime() - startTime);

            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            H5Dwrite(dataSet, H5T_NATIVE_DOUBLE, memDataSpace, fileDataSpace, xferPropList, dataBuffer);
            timenow = MPI_Wtime();
            writeDataTime += (timenow - startTime);
            if ((timenow - startTime) > 0.0) {
                rawWriteBDWTH = ((double)BufSizeTotalDouble / (1048576.0)) / (timenow - startTime); // Mbps
                rawWriteBDWTH *= (double)nprocs;                                                    // Mbps
            }
            else {
                printf("ERROR on write: Measured time is %20.16f \n", timenow - startTime);
            }

            /* Now CLOSE everything and READ it */
            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            H5Dclose(dataSet);
            H5Sclose(fileDataSpace);
            H5Sclose(memDataSpace);
            H5Gclose(topgroupid);

            otherCloseTime += (MPI_Wtime() - startTime);

            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            rc        = H5Fflush(fd, H5F_SCOPE_LOCAL);
            fflushTime += (MPI_Wtime() - startTime);
            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            rc        = H5Fclose(fd);
            fcloseTime += (MPI_Wtime() - startTime);

            if (rc < 0) {
                printf("H5Fclose failed with rc %d\n", rc);
                exit(1);
            }

            MPI_Barrier(comm);

            /* re-create mem data space */
            memDataSpace = H5Screate_simple(1, memDataSpaceDims, NULL);

            hid_t dtype_id, plist_id;

            /* Need to close and recreate the access property list */
            H5Pclose(accessPropList);
            accessPropList = H5Pcreate(H5P_FILE_ACCESS);
            H5Pset_fapl_mpio(accessPropList, comm, MPI_INFO_NULL);
#ifdef METACOLOK
            if (useMetaDataCollectives) {
                hbool_t metaDataCollRead = true;
                rc                       = H5Pset_all_coll_metadata_ops(accessPropList, metaDataCollRead);
                if (rc < 0) {
                    printf("H5Pset_all_coll_metadata_ops failed with rc %d\n", rc);
                    exit(1);
                }
            };
#endif

            /* Open an existing file. */
            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            fd        = H5Fopen(testFileName, H5F_ACC_RDWR, accessPropList);
            fopenTime += (MPI_Wtime() - startTime);

            MPI_Barrier(comm);
            startTime  = MPI_Wtime();
            topgroupid = H5Gopen(fd, "/Data", H5P_DEFAULT);
            groupTime += (MPI_Wtime() - startTime);

            if (fd < 0) {
                printf("H5Fopen error - fd is %lld\n", (hid_t)fd);
                exit(1);
            }

            /* Open an existing dataset. */
            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            dataSet   = H5Dopen(topgroupid, dataSetName1, H5P_DEFAULT);
            dataSetTime += (MPI_Wtime() - startTime);

            if (dataSet < 0) {
                printf("H5Dopen error - dataSet is %lld\n", (hid_t)dataSet);
                exit(1);
            }

            fileDataSpace = H5Dget_space(dataSet);

            if (fileDataSpace < 0) {
                printf("H5Dget_space error - fileDataSpace is %lld\n", (hid_t)fileDataSpace);
                exit(1);
            }
            dtype_id = H5Dget_type(dataSet);
            plist_id = H5Dget_create_plist(dataSet);

            /* Change rankLocation list if we are shifting the ranks... */
            if (rankshift > 0) {
                /* We are basically shifting the location of the ranks in the file... */
                for (i = 0; i < numDims; i++) {
                    fileStart[i] = fileStride[i] * fileCount[i] * rankLocationShiftedRd[i];
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }

            /* define hyperslab for memory data space */
            H5Sselect_hyperslab(memDataSpace, H5S_SELECT_SET, memStart, memStride, memCount, memBlock);

            /* define hyperslab for file data space */
            H5Sselect_hyperslab(fileDataSpace, H5S_SELECT_SET, fileStart, fileStride, fileCount, fileBlock);

            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            H5Dread(dataSet, H5T_NATIVE_DOUBLE, memDataSpace, fileDataSpace, xferPropList, checkBuffer);
            timenow = MPI_Wtime();
            readDataTime += (timenow - startTime);
            if ((timenow - startTime) > 0.0) {
                rawReadBDWTH = ((double)BufSizeTotalDouble / (1048576.0)) / (timenow - startTime); // Mbps
                rawReadBDWTH *= (double)nprocs;                                                    // Mbps
            }
            else {
                printf("ERROR on read: Measured time is %20.16f \n", timenow - startTime);
            }

            /* Verify the data only for <= "maxcheck" buffers */
            if (!maxcheck_set || (BufSizeTotalDouble <= maxcheck)) {
                for (i = 0; i < (NumDoubleElements); i++) {
                    if (rankshift == 0 && (dataBuffer[i] != checkBuffer[i])) {
                        printf("Rank %d - ERROR on read: index %d doesn't match - expected %20.16f got "
                               "%20.16f \n",
                               rank, i, dataBuffer[i], checkBuffer[i]);
                    }
                    else if (rankshift > 0 && (dataBufferShift[i] != checkBuffer[i])) {
                        printf("Rank %d - ERROR on read: index %d doesn't match - expected %20.16f got "
                               "%20.16f \n",
                               rank, i, dataBufferShift[i], checkBuffer[i]);
                    }
                }
            }

            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            rc        = H5Dclose(dataSet);
            rc        = H5Sclose(fileDataSpace);
            otherCloseTime += (MPI_Wtime() - startTime);

            if (addDerivedTypeDataset) {

                memStart[0]         = 0;
                memStride[0]        = NumSimParamElements;
                memBlock[0]         = NumSimParamElements;
                memCount[0]         = 1;
                memDataSpaceDims[0] = NumSimParamElements;
                for (i = 0; i < numDims; i++) {
                    fileStride[i]  = fileStride_spe[i];
                    fileBlock[i]   = fileBlock_spe[i];
                    fileCount[i]   = fileCount_spe[i];
                    fileStart[i]   = fileStride[i] * fileCount[i] * rankLocation[i];
                    dataSetDims[i] = fileStride[i] * fileCount[i] * dimRanks[i];
                }

                /* If we are chunking the data.. */
                if (useChunked) {
                    H5Pset_chunk(dataSetPropList, numDims, fileBlock);
                    H5Pset_dxpl_mpio_chunk_opt(xferPropList, H5FD_MPIO_CHUNK_MULTI_IO);
                    // H5Pset_dxpl_mpio_chunk_opt(xferPropList, H5FD_MPIO_CHUNK_ONE_IO); //Not implemented in
                    // CCIO yet!
                }

                /* Need to create a new memDataSpaceDims
                 * (memDataSpaceDims has changed) */
                H5Sclose(memDataSpace);
                memDataSpace = H5Screate_simple(1, memDataSpaceDims, NULL);

                fileDataSpace = H5Screate_simple(numDims, dataSetDims, NULL);
                hid_t sp_type = H5Tcreate(H5T_COMPOUND, sizeof(sim_params_t));
                H5Tinsert(sp_type, "total blocks", offsetof(sim_params_t, total_blocks), H5T_NATIVE_INT);
                H5Tinsert(sp_type, "time", offsetof(sim_params_t, time), H5T_NATIVE_DOUBLE);
                H5Tinsert(sp_type, "timestep", offsetof(sim_params_t, timestep), H5T_NATIVE_DOUBLE);
                H5Tinsert(sp_type, "number of steps", offsetof(sim_params_t, nsteps), H5T_NATIVE_INT);
                H5Tinsert(sp_type, "nxb", offsetof(sim_params_t, nxb), H5T_NATIVE_INT);
                H5Tinsert(sp_type, "nyb", offsetof(sim_params_t, nyb), H5T_NATIVE_INT);
                H5Tinsert(sp_type, "nzb", offsetof(sim_params_t, nzb), H5T_NATIVE_INT);

                /* define hyperslab for memory data space */
                H5Sselect_hyperslab(memDataSpace, H5S_SELECT_SET, memStart, memStride, memCount, memBlock);
                /* define hyperslab for file data space */
                H5Sselect_hyperslab(fileDataSpace, H5S_SELECT_SET, fileStart, fileStride, fileCount,
                                    fileBlock);

                MPI_Barrier(comm);
                startTime       = MPI_Wtime();
                hid_t dtgroupid = H5Gcreate(fd, "/Data/derivedData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                groupTime += (MPI_Wtime() - startTime);

                MPI_Barrier(comm);
                startTime = MPI_Wtime();
                dataSet   = H5Dcreate(dtgroupid, "simulation parameters", sp_type, fileDataSpace, H5P_DEFAULT,
                                    dataSetPropList, H5P_DEFAULT);
                dataSetTime += (MPI_Wtime() - startTime);

                MPI_Barrier(comm);
                startTime = MPI_Wtime();
                rc        = H5Dwrite(dataSet, sp_type, memDataSpace, fileDataSpace, xferPropList, sim_params);
                writeDataTime += (MPI_Wtime() - startTime);

                if (rc < 0) {
                    printf("H5Dwrite of dataSet failed with rc %d\n", rc);
                    exit(1);
                }

                MPI_Barrier(comm);
                startTime = MPI_Wtime();
                H5Dclose(dataSet);
                H5Sclose(fileDataSpace);
                H5Tclose(sp_type);
                H5Gclose(dtgroupid);
                otherCloseTime += (MPI_Wtime() - startTime);
            }

            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            H5Sclose(memDataSpace);
            H5Pclose(xferPropList);
            H5Pclose(dataSetPropList);
            H5Tclose(dtype_id);
            H5Pclose(plist_id);
            H5Gclose(topgroupid);
            otherCloseTime += (MPI_Wtime() - startTime);

            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            rc        = H5Fflush(fd, H5F_SCOPE_LOCAL);
            fflushTime += (MPI_Wtime() - startTime);

            MPI_Barrier(comm);
            startTime = MPI_Wtime();
            rc        = H5Fclose(fd);
            fcloseTime += (MPI_Wtime() - startTime);

            if (rc < 0) {
                printf("H5Fclose failed with rc %d\n", rc);
                exit(1);
            }

            free(dataSetDims);
            free(fileStart);
            free(fileStride);
            free(fileBlock);
            free(fileCount);

            MPI_Reduce(&writeDataTime, &maxWriteDataTime[loopIter], 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&readDataTime, &maxReadDataTime[loopIter], 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&dataSetTime, &maxDataSetTime[loopIter], 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&groupTime, &maxGroupTime[loopIter], 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&attrTime, &maxAttrTime[loopIter], 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&fopenTime, &maxFopenTime[loopIter], 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&fcloseTime, &maxFcloseTime[loopIter], 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&fflushTime, &maxFflushTime[loopIter], 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&otherCloseTime, &maxOtherCloseTime[loopIter], 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&rawWriteBDWTH, &minRawWriteBDWTH[loopIter], 1, MPI_DOUBLE, MPI_MIN, 0, comm);
            MPI_Reduce(&rawReadBDWTH, &minRawReadBDWTH[loopIter], 1, MPI_DOUBLE, MPI_MIN, 0, comm);

        } /* loopIter for-loop */

        /* Let rank 0 calculate and write the results */
        if (rank == 0) {

            /* Get the statistical values */
            getmoments(&maxWriteDataTime[0], NUM_ITERATIONS, &MOMWriteDataTime[0]);
            getmoments(&maxReadDataTime[0], NUM_ITERATIONS, &MOMReadDataTime[0]);
            getmoments(&maxDataSetTime[0], NUM_ITERATIONS, &MOMDataSetTime[0]);
            getmoments(&maxGroupTime[0], NUM_ITERATIONS, &MOMGroupTime[0]);
            getmoments(&maxAttrTime[0], NUM_ITERATIONS, &MOMAttrTime[0]);
            getmoments(&maxFcloseTime[0], NUM_ITERATIONS, &MOMFcloseTime[0]);
            getmoments(&maxFopenTime[0], NUM_ITERATIONS, &MOMFopenTime[0]);
            getmoments(&maxFflushTime[0], NUM_ITERATIONS, &MOMFflushTime[0]);
            getmoments(&maxOtherCloseTime[0], NUM_ITERATIONS, &MOMOtherCloseTime[0]);
            getmoments(&minRawWriteBDWTH[0], NUM_ITERATIONS, &MOMRawWriteBDWTH[0]);
            getmoments(&minRawReadBDWTH[0], NUM_ITERATIONS, &MOMRawReadBDWTH[0]);
            printf("Min      %10ld %10.6f  %12.6f %10.6f  %12.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f "
                   "%10.6f\n",
                   BufSizeTotalDouble, MOMWriteDataTime[MIN_ind], MOMRawWriteBDWTH[MIN_ind],
                   MOMReadDataTime[MIN_ind], MOMRawReadBDWTH[MIN_ind], MOMDataSetTime[MIN_ind],
                   MOMGroupTime[MIN_ind], MOMAttrTime[MIN_ind], MOMFopenTime[MIN_ind], MOMFcloseTime[MIN_ind],
                   MOMFflushTime[MIN_ind], MOMOtherCloseTime[MIN_ind]);
            printf("Med      %10ld %10.6f  %12.6f %10.6f  %12.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f "
                   "%10.6f\n",
                   BufSizeTotalDouble, MOMWriteDataTime[MED_ind], MOMRawWriteBDWTH[MED_ind],
                   MOMReadDataTime[MED_ind], MOMRawReadBDWTH[MED_ind], MOMDataSetTime[MED_ind],
                   MOMGroupTime[MED_ind], MOMAttrTime[MED_ind], MOMFopenTime[MED_ind], MOMFcloseTime[MED_ind],
                   MOMFflushTime[MED_ind], MOMOtherCloseTime[MED_ind]);
            printf("Max      %10ld %10.6f  %12.6f %10.6f  %12.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f "
                   "%10.6f\n",
                   BufSizeTotalDouble, MOMWriteDataTime[MAX_ind], MOMRawWriteBDWTH[MAX_ind],
                   MOMReadDataTime[MAX_ind], MOMRawReadBDWTH[MAX_ind], MOMDataSetTime[MAX_ind],
                   MOMGroupTime[MAX_ind], MOMAttrTime[MAX_ind], MOMFopenTime[MAX_ind], MOMFcloseTime[MAX_ind],
                   MOMFflushTime[MAX_ind], MOMOtherCloseTime[MAX_ind]);
            printf("Avg      %10ld %10.6f  %12.6f %10.6f  %12.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f "
                   "%10.6f\n",
                   BufSizeTotalDouble, MOMWriteDataTime[AVG_ind], MOMRawWriteBDWTH[AVG_ind],
                   MOMReadDataTime[AVG_ind], MOMRawReadBDWTH[AVG_ind], MOMDataSetTime[AVG_ind],
                   MOMGroupTime[AVG_ind], MOMAttrTime[AVG_ind], MOMFopenTime[AVG_ind], MOMFcloseTime[AVG_ind],
                   MOMFflushTime[AVG_ind], MOMOtherCloseTime[AVG_ind]);
            printf("Std      %10ld %10.6f  %12.6f %10.6f  %12.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f "
                   "%10.6f\n",
                   BufSizeTotalDouble, MOMWriteDataTime[STD_ind], MOMRawWriteBDWTH[STD_ind],
                   MOMReadDataTime[STD_ind], MOMRawReadBDWTH[STD_ind], MOMDataSetTime[STD_ind],
                   MOMGroupTime[STD_ind], MOMAttrTime[STD_ind], MOMFopenTime[STD_ind], MOMFcloseTime[STD_ind],
                   MOMFflushTime[STD_ind], MOMOtherCloseTime[STD_ind]);
        }

        /* Free memory for buffers */
        free(dataBuffer);
        free(checkBuffer);
        free(sim_params);
        if (rankshift != 0)
            free(dataBufferShift);

        /* Increase buffer sizes for next loop */
        for (j = 0; j < numDims; j++)
            curNEls[j] *= bufMult[j];

        MPI_Barrier(comm);
    } /* bufsize for-loop */

    H5Pclose(createPropList);
    H5Pclose(accessPropList);

    if (!keepFile) {
        unlink(testFileName);
    }

    if (rank == 0)
        printf("All done -- Finishing normally.\n");
    MPI_Barrier(comm);

    MPI_Finalize();
    return 0;
}

void
getmoments(double *ain, int n, double *momarr)
{
    int    i, j;
    double median, t, minv, maxv, avgv, stdv;

    double *a = (double *)malloc(sizeof(double) * n);
    for (i = 0; i < n; i++)
        a[i] = ain[i];

    /* 1st Loop through array to get min, max, and sum (for avg) */
    minv = a[0];
    maxv = a[0];
    avgv = a[0];
    for (i = 1; i < n; i++) {
        avgv += a[i];
        if (a[i] > maxv)
            maxv = a[i];
        if (a[i] < minv)
            minv = a[i];
    }
    avgv /= ((double)n);

    /* 2nd Loop through array to get std dev */
    stdv = 0.0;
    for (i = 0; i < n; i++)
        stdv += (a[i] - avgv) * (a[i] - avgv);
    stdv /= ((double)n);
    stdv = sqrt(stdv);

    /* Sort array 'a' to determine median */
    for (i = 1; i <= n - 1; i++) {
        for (j = 1; j <= n - i; j++) {
            if (a[j] <= a[j + 1]) {
                t        = a[j];
                a[j]     = a[j + 1];
                a[j + 1] = t;
            }
            else
                continue;
        }
    }
    /* Calculate the median (trivial, because sorted) */
    if (n % 2 == 0)
        median = (a[n / 2] + a[n / 2 + 1]) / 2.0;
    else
        median = a[n / 2 + 1];

    /* Set values in moment array */
    momarr[MIN_ind] = minv;
    momarr[MED_ind] = median;
    momarr[MAX_ind] = maxv;
    momarr[AVG_ind] = avgv;
    momarr[STD_ind] = stdv;
    free(a);
}
