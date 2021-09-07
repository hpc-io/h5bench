/*
 * vl_bench.c
 *
 *  Created on: May 29, 2020
 *      Author: tonglin
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "../commons/h5bench_util.h"
#include "../commons/async_adaptor.h"

#define ERROR_RETURN                                                                                         \
    do {                                                                                                     \
        printf("%s:L%d: failed.\n", __func__, __LINE__);                                                     \
        return -1;                                                                                           \
    } while (0)

char *FILE_PATH;
int
test_ds_append(int n_elem, int vlen)
{
    hid_t fid      = -1;
    hid_t did      = -1; /* Dataset ID */
    hid_t sid      = -1; /* Dataspace ID */
    hid_t dcpl     = -1; /* A copy of dataset creation property */
    hid_t ffapl    = -1;
    hid_t fapl     = -1;
    hid_t memtype  = -1;
    hid_t filetype = -1;
    int   SDIM     = 8;

    hsize_t dims[1]       = {0};             /* Current dimension sizes */
    hsize_t maxdims[1]    = {H5S_UNLIMITED}; /* Maximum dimension sizes */
    hsize_t chunk_dims[1] = {2048};

    int lbuf[10]; /* The data buffers */
    int i, j;     /* Local index variables */
    // h5_stat_t  sb1, sb2;                /* File info */

    if ((fapl = H5Pcreate(H5P_FILE_ACCESS)) < 0)
        ERROR_RETURN;
    if (H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST) < 0)
        ERROR_RETURN;

    if ((fid = H5Fcreate_async(FILE_PATH, H5F_ACC_TRUNC, H5P_DEFAULT, fapl, 0)) < 0)
        ERROR_RETURN;

    if (vlen == 1) {
        memtype = H5Tcopy(H5T_C_S1);
        H5Tset_size(memtype, H5T_VARIABLE);
        filetype = H5Tcopy(H5T_C_S1);
        H5Tset_size(filetype, H5T_VARIABLE);
    }

    /* Set to create a chunked dataset with extendible dimensions */
    if ((sid = H5Screate_simple(1, dims, maxdims)) < 0)
        ERROR_RETURN;
    if ((dcpl = H5Pcreate(H5P_DATASET_CREATE)) < 0)
        ERROR_RETURN;
    if (H5Pset_chunk(dcpl, 1, chunk_dims) < 0)
        ERROR_RETURN;

    /* Create the dataset */
    if (vlen != 1) {
        if ((did = H5Dcreate_async(fid, "my_test_dataset", H5T_NATIVE_INT, sid, H5P_DEFAULT, dcpl,
                                   H5P_DEFAULT, 0)) < 0)
            ERROR_RETURN;
    }
    else { // VLen
        if ((did = H5Dcreate_async(fid, "my_test_dataset", filetype, sid, H5P_DEFAULT, dcpl, H5P_DEFAULT,
                                   0)) < 0)
            ERROR_RETURN;
    }

    /* Append 6 rows to the dataset */
    unsigned long t1 = get_time_usec();
    if (vlen != 1) { // fixed length benchmark
        int dat = 0;
        for (i = 0; i < n_elem; i++) {
            dat = i;
            /* Append without boundary, callback and flush */
#ifdef DEV_VL
            if (H5Dappend(did, H5P_DEFAULT, 0, (size_t)1, H5T_NATIVE_INT, &dat) < 0)
                ERROR_RETURN;
#else
            if (H5DOappend(did, H5P_DEFAULT, 0, (size_t)1, H5T_NATIVE_INT, &dat) < 0)
                ERROR_RETURN;
#endif
        } /* end for */
    }
    else { // vlen benchmark
        char *data = "abcd";
        for (i = 0; i < n_elem; i++) {
            /* Append without boundary, callback and flush */
#ifdef DEV_VL
            if (H5Dappend(did, H5P_DEFAULT, 0, (size_t)1, memtype, &data) < 0)
                ERROR_RETURN;
#else
            if (H5DOappend(did, H5P_DEFAULT, 0, (size_t)1, memtype, &data) < 0)
                ERROR_RETURN;
#endif
        } /* end for */
    }
    char *elem_type = NULL;
    char *func_name = NULL;
    if (vlen == 1)
        elem_type = strdup("variable lenth(str)");
    else if (vlen == 0)
        elem_type = strdup("fixed length(int)");

#ifdef DEV_VL
    func_name = strdup("library level API(H5Dappend)");
#else
    func_name = strdup(" high level API(H5DOappend)");
#endif

    unsigned long t2 = get_time_usec();
    printf("Appended %d %s elements with %s, took %lu usec.\n", n_elem, elem_type, func_name, t2 - t1);
    free(elem_type);
    free(func_name);
    /* Closing */
    if (H5Dclose_async(did, 0) < 0)
        ERROR_RETURN;
    if (H5Sclose(sid) < 0)
        ERROR_RETURN;
    if (vlen == 1) {
        if (H5Tclose(filetype))
            ERROR_RETURN;
        if (H5Tclose(memtype))
            ERROR_RETURN;
    }

    if (H5Pclose(dcpl) < 0)
        ERROR_RETURN;
    if (H5Pclose(fapl) < 0)
        ERROR_RETURN;
    if (H5Fclose_async(fid, 0) < 0)
        ERROR_RETURN;

    printf("All test passed.\n");
    return 0;
}

int
main(int argc, char *argv[])
{
    printf("Usage: ./%s write_file_path FIXED/VLEN num_ops\n", argv[0]);
    FILE_PATH               = argv[1];
    int use_variable_length = -1;
    if (strcmp(argv[2], "FIXED") == 0) {
        use_variable_length = 0;
    }
    else if (strcmp(argv[2], "VLEN") == 0) {
        use_variable_length = 1;
    }
    else {
        printf("Mode not supported: [], can only be FIXED or VLEN. Use VLEN as default.\n");
        use_variable_length = 1;
    }
    test_ds_append(atoi(argv[3]), use_variable_length);
    return 0;
}
