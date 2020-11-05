/*
 * async_adaptor.h
 *
 *  Created on: Oct 21, 2020
 *      Author: tonglin
 */
#include <hdf5.h>

#ifndef COMMONS_ASYNC_ADAPTOR_H_
#define COMMONS_ASYNC_ADAPTOR_H_
#ifndef H5ES_WAIT_FOREVER
#ifndef _H5ESpublic_H
typedef int H5ES_status_t;// an enum type exists in newer version, this line is just to make compiler happy with older version headers
#endif

static hid_t H5EScreate(void) {
    return 0;
}
static herr_t H5EStest(hid_t es_id, H5ES_status_t *status) {
    return 0;
}
static herr_t H5ESwait(hid_t es_id, uint64_t timeout, H5ES_status_t *status) {
    return 0;
}
static herr_t H5EScancel(hid_t es_id, H5ES_status_t *status) {
    return 0;
}
static herr_t H5ESget_count(hid_t es_id, size_t *count) {
    return 0;
}
static herr_t H5ESget_err_status(hid_t es_id, hbool_t *err_occurred) {
    return 0;
}
static herr_t H5ESget_err_count(hid_t es_id, size_t *num_errs) {
    return 0;
}
static herr_t H5ESclose(hid_t es_id) {
    return 0;
}

static hid_t H5Acreate_async(hid_t loc_id, const char *attr_name, hid_t type_id, hid_t space_id,
        hid_t acpl_id, hid_t aapl_id, hid_t es_id) {
    return H5Acreate2( loc_id, attr_name, type_id, space_id, acpl_id, aapl_id);
}

static hid_t H5Acreate_by_name_async(hid_t loc_id, const char *obj_name, const char *attr_name, hid_t type_id,
        hid_t space_id, hid_t acpl_id, hid_t aapl_id, hid_t lapl_id, hid_t es_id) {
    return H5Acreate_by_name(loc_id, obj_name, attr_name, type_id, space_id, acpl_id, aapl_id, lapl_id);
}

static hid_t H5Aopen_async(hid_t obj_id, const char *attr_name, hid_t aapl_id, hid_t es_id) {
    return H5Aopen(obj_id, attr_name, aapl_id);
}

static hid_t H5Aopen_by_name_async(hid_t loc_id, const char *obj_name, const char *attr_name, hid_t aapl_id,
        hid_t lapl_id, hid_t es_id) {
    return H5Aopen_by_name(loc_id, obj_name, attr_name, aapl_id, lapl_id);
}

static hid_t H5Aopen_by_idx_async(hid_t loc_id, const char *obj_name, H5_index_t idx_type, H5_iter_order_t order,
        hsize_t n, hid_t aapl_id, hid_t lapl_id, hid_t es_id) {
    return H5Aopen_by_idx(loc_id, obj_name, idx_type, order, n, aapl_id, lapl_id);
}

static herr_t H5Awrite_async(hid_t attr_id, hid_t type_id, const void *buf, hid_t es_id) {
    return H5Awrite(attr_id, type_id, buf);
}

static herr_t H5Aread_async(hid_t attr_id, hid_t dtype_id, void *buf, hid_t es_id) {
    return H5Aread(attr_id, dtype_id, buf);
}

static herr_t H5Aclose_async(hid_t attr_id, hid_t es_id) {
    return H5Aclose(attr_id);
}

static herr_t H5Arename_async(hid_t loc_id, const char *old_name, const char *new_name, hid_t es_id) {
    return H5Arename(loc_id, old_name, new_name);
}

static herr_t H5Arename_by_name_async(hid_t loc_id, const char *obj_name, const char *old_attr_name,
        const char *new_attr_name, hid_t lapl_id, hid_t es_id) {
    return H5Arename_by_name(loc_id, obj_name, old_attr_name, new_attr_name, lapl_id);
}

static htri_t H5Aexists_async(hid_t obj_id, const char *attr_name, hid_t es_id) {
    return H5Aexists(obj_id, attr_name);
}

static htri_t H5Aexists_by_name_async(hid_t loc_id, const char *obj_name, const char *attr_name,
        hid_t lapl_id, hid_t es_id) {
    return H5Aexists_by_name(loc_id, obj_name, attr_name, lapl_id);
}

static hid_t H5Dcreate_async(hid_t loc_id, const char *name, hid_t type_id, hid_t space_id, hid_t lcpl_id,
        hid_t dcpl_id, hid_t dapl_id, hid_t es_id) {
    return H5Dcreate2(loc_id, name, type_id, space_id, lcpl_id, dcpl_id, dapl_id);
}

static herr_t H5Dclose_async(hid_t dset_id, hid_t es_id) {
    return H5Dclose(dset_id);
}

static herr_t H5Dread_async(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id,
        hid_t plist_id, void *buf_out, hid_t es_id) {
    return H5Dread(dset_id, mem_type_id, mem_space_id, file_space_id, plist_id, buf_out);
}

static herr_t H5Dwrite_async(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id,
        hid_t plist_id, const void *buf, hid_t es_id) {
    return H5Dwrite(dset_id, mem_type_id, mem_space_id, file_space_id, plist_id, buf);
}

static hid_t H5Fcreate_async(const char *filename, unsigned flags, hid_t fcpl_id, hid_t fapl_id, hid_t es_id) {
    return H5Fcreate(filename, flags, fcpl_id, fapl_id);
}

static hid_t H5Fopen_async(const char *filename, unsigned flags, hid_t access_plist, hid_t es_id) {
    return H5Fopen(filename, flags, access_plist);
}

static herr_t H5Fclose_async(hid_t file_id, hid_t es_id) {
    return H5Fclose(file_id);
}

static hid_t H5Gcreate_async(hid_t loc_id, const char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id,
        hid_t es_id) {
    return H5Gcreate2(loc_id, name, lcpl_id, gcpl_id, gapl_id);
}

static hid_t H5Gopen_async(hid_t loc_id, const char *name, hid_t gapl_id, hid_t es_id) {
    return H5Gopen2(loc_id, name, gapl_id);
}
static herr_t H5Gget_info_async(hid_t loc_id, H5G_info_t *group_info_out, hid_t es_id) {
    return H5Gget_info(loc_id, group_info_out);
}

static herr_t H5Gget_info_by_name_async(hid_t loc_id, const char *name, H5G_info_t *group_info,
        hid_t lapl_id, hid_t es_id) {
    return H5Gget_info_by_name(loc_id, name, group_info, lapl_id);
}

static herr_t H5Gget_info_by_idx_async(hid_t loc_id, const char *group_name,
        H5_index_t idx_type, H5_iter_order_t order, hsize_t n, H5G_info_t *group_info,
        hid_t lapl_id, hid_t es_id) {
    return H5Gget_info_by_idx(loc_id, group_name, idx_type, order, n, group_info, lapl_id);
}

static herr_t H5Gclose_async(hid_t group_id, hid_t es_id) {
    return H5Gclose(group_id);
}
#endif //H5ES_WAIT_FOREVER

#endif /* COMMONS_ASYNC_ADAPTOR_H_ */
