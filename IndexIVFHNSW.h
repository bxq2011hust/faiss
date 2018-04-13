/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_IVF_HNSW_H
#define FAISS_INDEX_IVF_HNSW_H

#include "IndexIVF.h"
#include "IndexHNSW.h"
#include "MetaIndexes.h"


namespace faiss {

struct HNSWInvertedLists: InvertedLists {
    
    std::vector<faiss::IndexIDMap *> indexes;
    HNSWInvertedLists (size_t nlist, size_t code_size, size_t M);

    size_t list_size(size_t list_no) const override;
    const uint8_t * get_codes (size_t list_no) const override;
    const idx_t * get_ids (size_t list_no) const override;

    size_t add_entries (
           size_t list_no, size_t n_entry,
           const idx_t* ids, const uint8_t *code) override;

    void update_entries (size_t list_no, size_t offset, size_t n_entry,
                         const idx_t *ids, const uint8_t *code) override;

    void resize (size_t list_no, size_t new_size) override;
    void set_efSearch(size_t efSearch);
    virtual ~HNSWInvertedLists ();
};

/** Inverted file with stored vectors. Here the inverted file
 * pre-selects the vectors to be searched, but they are not otherwise
 * encoded, the code array just contains the raw float entries.
 */
struct IndexIVFHNSW: IndexIVF {

    /// each IndexHNSW get k2 nearest neighbors of query
    float k_factor;
    IndexIVFHNSW (
            Index * quantizer, size_t d, size_t nlist_,size_t M, 
            MetricType = METRIC_L2);

    /// same as add_with_ids, with precomputed coarse quantizer
    virtual void add_core (idx_t n, const float * x, const long *xids,
                   const long *precomputed_idx);

    /// implemented for all IndexIVF* classes
    void add_with_ids(idx_t n, const float* x, const long* xids) override;

    void search_preassigned (idx_t n, const float *x, idx_t k,
                             const idx_t *assign,
                             const float *centroid_dis,
                             float *distances, idx_t *labels,
                             bool store_pairs) const override;

    void range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result) const override;

    /** Update a subset of vectors.
     *
     * The index must have a direct_map
     *
     * @param nv     nb of vectors to update
     * @param idx    vector indices to update, size nv
     * @param v      vectors of new values, size nv*d
     */
    void update_vectors (int nv, idx_t *idx, const float *v);

    void reconstruct_from_offset (long list_no, long offset,
                                  float* recons) const override;

    IndexIVFHNSW () {}
};


} // namespace faiss

#endif
