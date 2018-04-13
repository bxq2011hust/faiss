/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Copyright 2004-present Facebook. All Rights Reserved.
   Inverted list structure.
*/

#include "IndexIVFHNSW.h"

#include <cstdio>

#include "utils.h"

#include "FaissAssert.h"
#include "IndexFlat.h"
#include "AuxIndexStructures.h"

namespace faiss
{

/*****************************************
 * HNSWInvertedLists implementation
 ******************************************/

HNSWInvertedLists::HNSWInvertedLists(size_t nlist, size_t code_size, size_t M) : InvertedLists(nlist, code_size)
{
    indexes.reserve(nlist);
    for (size_t i = 0; i < nlist; ++i)
    {
        indexes.push_back(new IndexIDMap(new IndexHNSWFlat(code_size / sizeof(float), M)));
        indexes[i]->own_fields = true;
    }
}

size_t HNSWInvertedLists::add_entries(
    size_t list_no, size_t n_entry,
    const idx_t *ids_in, const uint8_t *code)
{
    if (n_entry == 0)
        return 0;
    assert(list_no < nlist);
    IndexIDMap *index = indexes[list_no];
    index->add_with_ids(n_entry, (const float *)code, ids_in);
    return index->id_map.size();
}

size_t HNSWInvertedLists::list_size(size_t list_no) const
{
    assert(list_no < nlist);
    return indexes[list_no]->id_map.size();
}

const uint8_t *HNSWInvertedLists::get_codes(size_t list_no) const
{
    assert(list_no < nlist);
    IndexFlatL2 *l2Flat = dynamic_cast<IndexFlatL2 *>((dynamic_cast<IndexHNSW *>(indexes[list_no]->index))->storage);
    return (const uint8_t *)l2Flat->xb.data();
}

const InvertedLists::idx_t *HNSWInvertedLists::get_ids(size_t list_no) const
{
    assert(list_no < nlist);
    return indexes[list_no]->id_map.data();
}

void HNSWInvertedLists::resize(size_t list_no, size_t new_size)
{
}

void HNSWInvertedLists::update_entries(
    size_t list_no, size_t offset, size_t n_entry,
    const idx_t *ids_in, const uint8_t *codes_in)
{
    // assert(list_no < nlist);
    // assert(n_entry + offset <= ids[list_no].size());
    FAISS_THROW_MSG("HNSW InvertedLists can't update");
}

HNSWInvertedLists::~HNSWInvertedLists()
{
    for (size_t i = 0; i < nlist; ++i)
    {
        delete indexes[i];
        indexes[i] = nullptr;
    }
}

/*****************************************
 * IndexIVFHNSW implementation
 ******************************************/

IndexIVFHNSW::IndexIVFHNSW(Index *quantizer,
                           size_t d, size_t nlist, size_t M, size_t w, size_t k2_, MetricType metric) : IndexIVF(quantizer, d, nlist, sizeof(float) * d, metric), k2(k2_)
{
    code_size = sizeof(float) * d;
    delete invlists;
    invlists = new HNSWInvertedLists(nlist, code_size, M);
    nprobe = w;
}

void IndexIVFHNSW::add_with_ids(idx_t n, const float *x, const long *xids)
{
    add_core(n, x, xids, nullptr);
}

void IndexIVFHNSW::add_core(idx_t n, const float *x, const long *xids,
                            const long *precomputed_idx)

{
    FAISS_THROW_IF_NOT(is_trained);
    assert(invlists);
    FAISS_THROW_IF_NOT_MSG(!(maintain_direct_map && xids),
                           "cannot have direct map and add with ids");
    const long *idx;
    ScopeDeleter<long> del;

    if (precomputed_idx)
    {
        idx = precomputed_idx;
    }
    else
    {
        long *idx0 = new long[n];
        del.set(idx0);
        float * dis = new float[n];
        quantizer->assign(n, x, idx0);
        // quantizer->search(n, x, 1, dis, idx0);
        idx = idx0;
        // delete [] dis;
    }
    long n_add = 0;
    for (size_t i = 0; i < n; i++)
    {
        long id = xids ? xids[i] : ntotal + i;
        long list_no = idx[i];

        if (list_no < 0)
            continue;
        const float *xi = x + i * d;
        size_t offset = invlists->add_entry(
            list_no, id, (const uint8_t *)xi);

        if (maintain_direct_map)
            direct_map.push_back(list_no << 32 | offset);
        n_add++;
    }
    if (verbose)
    {
        printf("IndexIVFHNSW::add_core: added %ld / %ld vectors\n",
               n_add, n);
    }
    ntotal += n_add;
}

namespace
{

void search_knn_inner_product(const IndexIVFHNSW &ivf,
                              size_t nx,
                              const float *x,
                              const long *keys,
                              float_minheap_array_t *res,
                              bool store_pairs)
{
    FAISS_THROW_MSG("HNSW inner_product isn't supported yet.");
    //     const size_t k = res->k;
    //     size_t nlistv = 0, ndis = 0;
    //     size_t d = ivf.d;

    // #pragma omp parallel for reduction(+: nlistv, ndis)
    //     for (size_t i = 0; i < nx; i++) {
    //         const float * xi = x + i * d;
    //         const long * keysi = keys + i * ivf.nprobe;
    //         float * __restrict simi = res->get_val (i);
    //         long * __restrict idxi = res->get_ids (i);
    //         minheap_heapify (k, simi, idxi);
    //         size_t nscan = 0;

    //         for (size_t ik = 0; ik < ivf.nprobe; ik++) {
    //             long key = keysi[ik];  /* select the list  */
    //             if (key < 0) {
    //                 // not enough centroids for multiprobe
    //                 continue;
    //             }
    //             FAISS_THROW_IF_NOT_FMT (
    //                 key < (long) ivf.nlist,
    //                 "Invalid key=%ld  at ik=%ld nlist=%ld\n",
    //                 key, ik, ivf.nlist);

    //             nlistv++;
    //             size_t list_size = ivf.invlists->list_size(key);
    //             const float * list_vecs =
    //                 (const float*)ivf.invlists->get_codes (key);
    //             const Index::idx_t * ids = store_pairs ? nullptr :
    //                 ivf.invlists->get_ids (key);

    //             for (size_t j = 0; j < list_size; j++) {
    //                 const float * yj = list_vecs + d * j;
    //                 float ip = fvec_inner_product (xi, yj, d);
    //                 if (ip > simi[0]) {
    //                     minheap_pop (k, simi, idxi);
    //                     long id = store_pairs ? (key << 32 | j) : ids[j];
    //                     minheap_push (k, simi, idxi, ip, id);
    //                 }
    //             }
    //             nscan += list_size;
    //             if (ivf.max_codes && nscan >= ivf.max_codes)
    //                 break;
    //         }
    //         ndis += nscan;
    //         minheap_reorder (k, simi, idxi);
    //     }
    //     indexIVF_stats.nq += nx;
    //     indexIVF_stats.nlist += nlistv;
    //     indexIVF_stats.ndis += ndis;
}

void search_knn_L2sqr(const IndexIVFHNSW &ivf,
                      size_t nx,
                      const float *x,
                      const long *keys,
                      float_maxheap_array_t *res,
                      bool store_pairs)
{
    const size_t k = res->k;
    size_t nlistv = 0, ndis = 0;
    size_t d = ivf.d;
#pragma omp parallel for reduction(+ \
                                   : nlistv, ndis)
    for (size_t i = 0; i < nx; i++)
    {
        const float *xi = x + i * d;
        const long *keysi = keys + i * ivf.nprobe;
        float *__restrict disi = res->get_val(i);
        long *__restrict idxi = res->get_ids(i);
        maxheap_heapify(k, disi, idxi);

        size_t nscan = 0;

        for (size_t ik = 0; ik < ivf.nprobe; ik++)
        {
            long key = keysi[ik]; /* select the list  */
            if (key < 0)
            {
                // not enough centroids for multiprobe
                continue;
            }
            FAISS_THROW_IF_NOT_FMT(
                key < (long)ivf.nlist,
                "Invalid key=%ld  at ik=%ld nlist=%ld\n",
                key, ik, ivf.nlist);

            nlistv++;
            size_t list_size = std::min(ivf.invlists->list_size(key), ivf.k2);
            float *dists = new float[list_size];
            Index::idx_t *ids = store_pairs ? nullptr : new Index::idx_t[list_size];
            ScopeDeleter<float> del1(dists);
            ScopeDeleter<Index::idx_t> del2(ids);
            (dynamic_cast<HNSWInvertedLists *>(ivf.invlists))->indexes[key]->search(1, xi, list_size, dists, ids);
            for (size_t j = 0; j < list_size; j++)
            {
                float disij = dists[j];
                if (disij < disi[0])
                {
                    maxheap_pop(k, disi, idxi);
                    long id = store_pairs ? (key << 32 | j) : ids[j];
                    maxheap_push(k, disi, idxi, disij, id);
                }
            }
            nscan += list_size;
            if (ivf.max_codes && nscan >= ivf.max_codes)
                break;
        }
        ndis += nscan;
        maxheap_reorder(k, disi, idxi);
    }
    indexIVF_stats.nq += nx;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
}

} // anonymous namespace

void IndexIVFHNSW::search_preassigned(idx_t n, const float *x, idx_t k,
                                      const idx_t *idx,
                                      const float * /* coarse_dis */,
                                      float *distances, idx_t *labels,
                                      bool store_pairs) const
{
    if (metric_type == METRIC_INNER_PRODUCT)
    {
        float_minheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        search_knn_inner_product(*this, n, x, idx, &res, store_pairs);
    }
    else if (metric_type == METRIC_L2)
    {
        float_maxheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        search_knn_L2sqr(*this, n, x, idx, &res, store_pairs);
    }
}

void IndexIVFHNSW::range_search(idx_t nx, const float *x, float radius,
                                RangeSearchResult *result) const
{
    //     idx_t * keys = new idx_t [nx * nprobe];
    //     ScopeDeleter<idx_t> del (keys);
    //     quantizer->assign (nx, x, keys, nprobe);

    // #pragma omp parallel
    //     {
    //         RangeSearchPartialResult pres(result);

    //         for (size_t i = 0; i < nx; i++) {
    //             const float * xi = x + i * d;
    //             const long * keysi = keys + i * nprobe;

    //             RangeSearchPartialResult::QueryResult & qres =
    //                 pres.new_result (i);

    //             for (size_t ik = 0; ik < nprobe; ik++) {
    //                 long key = keysi[ik];  /* select the list  */
    //                 if (key < 0 || key >= (long) nlist) {
    //                     fprintf (stderr, "Invalid key=%ld  at ik=%ld nlist=%ld\n",
    //                              key, ik, nlist);
    //                     throw;
    //                 }

    //                 const size_t list_size = invlists->list_size(key);
    //                 const float * list_vecs =
    //                     (const float*)invlists->get_codes (key);
    //                 const Index::idx_t * ids = invlists->get_ids (key);

    //                 for (size_t j = 0; j < list_size; j++) {
    //                     const float * yj = list_vecs + d * j;
    //                     if (metric_type == METRIC_L2) {
    //                         float disij = fvec_L2sqr (xi, yj, d);
    //                         if (disij < radius) {
    //                             qres.add (disij, ids[j]);
    //                         }
    //                     } else if (metric_type == METRIC_INNER_PRODUCT) {
    //                         float disij = fvec_inner_product(xi, yj, d);
    //                         if (disij > radius) {
    //                             qres.add (disij, ids[j]);
    //                         }
    //                     }
    //                 }
    //             }
    //         }

    //         pres.finalize ();
    //     }
}

void IndexIVFHNSW::update_vectors(int n, idx_t *new_ids, const float *x)
{
}

void IndexIVFHNSW::reconstruct_from_offset(long list_no, long offset,
                                           float *recons) const
{
    memcpy(recons, invlists->get_single_code(list_no, offset), code_size);
}

} // namespace faiss
