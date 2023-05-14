// https://github.com/MojtabaValizadeh/paresy

#include <set>
#include <map>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <warpcore/hash_set.cuh>

using UINT64 = std::uint64_t;

inline
cudaError_t checkCuda(cudaError_t res) {
#ifndef MEASUREMENT_MODE
  if (res != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(res));
    assert(res == cudaSuccess);
  }
#endif
  return res; 
}

// This version keeps the guide table in the constant memory
__constant__ UINT64 d_guideTable[64 * 128];

// Finding the left and right indices that makes the final RE to bring to the host later
__global__ void generateResIndices(
    const int index, 
    const int alphabetSize, 
    const int *d_leftIdx, 
    const int *d_rightIdx, 
    int *d_FinalREIdx)
{

    int resIdx = 0;
    while (d_FinalREIdx[resIdx] != -1) resIdx++;
    int queue[600];
    queue[0] = index;
    int head = 0;
    int tail = 1;
    while (head < tail) {
        int re = queue[head];
        int l = d_leftIdx[re];
        int r = d_rightIdx[re];
        d_FinalREIdx[resIdx++] = re;
        d_FinalREIdx[resIdx++] = l;
        d_FinalREIdx[resIdx++] = r;
        if (l >= alphabetSize) queue[tail++] = l;
        if (r >= alphabetSize) queue[tail++] = r;
        head++;
    }

}

// Initialising the hashSets with empty, epsilon and alphabet before starting the enumeration
template<class hash_set_t>
__global__ void hashSetsInitialisation(
    const int alphabetSize, 
    hash_set_t cHashSet, 
    hash_set_t iHashSet, 
    UINT64 *dev_cache)
{

    const auto group = warpcore::cg::tiled_partition <1> 
    (warpcore::cg::this_thread_block());

    int H, L;
    UINT64 HL;

    // Adding empty to the hashSet
    H = cHashSet.insert(UINT64(0), group);
    L = cHashSet.insert(UINT64(0), group);
    H = (H > 0) ? H : -H;
    L = (L > 0) ? L : -L;
    HL = H; HL <<= 32; HL |= L;
    iHashSet.insert(HL, group);

    // Adding eps to the hashSet
    H = cHashSet.insert(UINT64(0), group);
    L = cHashSet.insert(UINT64(1), group);
    H = (H > 0) ? H : -H;
    L = (L > 0) ? L : -L;
    HL = H; HL <<= 32; HL |= L;
    iHashSet.insert(HL, group);

    // Adding alphabet to the hashSet
    for (int i = 0; i < alphabetSize; ++i) {
        H = cHashSet.insert(dev_cache[2 * i], group);
        L = cHashSet.insert(dev_cache[2 * i + 1], group);
        H = (H > 0) ? H : -H;
        L = (L > 0) ? L : -L;
        HL = H; HL <<= 32; HL |= L;
        iHashSet.insert(HL, group);
    }

}

// Generating (r)? for r in indices between idx1 and idx2 in the language cache
template<class hash_set_t>
__global__ void QuestionMark(
    const int idx1, 
    const int idx2, 
    bool onTheFly, 
    UINT64 *d_langCache, 
    UINT64 *d_temp_langCache, 
    int *d_temp_leftIdx, 
    int *d_temp_rightIdx, 
    hash_set_t cHashSet, 
    hash_set_t iHashSet, 
    const UINT64 hPosBits, 
    const UINT64 lPosBits,
    const UINT64 hNegBits, 
    const UINT64 lNegBits, 
    int *d_FinalREIdx)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < idx2 - idx1 + 1) {

        UINT64 hCS = d_langCache[(idx1 + tid) * 2];
        UINT64 lCS = d_langCache[(idx1 + tid) * 2 + 1] | 1;

        if (onTheFly) {

            if (( hCS & hPosBits) == hPosBits && 
                ( lCS & lPosBits) == lPosBits &&
                (~hCS & hNegBits) == hNegBits && 
                (~lCS & lNegBits) == lNegBits) {
                *d_FinalREIdx = tid;
                d_temp_langCache[tid * 2] = hCS;
                d_temp_langCache[tid * 2 + 1] = lCS;
                d_temp_leftIdx[tid] = idx1 + tid;
                d_temp_rightIdx[tid] = 0;
            }

        } else {

            const auto group = warpcore::cg::tiled_partition <1> 
            (warpcore::cg::this_thread_block());
            int H = cHashSet.insert(hCS, group);
            int L = cHashSet.insert(lCS, group);
            H = (H > 0) ? H : -H;
            L = (L > 0) ? L : -L;
            UINT64 HL = H; HL <<= 32; HL |= L;
            bool CS_is_unique = (iHashSet.insert(HL, group) > 0) ? false : true;

            if (CS_is_unique) {
                d_temp_langCache[tid * 2] = hCS;
                d_temp_langCache[tid * 2 + 1] = lCS;
                d_temp_leftIdx[tid] = idx1 + tid;
                d_temp_rightIdx[tid] = 0; // just to avoid getting removed
                if (( hCS & hPosBits) == hPosBits && 
                    ( lCS & lPosBits) == lPosBits &&
                    (~hCS & hNegBits) == hNegBits && 
                    (~lCS & lNegBits) == lNegBits)
                    atomicCAS(d_FinalREIdx, -1, tid);
            } else {
                d_temp_langCache[tid * 2] = UINT64(0) - 1;
                d_temp_langCache[tid * 2 + 1] = UINT64(0) - 1;
                d_temp_leftIdx[tid] = -1;
                d_temp_rightIdx[tid] = -1;
            }

        }

    }

}

// Generating (r)* for r in indices between idx1 and idx2 in the language cache
template<class hash_set_t>
__global__ void Star(
    const int idx1, 
    const int idx2, 
    bool onTheFly, 
    UINT64 *d_langCache, 
    UINT64 *d_temp_langCache, 
    int *d_temp_leftIdx, 
    int *d_temp_rightIdx, 
    hash_set_t cHashSet, 
    hash_set_t iHashSet,
    const int alphabetSize, 
    const int ICsize, 
    const int gtColumns, 
    const UINT64 hPosBits, 
    const UINT64 lPosBits, 
    const UINT64 hNegBits, 
    const UINT64 lNegBits, 
    int *d_FinalREIdx)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < idx2 - idx1 + 1){

        UINT64 hCS = d_langCache[(idx1 + tid) * 2];
        UINT64 lCS = d_langCache[(idx1 + tid) * 2 + 1] | 1;

        int ix = alphabetSize + 1;
        UINT64 c = 1 << ix;

        while (ix < 63 && ix < ICsize){
            if (!(lCS & c)){
                int index = ix * gtColumns;
                while (d_guideTable[index]){
                    if ((d_guideTable[index] & lCS) && 
                    (d_guideTable[index + 1] & lCS))
                    {lCS |= c; break;}
                    index += 2;
                }
            }
            c <<= 1; ix++;
        }

        c = 1;

        while (ix < ICsize){
            if (!(hCS & c)){
                int index = ix * gtColumns;
                while (d_guideTable[index] || d_guideTable[index + 1]){
                    if (((d_guideTable[index]     & hCS) || 
                         (d_guideTable[index + 1] & lCS))
                    && (( d_guideTable[index + 2] & hCS) || 
                         (d_guideTable[index + 3] & lCS)))
                    {hCS |= c; break;}
                    index += 4;
                }
            }
            c <<= 1; ix++;
        }

        if (onTheFly) {

            if (( hCS & hPosBits) == hPosBits && 
                ( lCS & lPosBits) == lPosBits &&
                (~hCS & hNegBits) == hNegBits && 
                (~lCS & lNegBits) == lNegBits) {
                *d_FinalREIdx = tid;
                d_temp_langCache[tid * 2] = hCS;
                d_temp_langCache[tid * 2 + 1] = lCS;
                d_temp_leftIdx[tid] = idx1 + tid;
                d_temp_rightIdx[tid] = 0;
            }

        } else {

            const auto group = warpcore::cg::tiled_partition <1> 
            (warpcore::cg::this_thread_block());
            int H = cHashSet.insert(hCS, group);
            int L = cHashSet.insert(lCS, group);
            H = (H > 0) ? H : -H;
            L = (L > 0) ? L : -L;
            UINT64 HL = H; HL <<= 32; HL |= L;
            bool CS_is_unique = (iHashSet.insert(HL, group) > 0) ? false : true;

            if (CS_is_unique) {
                d_temp_langCache[tid * 2] = hCS;
                d_temp_langCache[tid * 2 + 1] = lCS;
                d_temp_leftIdx[tid] = idx1 + tid;
                d_temp_rightIdx[tid] = 0; // just to avoid getting removed
                if (( hCS & hPosBits) == hPosBits && 
                    ( lCS & lPosBits) == lPosBits &&
                    (~hCS & hNegBits) == hNegBits && 
                    (~lCS & lNegBits) == lNegBits) 
                    atomicCAS(d_FinalREIdx, -1, tid);
            } else {
                d_temp_langCache[tid * 2] = UINT64(0) - 1;
                d_temp_langCache[tid * 2 + 1] = UINT64(0) - 1;
                d_temp_leftIdx[tid] = -1;
                d_temp_rightIdx[tid] = -1;
            }

        }

    }

}

// Generating r1.r2 and r2.r1
// For r1 in indices between idx1 and idx2 in the language cache
// For r2 in indices between idx3 and idx4 in the language cache
template<class hash_set_t>
__global__ void Concat(
    const int idx1, 
    const int idx2, 
    const int idx3, 
    const int idx4, 
    bool onTheFly, 
    UINT64 *d_langCache, 
    UINT64 *d_temp_langCache, 
    int *d_temp_leftIdx, 
    int *d_temp_rightIdx, 
    hash_set_t cHashSet, 
    hash_set_t iHashSet, 
    const int alphabetSize, 
    const int ICsize, 
    const int gtColumns,
    const UINT64 hPosBits, 
    const UINT64 lPosBits, 
    const UINT64 hNegBits, 
    const UINT64 lNegBits, 
    int *d_FinalREIdx)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < (idx4 - idx3 + 1) * (idx2 - idx1 + 1)){

        int ldx = idx1 + tid / (idx4 - idx3 + 1);
        UINT64 hleft = d_langCache[ldx * 2];
        UINT64 lleft = d_langCache[ldx * 2 + 1];

        int rdx = idx3 + tid % (idx4 - idx3 + 1);
        UINT64 hright = d_langCache[rdx * 2];
        UINT64 lright = d_langCache[rdx * 2 + 1];

        UINT64 hCS1{};
        UINT64 lCS1{};

        if (lleft & 1)  {hCS1 |= hright; lCS1 |= lright;}
        if (lright & 1) {hCS1 |= hleft;  lCS1 |= lleft;}
        
        UINT64 hCS2 = hCS1;
        UINT64 lCS2 = lCS1;

        int ix = alphabetSize + 1;
        UINT64 c = 1 << ix;

        while (ix < 63 && ix < ICsize){

            if (!(lCS1 & c)){
                int index = ix * gtColumns;
                while (d_guideTable[index]){
                    if ((d_guideTable[index] & lleft) && 
                    (d_guideTable[index + 1] & lright))
                    {lCS1 |= c; break;}
                    index += 2;
                }
            }

            if (!(lCS2 & c)){
                int index = ix * gtColumns;
                while (d_guideTable[index]){
                    if ((d_guideTable[index] & lright) && 
                    (d_guideTable[index + 1] & lleft))
                    {lCS2 |= c; break;}
                    index += 2;
                }
            }

            c <<= 1; ix++;

        }

        c = 1;
        
        while (ix < ICsize){

            if (!(hCS1 & c)){
                int index = ix * gtColumns;
                while (d_guideTable[index] || d_guideTable[index + 1]){
                    if (((d_guideTable[index]     & hleft)  || 
                         (d_guideTable[index + 1] & lleft))
                    && (( d_guideTable[index + 2] & hright) || 
                         (d_guideTable[index + 3] & lright)))
                    {hCS1 |= c; break;}
                    index += 4;
                }
            }

            if (!(hCS2 & c)){
                int index = ix * gtColumns;
                while (d_guideTable[index] || d_guideTable[index + 1]){
                    if (((d_guideTable[index]     & hright) || 
                         (d_guideTable[index + 1] & lright))
                    && (( d_guideTable[index + 2] & hleft)  || 
                         (d_guideTable[index + 3] & lleft)))
                    {hCS2 |= c; break;}
                    index += 4;
                }
            }

            c <<= 1; ix++;

        }

        if (onTheFly) {

            if ((hCS1 & hPosBits) == hPosBits && 
                (lCS1 & lPosBits) == lPosBits &&
                (~hCS1 & hNegBits) == hNegBits && 
                (~lCS1 & lNegBits) == lNegBits) {
                *d_FinalREIdx = tid * 2;
                d_temp_langCache[tid * 4] = hCS1;
                d_temp_langCache[tid * 4 + 1] = lCS1;
                d_temp_leftIdx[tid * 2] = ldx;
                d_temp_rightIdx[tid * 2] = rdx;
            } else if ((hCS2 & hPosBits) == hPosBits && 
                       (lCS2 & lPosBits) == lPosBits &&
                      (~hCS2 & hNegBits) == hNegBits && 
                      (~lCS2 & lNegBits) == lNegBits) {
                *d_FinalREIdx = tid * 2 + 1;
                d_temp_langCache[tid * 4 + 2] = hCS2;
                d_temp_langCache[tid * 4 + 3] = lCS2;
                d_temp_leftIdx[tid * 2 + 1] = rdx;
                d_temp_rightIdx[tid * 2 + 1] = ldx;
            }

        } else {

            const auto group = warpcore::cg::tiled_partition <1> 
            (warpcore::cg::this_thread_block());
            int H, L; UINT64 HL;

            H = cHashSet.insert(hCS1, group);
            L = cHashSet.insert(lCS1, group);
            H = (H > 0) ? H : -H;
            L = (L > 0) ? L : -L;
            HL = H; HL <<= 32; HL |= L;
            bool CS1_is_unique = (iHashSet.insert(HL, group) > 0) ? false : true;

            H = cHashSet.insert(hCS2, group);
            L = cHashSet.insert(lCS2, group);
            H = (H > 0) ? H : -H;
            L = (L > 0) ? L : -L;
            HL = H; HL <<= 32; HL |= L;
            bool CS2_is_unique = (iHashSet.insert(HL, group) > 0) ? false : true;

            if (CS1_is_unique) {
                d_temp_langCache[tid * 4] = hCS1;
                d_temp_langCache[tid * 4 + 1] = lCS1;
                d_temp_leftIdx[tid * 2] = ldx;
                d_temp_rightIdx[tid * 2] = rdx;
                if (( hCS1 & hPosBits) == hPosBits && 
                    ( lCS1 & lPosBits) == lPosBits &&
                    (~hCS1 & hNegBits) == hNegBits && 
                    (~lCS1 & lNegBits) == lNegBits) 
                    atomicCAS(d_FinalREIdx, -1, tid * 2);
            } else {
                d_temp_langCache[tid * 4] = UINT64(0) - 1;
                d_temp_langCache[tid * 4 + 1] = UINT64(0) - 1;
                d_temp_leftIdx[tid * 2] = -1;
                d_temp_rightIdx[tid * 2] = -1;
            }

            if (CS2_is_unique) {
                d_temp_langCache[tid * 4 + 2] = hCS2;
                d_temp_langCache[tid * 4 + 3] = lCS2;
                d_temp_leftIdx[tid * 2 + 1] = rdx;
                d_temp_rightIdx[tid * 2 + 1] = ldx;
                if (( hCS2 & hPosBits) == hPosBits && 
                    ( lCS2 & lPosBits) == lPosBits &&
                    (~hCS2 & hNegBits) == hNegBits && 
                    (~lCS2 & lNegBits) == lNegBits) 
                    atomicCAS(d_FinalREIdx, -1, tid * 2 + 1);
            } else {
                d_temp_langCache[tid * 4 + 2] = UINT64(0) - 1;
                d_temp_langCache[tid * 4 + 3] = UINT64(0) - 1;
                d_temp_leftIdx[tid * 2 + 1] = -1;
                d_temp_rightIdx[tid * 2 + 1] = -1;
            }

        }

    }

}

// Generating r1+r2 (union)
// For r1 in indices between idx1 and idx2 in the language cache
// For r2 in indices between idx3 and idx4 in the language cache
template<class hash_set_t>
__global__ void Or(
    const int idx1, 
    const int idx2, 
    const int idx3, 
    const int idx4, 
    bool onTheFly, 
    UINT64 *d_langCache, 
    UINT64 *d_temp_langCache,
    int *d_temp_leftIdx, 
    int *d_temp_rightIdx, 
    hash_set_t cHashSet, 
    hash_set_t iHashSet, 
    const UINT64 hPosBits, 
    const UINT64 lPosBits, 
    const UINT64 hNegBits, 
    const UINT64 lNegBits, 
    int *d_FinalREIdx)
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < (idx4 - idx3 + 1) * (idx2 - idx1 + 1)){

        int ldx = idx1 + tid / (idx4 - idx3 + 1);
        UINT64 hleft = d_langCache[ldx * 2];
        UINT64 lleft = d_langCache[ldx * 2 + 1];

        int rdx = idx3 + tid % (idx4 - idx3 + 1);
        UINT64 hright = d_langCache[rdx * 2];
        UINT64 lright = d_langCache[rdx * 2 + 1];

        UINT64 hCS = hleft | hright;
        UINT64 lCS = lleft | lright;

        if (onTheFly) {

            if (( hCS & hPosBits) == hPosBits && 
                ( lCS & lPosBits) == lPosBits &&
                (~hCS & hNegBits) == hNegBits && 
                (~lCS & lNegBits) == lNegBits) {
                *d_FinalREIdx = tid;
                d_temp_langCache[tid * 2] = hCS;
                d_temp_langCache[tid * 2 + 1] = lCS;
                d_temp_leftIdx[tid] = ldx;
                d_temp_rightIdx[tid] = rdx;
            }

        } else {

            const auto group = warpcore::cg::tiled_partition <1> 
            (warpcore::cg::this_thread_block());
            int H = cHashSet.insert(hCS, group);
            int L = cHashSet.insert(lCS, group);
            H = (H > 0) ? H : -H;
            L = (L > 0) ? L : -L;
            UINT64 HL = H; HL <<= 32; HL |= L;
            bool CS_is_unique = (iHashSet.insert(HL, group) > 0) ? false : true;

            if (CS_is_unique) {
                d_temp_langCache[tid * 2] = hCS;
                d_temp_langCache[tid * 2 + 1] = lCS;
                d_temp_leftIdx[tid] = ldx;
                d_temp_rightIdx[tid] = rdx;
                if (( hCS & hPosBits) == hPosBits && 
                    ( lCS & lPosBits) == lPosBits &&
                    (~hCS & hNegBits) == hNegBits && 
                    (~lCS & lNegBits) == lNegBits) 
                    atomicCAS(d_FinalREIdx, -1, tid);
            } else {
                d_temp_langCache[tid * 2] = UINT64(0) - 1;
                d_temp_langCache[tid * 2 + 1] = UINT64(0) - 1;
                d_temp_leftIdx[tid] = -1;
                d_temp_rightIdx[tid] = -1;
            }

        }

    }

}

// Generating eps+r (For those cost functions where cost(eps+r) < cost(r?))
// For r in indices between idx1 and idx2 in the language cache
template<class hash_set_t>
__global__ void OrEpsilon(
    const int idx1, const 
    int idx2, bool onTheFly, 
    UINT64 *d_langCache, 
    UINT64 *d_temp_langCache, 
    int *d_temp_leftIdx, 
    int *d_temp_rightIdx, 
    hash_set_t cHashSet, 
    hash_set_t iHashSet,
    const UINT64 hPosBits, 
    const UINT64 lPosBits, 
    const UINT64 hNegBits, 
    const UINT64 lNegBits, 
    int *d_FinalREIdx) 
{

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < idx2 - idx1 + 1) {

        UINT64 hCS = d_langCache[(idx1 + tid) * 2];
        UINT64 lCS = d_langCache[(idx1 + tid) * 2 + 1] | 1;

        if (onTheFly) {

            if (( hCS & hPosBits) == hPosBits && 
                ( lCS & lPosBits) == lPosBits &&
                (~hCS & hNegBits) == hNegBits && 
                (~lCS & lNegBits) == lNegBits) {
                *d_FinalREIdx = tid;
                d_temp_langCache[tid * 2] = hCS;
                d_temp_langCache[tid * 2 + 1] = lCS;
                d_temp_leftIdx[tid] = -2;
                d_temp_rightIdx[tid] = idx1 + tid;
            }

        } else {

            const auto group = warpcore::cg::tiled_partition <1> 
            (warpcore::cg::this_thread_block());
            int H = cHashSet.insert(hCS, group);
            int L = cHashSet.insert(lCS, group);
            H = (H > 0) ? H : -H;
            L = (L > 0) ? L : -L;
            UINT64 HL = H; HL <<= 32; HL |= L;
            bool CS_is_unique = (iHashSet.insert(HL, group) > 0) ? false : true;

            if (CS_is_unique) {
                d_temp_langCache[tid * 2] = hCS;
                d_temp_langCache[tid * 2 + 1] = lCS;
                d_temp_leftIdx[tid] = -2;
                d_temp_rightIdx[tid] = idx1 + tid;
                if (( hCS & hPosBits) == hPosBits && 
                    ( lCS & lPosBits) == lPosBits &&
                    (~hCS & hNegBits) == hNegBits && 
                    (~lCS & lNegBits) == lNegBits) 
                    atomicCAS(d_FinalREIdx, -1, tid);
            } else {
                d_temp_langCache[tid * 2] = UINT64(0) - 1;
                d_temp_langCache[tid * 2 + 1] = UINT64(0) - 1;
                d_temp_leftIdx[tid] = -1;
                d_temp_rightIdx[tid] = -1;
            }

        }

    }

}

// Shortlex ordering
struct strComparison {
    bool operator () (const std::string &str1, const std::string &str2) {
        if (str1.length() == str2.length()) return str1 < str2;
        return str1.length() < str2.length();
    }
};

// Adding parentheses if needed
std::string bracket(std::string s) {
    int p = 0;
    for (int i = 0; i < s.length(); i++){
        if (s[i] == '(') p++;
        else if (s[i] == ')') p--;
        else if (s[i] == '+' && p <= 0) return "(" + s + ")";
    }
    return s;
}

// Generating the final RE string recursively
// When all the left and right indices are ready in the host
std::string toString(
    int index, 
    std::map<int, std::pair<int, int>> &indicesMap, 
    const std::set<char> &alphabet, 
    const int *startPoints)
{

    if (index == -2) return "eps"; // Epsilon
    if (index == -1) return "Error";
    if (index < alphabet.size()) {
        std::string s(1, *next(alphabet.begin(), index));
        return s;
    }
    int i = 0;
    while (index >= startPoints[i]){i++;}
    i--;

    if (i % 4 == 0) {
        std::string res = toString(indicesMap[index].first, indicesMap, alphabet, startPoints);
        if (res.length() > 1) return "(" + res + ")?";
        return res + "?";
    }

    if (i % 4 == 1) {
        std::string res = toString(indicesMap[index].first, indicesMap, alphabet, startPoints);
        if (res.length() > 1) return "(" + res + ")*";
        return res + "*";
    }

    if (i % 4 == 2) {
        std::string left  = toString(indicesMap[index].first,  indicesMap, alphabet, startPoints);
        std::string right = toString(indicesMap[index].second, indicesMap, alphabet, startPoints);
        return bracket(left) + bracket(right);
    }

    std::string left  = toString(indicesMap[index].first,  indicesMap, alphabet, startPoints);
    std::string right = toString(indicesMap[index].second, indicesMap, alphabet, startPoints);
    return left + "+" + right;

}

// Bringing the left and right indices of the RE from device to host
// If RE is found, this index is from the temp memory               (temp = true)
// For printing other REs if needed, indices are in the main memory (temp = false)
std::string REtoString (
    bool temp, 
    const int FinalREIdx, 
    const int lastIdx, 
    const std::set<char> &alphabet, 
    const int *startPoints, 
    const int *d_leftIdx,
    const int *d_rightIdx, 
    const int *d_temp_leftIdx, 
    const int *d_temp_rightIdx)
{

    auto *LIdx = new int [1];
    auto *RIdx = new int [1];

    if (temp) {
        checkCuda( cudaMemcpy(LIdx, d_temp_leftIdx  + FinalREIdx, sizeof(int), cudaMemcpyDeviceToHost) );
        checkCuda( cudaMemcpy(RIdx, d_temp_rightIdx + FinalREIdx, sizeof(int), cudaMemcpyDeviceToHost) );
    } else {
        checkCuda( cudaMemcpy(LIdx, d_leftIdx +  FinalREIdx, sizeof(int), cudaMemcpyDeviceToHost) );
        checkCuda( cudaMemcpy(RIdx, d_rightIdx + FinalREIdx, sizeof(int), cudaMemcpyDeviceToHost) );
    }

    auto alphabetSize = static_cast<int> (alphabet.size());

    int *d_resIndices;
    checkCuda( cudaMalloc(&d_resIndices, 600 * sizeof(int)) );

    thrust::device_ptr<int> d_resIndices_ptr(d_resIndices);
    thrust::fill(d_resIndices_ptr, d_resIndices_ptr + 600, -1);

    if (*LIdx >= alphabetSize) generateResIndices<<<1, 1>>>(*LIdx, alphabetSize, d_leftIdx, d_rightIdx, d_resIndices);
    if (*RIdx >= alphabetSize) generateResIndices<<<1, 1>>>(*RIdx, alphabetSize, d_leftIdx, d_rightIdx, d_resIndices);

    int resIndices[600];
    checkCuda( cudaMemcpy(resIndices, d_resIndices, 600 * sizeof(int), cudaMemcpyDeviceToHost) );

    std::map<int, std::pair<int, int>> indicesMap;

    if (temp) indicesMap.insert(std::make_pair(INT_MAX - 1, std::make_pair(*LIdx, *RIdx)));
    else      indicesMap.insert(std::make_pair(FinalREIdx, std::make_pair(*LIdx, *RIdx)));

    int i = 0;
    while (resIndices[i] != -1 && i + 2 < 600) {
        int re = resIndices[i];
        int l = resIndices[i + 1];
        int r = resIndices[i + 2];
        indicesMap.insert( std::make_pair(re,  std::make_pair(l, r)));
        i += 3;
    }

    if (i + 2 >= 600) return "Size of the output is too big";

    cudaFree(d_resIndices);

    if (temp) return toString(INT_MAX - 1, indicesMap, alphabet, startPoints);
    else      return toString(FinalREIdx, indicesMap, alphabet, startPoints);
    
}

// Transfering the unique CSs from temp to main memory
void storeUniqueREs(
    int N, 
    int &lastIdx, 
    const int langCacheCapacity, 
    bool &onTheFly, 
    UINT64 *d_langCache, 
    UINT64 *d_temp_langCache,
    int *d_leftIdx, 
    int *d_rightIdx, 
    int *d_temp_leftIdx, 
    int *d_temp_rightIdx)
{

    thrust::device_ptr<UINT64> new_end_ptr;
    thrust::device_ptr<UINT64> d_langCache_ptr(d_langCache + 2 * lastIdx);
    thrust::device_ptr<UINT64> d_temp_langCache_ptr(d_temp_langCache);
    thrust::device_ptr<int> d_leftIdx_ptr(d_leftIdx + lastIdx);
    thrust::device_ptr<int> d_rightIdx_ptr(d_rightIdx + lastIdx);
    thrust::device_ptr<int> d_temp_leftIdx_ptr(d_temp_leftIdx);
    thrust::device_ptr<int> d_temp_rightIdx_ptr(d_temp_rightIdx);

    new_end_ptr = // end of d_temp_langCache
    thrust::remove(d_temp_langCache_ptr, d_temp_langCache_ptr + 2 * N, UINT64(0) - 1);
    thrust::remove(d_temp_leftIdx_ptr, d_temp_leftIdx_ptr + N, -1);
    thrust::remove(d_temp_rightIdx_ptr, d_temp_rightIdx_ptr + N, -1);

    // It stores all (or a part of) unique CSs until language cahce gets full
    // If language cache gets full, it makes onTheFly mode on
    int numberOfNewUniqueREs = static_cast<int>(new_end_ptr - d_temp_langCache_ptr) / 2;
    if (lastIdx + numberOfNewUniqueREs > langCacheCapacity) {
        N = langCacheCapacity - lastIdx;
        onTheFly = true;
    } else N = numberOfNewUniqueREs;

    thrust::copy_n(d_temp_langCache_ptr, 2 * N, d_langCache_ptr);
    thrust::copy_n(d_temp_leftIdx_ptr, N, d_leftIdx_ptr);
    thrust::copy_n(d_temp_rightIdx_ptr, N, d_rightIdx_ptr);

    lastIdx += N;

}

// Generating the infix of a string
std::set<std::string, strComparison> infixesOf(
    const std::string &word)
{
    std::set<std::string, strComparison> ic;
    for (int len = 0; len <= word.length(); ++len) {
        for (int index = 0; index < word.length() - len + 1 ; ++index) {
            ic.insert(word.substr(index, len));
        }
    }
    return ic;
}

// Generating of the guide table only once for the whole enumeration process
bool generatingGuideTable(
    int &ICsize, 
    int &gtColumns, 
    UINT64 &lPosBits, 
    UINT64 &hPosBits, 
    UINT64 &lNegBits, 
    UINT64 &hNegBits, 
    UINT64 **guideTable,
    const std::vector<std::string> &pos, 
    const std::vector<std::string> &neg)
{
    // Generating infix-closure (ic) of the input strings
    std::set<std::string, strComparison> ic = {""};

    for (const std::string& word : pos) {
        std::set<std::string, strComparison> set1 = infixesOf(word);
        ic.insert(set1.begin(), set1.end());
    }

    for (const std::string& word : neg) {
        std::set<std::string, strComparison> set1 = infixesOf(word);
        ic.insert(set1.begin(), set1.end());
    }

    std::vector<std::vector<UINT64>> gt;

    for(auto& word : ic) {

        size_t wordIndex = distance(ic.begin(), ic.find(word));
        std::vector<UINT64> row;

        for (int i = 1; i < word.length(); ++i) {

            int index1 = 0;
            for (auto& w : ic) {
                if (w == word.substr(0, i)) break;
                index1++;
            }

            int index2 = 0;
            for (auto& w : ic) {
                if (w == word.substr(i)) break;
                index2++;
            }

            if (wordIndex < 63){
                row.push_back((UINT64) 1 << index1);
                row.push_back((UINT64) 1 << index2);
            } else {
                if (index1 < 63) {
                    row.push_back(0);
                    row.push_back((UINT64) 1 << index1);
                } else {
                    row.push_back((UINT64) 1 << (index1 - 63));
                    row.push_back(0);
                }
                if (index2 < 63) {
                    row.push_back(0);
                    row.push_back((UINT64) 1 << index2);
                } else {
                    row.push_back((UINT64) 1 << (index2 - 63));
                    row.push_back(0);
                }
            }

        }

        row.push_back(0);

        if (wordIndex >= 63){
            row.push_back(0);
        }

        gt.push_back(row);

    }

    ICsize = static_cast<int> (ic.size());
    gtColumns = static_cast<int> (gt.back().size());

    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    int constantMemCapacity = props.totalConstMem;

    if (ICsize > 2 * 63) {
        printf("Your input needs %u bits which exceeds 126 bits ", ICsize);
        printf("(current version).\nPlease use less/shorter strings and run the code again.\n");
        return false;
    }

    if (constantMemCapacity < ICsize * gtColumns * sizeof(UINT64)) {
        printf("Some of the input strings are too long for this version.\n");
        printf("Please use shorter strings and run the code again.\n");
        return false;
    }

    *guideTable = new UINT64 [ICsize * gtColumns];

    for (int i = 0; i < ICsize; ++i) {
        for (int j = 0; j < gt.at(i).size(); ++j) {
            (*guideTable)[i * gtColumns + j] = gt.at(i).at(j);
        }
    }

    // Generating lPosBits, hPosBits, lNegBits and hNegBits
    // For checking CSs if they are compatible with Pos and Neg

    for (auto &p : pos){
        size_t wordIndex = distance(ic.begin(), ic.find(p));
        if (wordIndex < 63){
            lPosBits |= ((UINT64) 1 << wordIndex);
        } else {
            hPosBits |= ((UINT64) 1 << (wordIndex - 63));
        }
    }

    for (auto &n : neg){
        size_t wordIndex = distance(ic.begin(), ic.find(n));
        if (wordIndex < 63){
            lNegBits |= ((UINT64) 1 << wordIndex);
        } else {
            hNegBits |= ((UINT64) 1 << (wordIndex - 63));
        }
    }

    return true;
}

// Regular Expression Inference (REI)
std::string REI(
    const unsigned short *costFun, 
    const unsigned short maxCost, 
    int &REcost, 
    UINT64 &allREs,
    int &ICsize, 
    const std::vector<std::string> &pos, 
    const std::vector<std::string> &neg)
{

    // --------------------------
    // Generating the guide table
    // --------------------------

    int gtColumns;
    UINT64 lPosBits{}, hPosBits{}, lNegBits{}, hNegBits{}, *guideTable;
    if (!generatingGuideTable(ICsize, gtColumns, lPosBits, hPosBits, lNegBits, hNegBits, &guideTable, pos, neg)) 
        return "see_the_error";
    checkCuda( cudaMemcpyToSymbol(d_guideTable, guideTable, ICsize * gtColumns * sizeof(UINT64)) );

    // -----------------------------------------
    // Checking empty, epsilon, and the alphabet
    // -----------------------------------------

    // Initialisation of the alphabet
    std::set<char> alphabet;
    for (auto & word : pos) for (auto ch : word) alphabet.insert(ch);
    for (auto & word : neg) for (auto ch : word) alphabet.insert(ch);

    #ifndef MEASUREMENT_MODE
        printf("Cost %-2d | (A) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n", 
                costFun[0], allREs, 0, static_cast<int>(alphabet.size()) + 2);
    #endif

    // Checking empty
    allREs++;
    if (pos.empty()) return "Empty";

    // Checking epsilon
    allREs++;
    if ((pos.size() == 1) && (pos.at(0).empty())) return "eps";

    // Index of the last free position in the LanguagelangCache
    int lastIdx{};

    // Checking the alphabet
    UINT64 idx = 2; // Pointing to the position of the first char of the alphabet (idx 0 is for epsilon)
    auto *langCache = new UINT64 [alphabet.size() * 2];
    auto alphabetSize = static_cast<int> (alphabet.size());
    for (int i = 0; i < alphabetSize; ++i) {

        langCache[i * 2]     = 0;   // hCS
        langCache[i * 2 + 1] = idx; // lCS

        allREs++;

        std::string s(1, *next(alphabet.begin(), i));
        if ((pos.size() == 1) && (pos.at(0) == s)) return s;
        
        idx <<= 1;
        lastIdx++;
    }

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    // cost function
    int c1 = costFun[0]; // cost of a
    int c2 = costFun[1]; // cost of ?
    int c3 = costFun[2]; // cost of *
    int c4 = costFun[3]; // cost of .
    int c5 = costFun[4]; // cost of +

    const int langCacheCapacity      = 200000000;
    const int temp_langCacheCapacity = 100000000;
    
    // 4 for "*", ".", "+" and "?"
    int *startPoints = new int [(maxCost + 2) * 4]();
    startPoints[c1 * 4 + 3] = lastIdx;
    startPoints[(c1 + 1) * 4] = lastIdx;

    int *d_FinalREIdx;
    auto *FinalREIdx = new int [1]; *FinalREIdx = -1;
    checkCuda( cudaMalloc(&d_FinalREIdx, sizeof(int)) );
    checkCuda( cudaMemcpy(d_FinalREIdx, FinalREIdx, sizeof(int), cudaMemcpyHostToDevice) );

    UINT64 *d_langCache, *d_temp_langCache;
    int *d_leftIdx, *d_rightIdx, *d_temp_leftIdx, *d_temp_rightIdx;
    checkCuda( cudaMalloc(&d_langCache, 2 * langCacheCapacity * sizeof(UINT64)) );
    checkCuda( cudaMalloc(&d_temp_langCache, 2 * temp_langCacheCapacity * sizeof(UINT64)) );
    checkCuda( cudaMalloc(&d_leftIdx, langCacheCapacity * sizeof(int)) );
    checkCuda( cudaMalloc(&d_rightIdx, langCacheCapacity * sizeof(int)) );
    checkCuda( cudaMalloc(&d_temp_leftIdx, temp_langCacheCapacity * sizeof(int)) );
    checkCuda( cudaMalloc(&d_temp_rightIdx, temp_langCacheCapacity * sizeof(int)) );

    using hash_set_t = warpcore::HashSet<
    UINT64,         // key type
    UINT64(0) - 1,  // empty key
    UINT64(0) - 2,  // tombstone key
    warpcore::probing_schemes::QuadraticProbing<warpcore::hashers::MurmurHash <UINT64>>>;
    
    hash_set_t cHashSet(2 * langCacheCapacity);
    hash_set_t iHashSet(2 * langCacheCapacity);

    checkCuda( cudaMemcpy(d_langCache, langCache, 2 * alphabet.size() * sizeof(UINT64), cudaMemcpyHostToDevice) );
    hashSetsInitialisation<hash_set_t><<<1, 1>>>(alphabetSize ,cHashSet, iHashSet, d_langCache);

    // ---------------------------
    // Enumeration of the next REs
    // ---------------------------

    bool onTheFly = false, lastRound = false;
    int shortageCost = -1;

    for (REcost = c1 + 1; REcost <= maxCost; ++REcost) {


        // Once it uses a previous cost that is not fully stored, it should continue as the last round
	    if (onTheFly) {
            int dif = REcost - shortageCost;
            if (dif == c2 || dif == c3 || dif == c1 + c4 || dif == c1 + c5) lastRound = true;
        }

        // Question mark
        if (REcost - c2 >= c1 && c1 + c5 >= c2) {

            int qIdx1 = startPoints[(REcost - c2) * 4 + 2];
            int qIdx2 = startPoints[(REcost - c2 + 1) * 4] - 1;
            int qN = qIdx2 - qIdx1 + 1;

            if (qN){
                int x = qIdx1, y;
                do {
                    y = x + std::min(temp_langCacheCapacity - 1, qIdx2 - x);
                    qN = (y - x + 1);
                    #ifndef MEASUREMENT_MODE
                        printf("Cost %-2d | (Q) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n", 
                                REcost, allREs, lastIdx, qN);
                    #endif
                    int qBlc = (qN + 1023) / 1024;
                    QuestionMark<hash_set_t><<<qBlc, 1024>>>(x, y, onTheFly, d_langCache, d_temp_langCache, 
                                                             d_temp_leftIdx, d_temp_rightIdx, cHashSet, iHashSet, 
                                                             hPosBits, lPosBits, hNegBits, lNegBits, d_FinalREIdx);
                    checkCuda( cudaPeekAtLastError() );
                    checkCuda( cudaMemcpy(FinalREIdx, d_FinalREIdx, sizeof(int), cudaMemcpyDeviceToHost) );
                    allREs += qN;
                    if (*FinalREIdx != -1) {startPoints[REcost * 4 + 1] = INT_MAX; goto exitEnumeration;}
                    if (!onTheFly) storeUniqueREs(qN, lastIdx, langCacheCapacity, onTheFly, d_langCache, d_temp_langCache, 
                                                  d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < qIdx2);
            }

        }
        startPoints[REcost * 4 + 1] = lastIdx;

        // Star
        if (REcost - c3 >= c1) {

            int sIdx1 = startPoints[(REcost - c3) * 4 + 2];
            int sIdx2 = startPoints[(REcost - c3 + 1) * 4] - 1;
            int sN = sIdx2 - sIdx1 + 1;

            if (sN){
                int x = sIdx1, y;
                do {
                    y = x + std::min(temp_langCacheCapacity - 1, sIdx2 - x);
                    sN = (y - x + 1);
                    #ifndef MEASUREMENT_MODE
                        printf("Cost %-2d | (S) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n", 
                                REcost, allREs, lastIdx, sN);
                    #endif
                    int sBlc = (sN + 1023) / 1024;
                    Star<hash_set_t><<<sBlc, 1024>>>(x, y, onTheFly, d_langCache, d_temp_langCache, 
                                                     d_temp_leftIdx, d_temp_rightIdx, cHashSet, iHashSet, 
                                                     alphabetSize, ICsize, gtColumns, 
                                                     hPosBits, lPosBits, hNegBits, lNegBits, d_FinalREIdx);
                    checkCuda( cudaPeekAtLastError() );
                    checkCuda( cudaMemcpy(FinalREIdx, d_FinalREIdx, sizeof(int), cudaMemcpyDeviceToHost) );
                    allREs += sN;
                    if (*FinalREIdx != -1) {startPoints[REcost * 4 + 2] = INT_MAX; goto exitEnumeration;}
                    if (!onTheFly) storeUniqueREs(sN, lastIdx, langCacheCapacity, onTheFly, d_langCache, d_temp_langCache, 
                                                  d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < sIdx2);
            }

        }
        startPoints[REcost * 4 + 2] = lastIdx;

        // Concat
        for (int i = c1; 2 * i <= REcost - c4; ++i) {

            int cIdx1 = startPoints[i * 4];
            int cIdx2 = startPoints[(i + 1) * 4] - 1;
            int cIdx3 = startPoints[(REcost - i - c4) * 4];
            int cIdx4 = startPoints[(REcost - i - c4 + 1) * 4] - 1;
            int cN = (cIdx4 - cIdx3 + 1) * (cIdx2 - cIdx1 + 1);
            
            if (cN) {
                int x = cIdx3, y;
                do {
                    y = x + std::min(temp_langCacheCapacity / (2 * (cIdx2 - cIdx1 + 1)) - 1, cIdx4 - x); // 2 is for concat only (lr and rl)
                    cN = (y - x + 1) * (cIdx2 - cIdx1 + 1);
                    #ifndef MEASUREMENT_MODE
                        printf("Cost %-2d | (C) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n", 
                                REcost, allREs, lastIdx, 2 * cN);
                    #endif
                    int cBlc = (cN + 1023) / 1024;
                    Concat<hash_set_t><<<cBlc, 1024>>>(cIdx1, cIdx2, x, y, onTheFly, d_langCache, d_temp_langCache, 
                                                       d_temp_leftIdx, d_temp_rightIdx, cHashSet, iHashSet, alphabetSize, 
                                                       ICsize, gtColumns, hPosBits, lPosBits, hNegBits, lNegBits, d_FinalREIdx);
                    checkCuda( cudaPeekAtLastError() );
                    checkCuda( cudaMemcpy(FinalREIdx, d_FinalREIdx, sizeof(int), cudaMemcpyDeviceToHost) );
                    allREs += 2 * cN;
                    if (*FinalREIdx != -1) {startPoints[REcost * 4 + 3] = INT_MAX; goto exitEnumeration;}
                    if (!onTheFly) storeUniqueREs(2 * cN, lastIdx, langCacheCapacity, onTheFly, d_langCache, d_temp_langCache, 
                                                  d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < cIdx4);
            }

        }
        startPoints[REcost * 4 + 3] = lastIdx;

        // Or
        if (c1 + c5 < c2 && 2 * c1 <= REcost - c5) { // Where cost(eps+x) < cost(x?)

            int oIdx1 = startPoints[(REcost - c1 - c5) * 4];
            int oIdx2 = startPoints[(REcost - c1 - c5 + 1) * 4] - 1;
            int oN = oIdx2 - oIdx1 + 1;

            if (oN){
                int x = oIdx1, y;
                do {
                    y = x + std::min(temp_langCacheCapacity - 1, oIdx2 - x);
                    oN = (y - x + 1);
                    #ifndef MEASUREMENT_MODE
                        printf("Cost %-2d | (O) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n", 
                                REcost, allREs, lastIdx, oN);
                    #endif
                    int oBlc = (oN + 1023) / 1024;
                    OrEpsilon<hash_set_t><<<oBlc, 1024>>>(x, y, onTheFly, d_langCache, d_temp_langCache, 
                                                          d_temp_leftIdx, d_temp_rightIdx, cHashSet, iHashSet, 
                                                          hPosBits, lPosBits, hNegBits, lNegBits, d_FinalREIdx);
                    checkCuda( cudaPeekAtLastError() );
                    checkCuda( cudaMemcpy(FinalREIdx, d_FinalREIdx, sizeof(int), cudaMemcpyDeviceToHost) );
                    allREs += oN;
                    if (*FinalREIdx != -1) {startPoints[(REcost + 1) * 4] = INT_MAX; goto exitEnumeration;}
                    if (!onTheFly) storeUniqueREs(oN, lastIdx, langCacheCapacity, onTheFly, d_langCache, d_temp_langCache, 
                                                  d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < oIdx2);
            }
            
        }
        for (int i = c1; 2 * i <= REcost - c5; ++i) {

            int oIdx1 = startPoints[i * 4];
            int oIdx2 = startPoints[(i + 1) * 4] - 1;
            int oIdx3 = startPoints[(REcost - i - c5) * 4];
            int oIdx4 = startPoints[(REcost - i - c5 + 1) * 4] - 1;
            int oN = (oIdx4 - oIdx3 + 1) * (oIdx2 - oIdx1 + 1);

            if (oN) {
                int x = oIdx3, y;
                do {
                    y = x + std::min(temp_langCacheCapacity / (oIdx2 - oIdx1 + 1) - 1, oIdx4 - x);
                    oN = (y - x + 1) * (oIdx2 - oIdx1 + 1);
                    #ifndef MEASUREMENT_MODE
                        printf("Cost %-2d | (O) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n", 
                                REcost, allREs, lastIdx, oN);
                    #endif
                    int oBlc = (oN + 1023) / 1024;
                    Or<hash_set_t><<<oBlc, 1024>>>(oIdx1, oIdx2, x, y, onTheFly, d_langCache, d_temp_langCache, 
                                                   d_temp_leftIdx, d_temp_rightIdx, cHashSet, iHashSet, hPosBits, 
                                                   lPosBits, hNegBits, lNegBits, d_FinalREIdx);
                    checkCuda( cudaPeekAtLastError() );
                    checkCuda( cudaMemcpy(FinalREIdx, d_FinalREIdx, sizeof(int), cudaMemcpyDeviceToHost) );
                    allREs += oN;
                    if (*FinalREIdx != -1) {startPoints[(REcost + 1) * 4] = INT_MAX; goto exitEnumeration;}
                    if (!onTheFly) storeUniqueREs(oN, lastIdx, langCacheCapacity, onTheFly, d_langCache, d_temp_langCache, 
                                                  d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);
                    x = y + 1;
                } while (y < oIdx4);
            }

        }
        startPoints[(REcost + 1) * 4] = lastIdx;

        if (lastRound) break;
        if (onTheFly && shortageCost == -1) shortageCost = REcost;

    }

    if (REcost == maxCost + 1) REcost--;

    exitEnumeration:

    std::string output = "not_found";
    bool isREFromTempLangCache = true;
    if (*FinalREIdx != -1) output = REtoString(isREFromTempLangCache, *FinalREIdx, lastIdx, alphabet, startPoints, 
        d_leftIdx, d_rightIdx, d_temp_leftIdx, d_temp_rightIdx);

    // cleanup
    cudaFree(d_langCache);
    cudaFree(d_FinalREIdx);
    cudaFree(d_temp_langCache);
    cudaFree(d_leftIdx);
    cudaFree(d_rightIdx);
    cudaFree(d_temp_leftIdx);
    cudaFree(d_temp_rightIdx);

    return output;
}

// Reading the input file
bool readFile(
    const std::string& fileName, 
    std::vector<std::string> &pos, 
    std::vector<std::string> &neg)
{

    std::string line;
    std::ifstream textFile (fileName);

    if (textFile.is_open()) {

        while (line != "++") {
            getline (textFile, line);
            if (textFile.eof()) {
                printf("Unable to find \"++\" for positive words");
                printf("\nPlease check the input file.\n");
                return false;
            }
        }

        getline (textFile, line);

        while (line != "--") {
            std::string word = "";
            for (auto c : line) if (c != ' ' && c != '"') word += c;
            pos.push_back(word);
            getline (textFile, line);
            if (line != "--" && textFile.eof()) {
                printf("Unable to find \"--\" for negative words");
                printf("\nPlease check the input file.\n");
                return false;
            }
        }

        while (getline (textFile, line)) {
            std::string word = "";
            for (auto c : line) if (c != ' ' && c != '"') word += c;
            for (auto &p : pos) {
                if (word == p) {
                    printf("\"%s\" is in both Pos and Neg examples", word.c_str());
                    printf("\nPlease check the input file, and remove one of those.\n");
                    return false;
                }
            }
            neg.push_back(word);
        }

        textFile.close();

        return true;

    }

    printf("Unable to open the file");
    return false;

}

int main (int argc, char *argv[]) {

    // -----------------
    // Reading the input
    // -----------------

    if (argc != 8) {
        printf("Arguments should be in the form of\n");
        printf("-----------------------------------------------------------------\n");
        printf("%s <input_file_address> <c1> <c2> <c3> <c4> <c5> <maxCost>\n", argv[0]);
        printf("-----------------------------------------------------------------\n");
        printf("\nFor example\n");
        printf("-----------------------------------------------------------------\n");
        printf("%s ./input.txt 1 1 1 1 1 500\n", argv[0]);
        printf("-----------------------------------------------------------------\n");
        return 0;
    }

    bool argError = false;
    for (int i = 2; i < argc; ++i) {
        if (std::atoi(argv[i]) <= 0 || std::atoi(argv[i]) > SHRT_MAX) {
            printf("Argument number %d, \"%s\", should be a positive short integer.\n", i, argv[i]);
            argError = true;
        }
    }
    if (argError) return 0;

    std::string fileName = argv[1];
    std::vector<std::string> pos, neg;
    if (!readFile(fileName, pos, neg)) return 0;
    unsigned short costFun[5];
    for (int i = 0; i < 5; i++) 
        costFun[i] = std::atoi(argv[i + 2]);
    unsigned short maxCost = std::atoi(argv[7]);

    // ----------------------------------
    // Regular Expression Inference (REI)
    // ----------------------------------

    #ifdef MEASUREMENT_MODE
        auto start = std::chrono::high_resolution_clock::now();
    #endif

    UINT64 allREs{}; int ICsize, REcost = costFun[0];
    std::string output = REI(costFun, maxCost, REcost, allREs, ICsize, pos, neg);
    if (output == "see_the_error") return 0;

    #ifdef MEASUREMENT_MODE
        auto stop = std::chrono::high_resolution_clock::now();
    #endif

    // -------------------
    // Printing the output
    // -------------------

    printf("\nPositive: "); for (const auto& p : pos) printf("\"%s\" ", p.c_str());
    printf("\nNegative: "); for (const auto& n : neg) printf("\"%s\" ", n.c_str());
    printf("\nCost Function: \"a\"=%u, \"?\"=%u, \"*\"=%u, \".\"=%u, \"+\"=%u", 
           costFun[0], costFun[1], costFun[2], costFun[3], costFun[4]);
    printf("\nSize of IC: %u", ICsize);
    #ifdef MEASUREMENT_MODE
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        printf("\nCost of Final RE: %d", REcost);
        printf("\nNumber of All REs: %lu", allREs);
        printf("\nRunning Time: %f s", (double) duration * 0.000001);
    #endif
    printf("\n\nRE: \"%s\"\n", output.c_str());

    return 0;
    
}
