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
cudaError_t checkCuda(cudaError_t result) {
#ifndef MEASUREMENT_MODE
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__constant__ UINT64 dev_guideTable[64 * 128];

__global__ void generateResultIndices(const int index, const int nonatomicIndex, const int *dev_left_indices, const int *dev_right_indices, int *dev_result) {
    int resultIdx = 0;
    while (dev_result[resultIdx] != -1) resultIdx++;
    int queue[600];
    queue[0] = index;
    int head = 0;
    int tail = 1;
    while (head < tail) {
        int re = queue[head];
        int l = dev_left_indices[re];
        int r = dev_right_indices[re];
        dev_result[resultIdx++] = re;
        dev_result[resultIdx++] = l;
        dev_result[resultIdx++] = r;
        if (l >= nonatomicIndex) queue[tail++] = l;
        if (r >= nonatomicIndex) queue[tail++] = r;
        head++;
    }
}

template<class hash_set_t>
__global__ void InitialiseHashSetsByAlphabet(hash_set_t cHashSet, hash_set_t iHashSet, UINT64 *dev_cache) {
    const auto group = warpcore::cg::tiled_partition <1> (warpcore::cg::this_thread_block());
    int H = cHashSet.insert(dev_cache[2 * threadIdx.x], group);
    int L = cHashSet.insert(dev_cache[2 * threadIdx.x + 1], group);
    H = (H > 0) ? H : -H;
    L = (L > 0) ? L : -L;
    UINT64 HL = H; HL <<= 32; HL |= L;
    iHashSet.insert(HL, group);
}

template<class hash_set_t>
__global__ void QuestionMark(const int idx1, const int idx2, bool onTheFly, hash_set_t cHashSet, hash_set_t iHashSet, UINT64 *dev_cache, UINT64 *dev_temp_cache, int *dev_temp_left_indices, 
                             int *dev_temp_right_indices, const UINT64 lPosBits, const UINT64 hPosBits, const UINT64 lNegBits, const UINT64 hNegBits, int *dev_result) {

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < idx2 - idx1 + 1) {

        UINT64 hREc = dev_cache[(idx1 + tid) * 2];
        UINT64 lREc = dev_cache[(idx1 + tid) * 2 + 1] | 1;

        if (onTheFly) {

            if ((hREc & hPosBits) == hPosBits && (lREc & lPosBits) == lPosBits &&
                    (~hREc & hNegBits) == hNegBits && (~lREc & lNegBits) == lNegBits) {
                *dev_result = tid;
                dev_temp_cache[tid * 2] = hREc;
                dev_temp_cache[tid * 2 + 1] = lREc;
                dev_temp_left_indices[tid] = idx1 + tid;
                dev_temp_right_indices[tid] = 0;
            }

        } else {

            const auto group = warpcore::cg::tiled_partition <1> (warpcore::cg::this_thread_block());
            int H = cHashSet.insert(hREc, group);
            int L = cHashSet.insert(lREc, group);
            H = (H > 0) ? H : -H;
            L = (L > 0) ? L : -L;
            UINT64 HL = H; HL <<= 32; HL |= L;
            bool REc_is_unique = (iHashSet.insert(HL, group) > 0) ? false : true;

            if (REc_is_unique) {
                dev_temp_cache[tid * 2] = hREc;
                dev_temp_cache[tid * 2 + 1] = lREc;
                dev_temp_left_indices[tid] = idx1 + tid;
                dev_temp_right_indices[tid] = 0; // just to avoid removing
                if ((hREc & hPosBits) == hPosBits && (lREc & lPosBits) == lPosBits &&
                    (~hREc & hNegBits) == hNegBits && (~lREc & lNegBits) == lNegBits) atomicCAS(dev_result, -1, tid);
            } else {
                dev_temp_cache[tid * 2] = UINT64(0) - 1;
                dev_temp_cache[tid * 2 + 1] = UINT64(0) - 1;
                dev_temp_left_indices[tid] = -1;
                dev_temp_right_indices[tid] = -1;
            }

        }

    }

}

template<class hash_set_t>
__global__ void Star(const int idx1, const int idx2, bool onTheFly, hash_set_t cHashSet, hash_set_t iHashSet, UINT64 *dev_cache, UINT64 *dev_temp_cache, int *dev_temp_left_indices, int *dev_temp_right_indices, 
                     const int nonatomicIndex, const int GTrows, const int GTcolumns, const UINT64 lPosBits, const UINT64 hPosBits, const UINT64 lNegBits, const UINT64 hNegBits, int *dev_result) {

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < idx2 - idx1 + 1){

        UINT64 hREc = dev_cache[(idx1 + tid) * 2];
        UINT64 lREc = dev_cache[(idx1 + tid) * 2 + 1] | 1;

        int ix = nonatomicIndex;
        UINT64 c = 1 << ix;

        while (ix < 63 && ix < GTrows){
            if (!(lREc & c)){
                int index = ix * GTcolumns;
                while (dev_guideTable[index]){
                    if ((dev_guideTable[index] & lREc) && (dev_guideTable[index + 1] & lREc))
                    {lREc |= c; break;}
                    index += 2;
                }
            }
            c <<= 1; ix++;
        }

        c = 1;

        while (ix < GTrows){
            if (!(hREc & c)){
                int index = ix * GTcolumns;
                while (dev_guideTable[index] || dev_guideTable[index + 1]){
                    if (((dev_guideTable[index] & hREc) || (dev_guideTable[index + 1] & lREc))
                    && ((dev_guideTable[index + 2] & hREc) || (dev_guideTable[index + 3] & lREc)))
                    {hREc |= c; break;}
                    index += 4;
                }
            }
            c <<= 1; ix++;
        }

        if (onTheFly) {

            if ((hREc & hPosBits) == hPosBits && (lREc & lPosBits) == lPosBits &&
                    (~hREc & hNegBits) == hNegBits && (~lREc & lNegBits) == lNegBits) {
                *dev_result = tid;
                dev_temp_cache[tid * 2] = hREc;
                dev_temp_cache[tid * 2 + 1] = lREc;
                dev_temp_left_indices[tid] = idx1 + tid;
                dev_temp_right_indices[tid] = 0;
            }

        } else {

            const auto group = warpcore::cg::tiled_partition <1> (warpcore::cg::this_thread_block());
            int H = cHashSet.insert(hREc, group);
            int L = cHashSet.insert(lREc, group);
            H = (H > 0) ? H : -H;
            L = (L > 0) ? L : -L;
            UINT64 HL = H; HL <<= 32; HL |= L;
            bool REc_is_unique = (iHashSet.insert(HL, group) > 0) ? false : true;

            if (REc_is_unique) {
                dev_temp_cache[tid * 2] = hREc;
                dev_temp_cache[tid * 2 + 1] = lREc;
                dev_temp_left_indices[tid] = idx1 + tid;
                dev_temp_right_indices[tid] = 0; // just to avoid removing
                if ((hREc & hPosBits) == hPosBits && (lREc & lPosBits) == lPosBits &&
                    (~hREc & hNegBits) == hNegBits && (~lREc & lNegBits) == lNegBits) atomicCAS(dev_result, -1, tid);
            } else {
                dev_temp_cache[tid * 2] = UINT64(0) - 1;
                dev_temp_cache[tid * 2 + 1] = UINT64(0) - 1;
                dev_temp_left_indices[tid] = -1;
                dev_temp_right_indices[tid] = -1;
            }

        }

    }

}

template<class hash_set_t>
__global__ void Concat(const int idx1, const int idx2, const int idx3, const int idx4, bool onTheFly, hash_set_t cHashSet, hash_set_t iHashSet, UINT64 *dev_cache, UINT64 *dev_temp_cache, 
                       int *dev_temp_left_indices, int *dev_temp_right_indices, const int nonatomicIndex, const int GTrows, const int GTcolumns, const UINT64 lPosBits, 
                       const UINT64 hPosBits, const UINT64 lNegBits, const UINT64 hNegBits, int *dev_result){

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < (idx4 - idx3 + 1) * (idx2 - idx1 + 1)){

        int ldx = idx1 + tid / (idx4 - idx3 + 1);
        UINT64 hleft = dev_cache[ldx * 2];
        UINT64 lleft = dev_cache[ldx * 2 + 1];

        int rdx = idx3 + tid % (idx4 - idx3 + 1);
        UINT64 hright = dev_cache[rdx * 2];
        UINT64 lright = dev_cache[rdx * 2 + 1];

        UINT64 hREc1{};
        UINT64 lREc1{};

        if (lleft & 1) {hREc1 |= hright; lREc1 |= lright;}
        if (lright & 1) {hREc1 |= hleft; lREc1 |= lleft;}
        
        UINT64 hREc2 = hREc1;
        UINT64 lREc2 = lREc1;

        int ix = nonatomicIndex;
        UINT64 c = 1 << ix;

        while (ix < 63 && ix < GTrows){

            if (!(lREc1 & c)){
                int index = ix * GTcolumns;
                while (dev_guideTable[index]){
                    if ((dev_guideTable[index] & lleft) && (dev_guideTable[index + 1] & lright))
                    {lREc1 |= c; break;}
                    index += 2;
                }
            }

            if (!(lREc2 & c)){
                int index = ix * GTcolumns;
                while (dev_guideTable[index]){
                    if ((dev_guideTable[index] & lright) && (dev_guideTable[index + 1] & lleft))
                    {lREc2 |= c; break;}
                    index += 2;
                }
            }

            c <<= 1; ix++;

        }

        c = 1;
        
        while (ix < GTrows){

            if (!(hREc1 & c)){
                int index = ix * GTcolumns;
                while (dev_guideTable[index] || dev_guideTable[index + 1]){
                    if (((dev_guideTable[index] & hleft) || (dev_guideTable[index + 1] & lleft))
                    && ((dev_guideTable[index + 2] & hright) || (dev_guideTable[index + 3] & lright)))
                    {hREc1 |= c; break;}
                    index += 4;
                }
            }

            if (!(hREc2 & c)){
                int index = ix * GTcolumns;
                while (dev_guideTable[index] || dev_guideTable[index + 1]){
                    if (((dev_guideTable[index] & hright) || (dev_guideTable[index + 1] & lright))
                    && ((dev_guideTable[index + 2] & hleft) || (dev_guideTable[index + 3] & lleft)))
                    {hREc2 |= c; break;}
                    index += 4;
                }
            }

            c <<= 1; ix++;

        }

        if (onTheFly) {

            if ((hREc1 & hPosBits) == hPosBits && (lREc1 & lPosBits) == lPosBits &&
                    (~hREc1 & hNegBits) == hNegBits && (~lREc1 & lNegBits) == lNegBits) {
                *dev_result = tid * 2;
                dev_temp_cache[tid * 4] = hREc1;
                dev_temp_cache[tid * 4 + 1] = lREc1;
                dev_temp_left_indices[tid * 2] = ldx;
                dev_temp_right_indices[tid * 2] = rdx;
            } else if ((hREc2 & hPosBits) == hPosBits && (lREc2 & lPosBits) == lPosBits &&
                    (~hREc2 & hNegBits) == hNegBits && (~lREc2 & lNegBits) == lNegBits) {
                *dev_result = tid * 2 + 1;
                dev_temp_cache[tid * 4 + 2] = hREc2;
                dev_temp_cache[tid * 4 + 3] = lREc2;
                dev_temp_left_indices[tid * 2 + 1] = rdx;
                dev_temp_right_indices[tid * 2 + 1] = ldx;
            }

        } else {

            const auto group = warpcore::cg::tiled_partition <1> (warpcore::cg::this_thread_block());
            int H, L; UINT64 HL;

            H = cHashSet.insert(hREc1, group);
            L = cHashSet.insert(lREc1, group);
            H = (H > 0) ? H : -H;
            L = (L > 0) ? L : -L;
            HL = H; HL <<= 32; HL |= L;
            bool REc1_is_unique = (iHashSet.insert(HL, group) > 0) ? false : true;

            H = cHashSet.insert(hREc2, group);
            L = cHashSet.insert(lREc2, group);
            H = (H > 0) ? H : -H;
            L = (L > 0) ? L : -L;
            HL = H; HL <<= 32; HL |= L;
            bool REc2_is_unique = (iHashSet.insert(HL, group) > 0) ? false : true;

            if (REc1_is_unique) {
                dev_temp_cache[tid * 4] = hREc1;
                dev_temp_cache[tid * 4 + 1] = lREc1;
                dev_temp_left_indices[tid * 2] = ldx;
                dev_temp_right_indices[tid * 2] = rdx;
                if ((hREc1 & hPosBits) == hPosBits && (lREc1 & lPosBits) == lPosBits &&
                    (~hREc1 & hNegBits) == hNegBits && (~lREc1 & lNegBits) == lNegBits) atomicCAS(dev_result, -1, tid * 2);
            } else {
                dev_temp_cache[tid * 4] = UINT64(0) - 1;
                dev_temp_cache[tid * 4 + 1] = UINT64(0) - 1;
                dev_temp_left_indices[tid * 2] = -1;
                dev_temp_right_indices[tid * 2] = -1;
            }

            if (REc2_is_unique) {
                dev_temp_cache[tid * 4 + 2] = hREc2;
                dev_temp_cache[tid * 4 + 3] = lREc2;
                dev_temp_left_indices[tid * 2 + 1] = rdx;
                dev_temp_right_indices[tid * 2 + 1] = ldx;
                if ((hREc2 & hPosBits) == hPosBits && (lREc2 & lPosBits) == lPosBits &&
                    (~hREc2 & hNegBits) == hNegBits && (~lREc2 & lNegBits) == lNegBits) atomicCAS(dev_result, -1, tid * 2 + 1);
            } else {
                dev_temp_cache[tid * 4 + 2] = UINT64(0) - 1;
                dev_temp_cache[tid * 4 + 3] = UINT64(0) - 1;
                dev_temp_left_indices[tid * 2 + 1] = -1;
                dev_temp_right_indices[tid * 2 + 1] = -1;
            }

        }

    }

}

template<class hash_set_t>
__global__ void Or(const int idx1, const int idx2, const int idx3, const int idx4, bool onTheFly, hash_set_t cHashSet, hash_set_t iHashSet, UINT64 *dev_cache, UINT64 *dev_temp_cache,
                   int *dev_temp_left_indices, int *dev_temp_right_indices, const UINT64 lPosBits, const UINT64 hPosBits, const UINT64 lNegBits, const UINT64 hNegBits, int *dev_result) {

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < (idx4 - idx3 + 1) * (idx2 - idx1 + 1)){

        int ldx = idx1 + tid / (idx4 - idx3 + 1);
        UINT64 hleft = dev_cache[ldx * 2];
        UINT64 lleft = dev_cache[ldx * 2 + 1];

        int rdx = idx3 + tid % (idx4 - idx3 + 1);
        UINT64 hright = dev_cache[rdx * 2];
        UINT64 lright = dev_cache[rdx * 2 + 1];

        UINT64 hREc = hleft | hright;
        UINT64 lREc = lleft | lright;

        if (onTheFly) {

            if ((hREc & hPosBits) == hPosBits && (lREc & lPosBits) == lPosBits &&
                    (~hREc & hNegBits) == hNegBits && (~lREc & lNegBits) == lNegBits) {
                *dev_result = tid;
                dev_temp_cache[tid * 2] = hREc;
                dev_temp_cache[tid * 2 + 1] = lREc;
                dev_temp_left_indices[tid] = ldx;
                dev_temp_right_indices[tid] = rdx;
            }

        } else {

            const auto group = warpcore::cg::tiled_partition <1> (warpcore::cg::this_thread_block());
            int H = cHashSet.insert(hREc, group);
            int L = cHashSet.insert(lREc, group);
            H = (H > 0) ? H : -H;
            L = (L > 0) ? L : -L;
            UINT64 HL = H; HL <<= 32; HL |= L;
            bool REc_is_unique = (iHashSet.insert(HL, group) > 0) ? false : true;

            if (REc_is_unique) {
                dev_temp_cache[tid * 2] = hREc;
                dev_temp_cache[tid * 2 + 1] = lREc;
                dev_temp_left_indices[tid] = ldx;
                dev_temp_right_indices[tid] = rdx;
                if ((hREc & hPosBits) == hPosBits && (lREc & lPosBits) == lPosBits &&
                    (~hREc & hNegBits) == hNegBits && (~lREc & lNegBits) == lNegBits) atomicCAS(dev_result, -1, tid);
            } else {
                dev_temp_cache[tid * 2] = UINT64(0) - 1;
                dev_temp_cache[tid * 2 + 1] = UINT64(0) - 1;
                dev_temp_left_indices[tid] = -1;
                dev_temp_right_indices[tid] = -1;
            }

        }

    }

}

template<class hash_set_t>
__global__ void OrEpsilon(const int idx1, const int idx2, bool onTheFly, hash_set_t cHashSet, hash_set_t iHashSet, UINT64 *dev_cache, UINT64 *dev_temp_cache, int *dev_temp_left_indices, 
                             int *dev_temp_right_indices, const UINT64 lPosBits, const UINT64 hPosBits, const UINT64 lNegBits, const UINT64 hNegBits, int *dev_result) {

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < idx2 - idx1 + 1) {

        UINT64 hREc = dev_cache[(idx1 + tid) * 2];
        UINT64 lREc = dev_cache[(idx1 + tid) * 2 + 1] | 1;

        if (onTheFly) {

            if ((hREc & hPosBits) == hPosBits && (lREc & lPosBits) == lPosBits &&
                    (~hREc & hNegBits) == hNegBits && (~lREc & lNegBits) == lNegBits) {
                *dev_result = tid;
                dev_temp_cache[tid * 2] = hREc;
                dev_temp_cache[tid * 2 + 1] = lREc;
                dev_temp_left_indices[tid] = -2;
                dev_temp_right_indices[tid] = idx1 + tid;
            }

        } else {

            const auto group = warpcore::cg::tiled_partition <1> (warpcore::cg::this_thread_block());
            int H = cHashSet.insert(hREc, group);
            int L = cHashSet.insert(lREc, group);
            H = (H > 0) ? H : -H;
            L = (L > 0) ? L : -L;
            UINT64 HL = H; HL <<= 32; HL |= L;
            bool REc_is_unique = (iHashSet.insert(HL, group) > 0) ? false : true;

            if (REc_is_unique) {
                dev_temp_cache[tid * 2] = hREc;
                dev_temp_cache[tid * 2 + 1] = lREc;
                dev_temp_left_indices[tid] = -2;
                dev_temp_right_indices[tid] = idx1 + tid;
                if ((hREc & hPosBits) == hPosBits && (lREc & lPosBits) == lPosBits &&
                    (~hREc & hNegBits) == hNegBits && (~lREc & lNegBits) == lNegBits) atomicCAS(dev_result, -1, tid);
            } else {
                dev_temp_cache[tid * 2] = UINT64(0) - 1;
                dev_temp_cache[tid * 2 + 1] = UINT64(0) - 1;
                dev_temp_left_indices[tid] = -1;
                dev_temp_right_indices[tid] = -1;
            }

        }

    }

}

struct strLenComparison {
    bool operator () (const std::string &str1, const std::string &str2) {
        if (str1.length() == str2.length()) return str1 < str2;
        return str1.length() < str2.length();
    }
};

std::string bracket(std::string s) {
    int p = 0;
    for (int i = 0; i < s.length(); i++){
        if (s[i] == '(') p++;
        else if (s[i] == ')') p--;
        else if (s[i] == '+' && p <= 0) return "(" + s + ")";
    }
    return s;
}

std::string toString(int index, const std::set<char> &alphabet, std::map<int, std::pair<int, int>> &indicesMap, const int *startPoints) {

    if (index == -2) return "eps"; // Epsilon
    if (index == -1) return "Error";
    if (index < alphabet.size()) {std::string s(1, *next(alphabet.begin(), index)); return s;}
    int i = 0;
    while (index >= startPoints[i]){i++;}
    i--;

    if (i % 4 == 0) {
        std::string res = toString(indicesMap[index].first, alphabet, indicesMap, startPoints);
        if (res.length() > 1) return "(" + res + ")?";
        return res + "?";
    }

    if (i % 4 == 1) {
        std::string res = toString(indicesMap[index].first, alphabet, indicesMap, startPoints);
        if (res.length() > 1) return "(" + res + ")*";
        return res + "*";
    }

    if (i % 4 == 2) {
        std::string left = toString(indicesMap[index].first, alphabet, indicesMap, startPoints);
        std::string right = toString(indicesMap[index].second, alphabet, indicesMap, startPoints);
        return bracket(left) + bracket(right);
    }

    std::string left = toString(indicesMap[index].first, alphabet, indicesMap, startPoints);
    std::string right = toString(indicesMap[index].second, alphabet, indicesMap, startPoints);
    return left + "+" + right;

}

std::string REtoString (const int result, const int lastIdx, const std::set<char> &alphabet, const int *startPoints, const int *dev_left_indices,
                   const int *dev_right_indices, const int *dev_temp_left_indices, const int *dev_temp_right_indices) {

    auto *LIdx = new int [1];
    auto *RIdx = new int [1];

    checkCuda( cudaMemcpy(LIdx, dev_temp_left_indices + result, sizeof(int), cudaMemcpyDeviceToHost) );
    checkCuda( cudaMemcpy(RIdx, dev_temp_right_indices + result, sizeof(int), cudaMemcpyDeviceToHost) );

    auto nonatomicIndex = static_cast<int> (alphabet.size());

    int *dev_resultIndices;
    checkCuda( cudaMalloc(&dev_resultIndices, 600 * sizeof(int)) );

    thrust::device_ptr<int> dev_resultIndices_ptr(dev_resultIndices);
    thrust::fill(dev_resultIndices_ptr, dev_resultIndices_ptr + 600, -1);

    if (*LIdx >= nonatomicIndex) generateResultIndices<<<1, 1>>>(*LIdx, nonatomicIndex, dev_left_indices, dev_right_indices, dev_resultIndices);
    if (*RIdx >= nonatomicIndex) generateResultIndices<<<1, 1>>>(*RIdx, nonatomicIndex, dev_left_indices, dev_right_indices, dev_resultIndices);

    int resultIndices[600];
    checkCuda( cudaMemcpy(resultIndices, dev_resultIndices, 600 * sizeof(int), cudaMemcpyDeviceToHost) );

    std::map<int, std::pair<int, int>> indicesMap;

    indicesMap.insert( std::make_pair(INT_MAX - 1,  std::make_pair(*LIdx, *RIdx)));

    int i = 0;
    while (resultIndices[i] != -1 && i + 2 < 600) {
        int re = resultIndices[i];
        int l = resultIndices[i + 1];
        int r = resultIndices[i + 2];
        indicesMap.insert( std::make_pair(re,  std::make_pair(l, r)));
        i += 3;
    }

    if (i + 2 >= 600) return "Size of the result is too big";

    cudaFree(dev_resultIndices);

    return toString(INT_MAX - 1, alphabet, indicesMap, startPoints);
    
}

void storeUniqueREs(int N, int &lastIdx, const int cache_capacity, bool &onTheFly, UINT64 *dev_cache, UINT64 *dev_temp_cache,
                    int *dev_left_indices, int *dev_right_indices, int *dev_temp_left_indices, int *dev_temp_right_indices) {

    thrust::device_ptr<UINT64> new_end_ptr;
    thrust::device_ptr<UINT64> dev_cache_ptr(dev_cache + 2 * lastIdx);
    thrust::device_ptr<UINT64> dev_temp_cache_ptr(dev_temp_cache);
    thrust::device_ptr<int> dev_left_indices_ptr(dev_left_indices + lastIdx);
    thrust::device_ptr<int> dev_right_indices_ptr(dev_right_indices + lastIdx);
    thrust::device_ptr<int> dev_temp_left_indices_ptr(dev_temp_left_indices);
    thrust::device_ptr<int> dev_temp_right_indices_ptr(dev_temp_right_indices);

    new_end_ptr = // end of dev_temp_cache
    thrust::remove(dev_temp_cache_ptr, dev_temp_cache_ptr + 2 * N, UINT64(0) - 1);
    thrust::remove(dev_temp_left_indices_ptr, dev_temp_left_indices_ptr + N, -1);
    thrust::remove(dev_temp_right_indices_ptr, dev_temp_right_indices_ptr + N, -1);

    int numberOfNewUniqueREs = static_cast<int>(new_end_ptr - dev_temp_cache_ptr) / 2;
    if (lastIdx + numberOfNewUniqueREs > cache_capacity) {
        N = cache_capacity - lastIdx;
        onTheFly = true;
    } else N = numberOfNewUniqueREs;

    thrust::copy_n(dev_temp_cache_ptr, 2 * N, dev_cache_ptr);
    thrust::copy_n(dev_temp_left_indices_ptr, N, dev_left_indices_ptr);
    thrust::copy_n(dev_temp_right_indices_ptr, N, dev_right_indices_ptr);

    lastIdx += N;

}

std::string enumerateAndContainsCheckREc(const std::set<char> &alphabet, const unsigned short maxCost, const int GTrows,
                              const int GTcolumns, const UINT64 lPosBits, const UINT64 hPosBits, const UINT64 lNegBits, const UINT64 hNegBits,
                              const unsigned short *costFun, const UINT64 *guideTable, UINT64 &allREs, int &REcost) {
    
    // Initialisation

    int cache_capacity      = 200000000;
    int temp_cache_capacity = 100000000;

    int c1 = costFun[0]; // cost of a
    int c2 = costFun[1]; // cost of ?
    int c3 = costFun[2]; // cost of *
    int c4 = costFun[3]; // cost of .
    int c5 = costFun[4]; // cost of +

    allREs = 2;
    int lastIdx{};
    UINT64 idx = 2;
    auto *cache = new UINT64 [alphabet.size() * 2];
    auto nonatomicIndex = static_cast<int> (alphabet.size());

    // Alphabet
    #ifndef MEASUREMENT_MODE
        printf("Cost %-2d | (A) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n", c1, allREs, lastIdx, static_cast<int>(alphabet.size()));
    #endif

    for (; lastIdx < nonatomicIndex; ++lastIdx) {

        UINT64 hREc = 0;
        UINT64 lREc = idx;

        cache[lastIdx * 2] = hREc;
        cache[lastIdx * 2 + 1] = lREc;

        if ((hREc & hPosBits) == hPosBits && (lREc & lPosBits) == lPosBits &&
            (~hREc & hNegBits) == hNegBits && (~lREc & lNegBits) == lNegBits) {
            std::string s(1, *next(alphabet.begin(), lastIdx));
            return s;
        }
        
        idx <<= 1;
        allREs++;
    }

    // 4 for "*", ".", "+" and "?"
    auto* startPoints = new int [(maxCost + 2) * 4]();

    startPoints[c1 * 4 + 3] = lastIdx;
    startPoints[(c1 + 1) * 4] = lastIdx;

    int *dev_result;
    auto *result = new int [1]; *result = -1;
    checkCuda( cudaMalloc(&dev_result, sizeof(int)) );
    checkCuda( cudaMemcpy(dev_result, result, sizeof(int), cudaMemcpyHostToDevice) );

    UINT64 *dev_cache, *dev_temp_cache;
    int *dev_left_indices, *dev_right_indices, *dev_temp_left_indices, *dev_temp_right_indices;
    checkCuda( cudaMalloc(&dev_cache, 2 * cache_capacity * sizeof(UINT64)) );
    checkCuda( cudaMalloc(&dev_temp_cache, 2 * temp_cache_capacity * sizeof(UINT64)) );
    checkCuda( cudaMalloc(&dev_left_indices, cache_capacity * sizeof(int)) );
    checkCuda( cudaMalloc(&dev_right_indices, cache_capacity * sizeof(int)) );
    checkCuda( cudaMalloc(&dev_temp_left_indices, temp_cache_capacity * sizeof(int)) );
    checkCuda( cudaMalloc(&dev_temp_right_indices, temp_cache_capacity * sizeof(int)) );

    using hash_set_t = warpcore::HashSet<
    UINT64,         // key type
    UINT64(0) - 1,  // empty key
    UINT64(0) - 2,  // tombstone key
    warpcore::probing_schemes::QuadraticProbing<warpcore::hashers::MurmurHash <UINT64>>>;
    
    hash_set_t cHashSet(2 * cache_capacity);
    hash_set_t iHashSet(2 * cache_capacity);

    InitialiseHashSetsByAlphabet<hash_set_t><<<1, nonatomicIndex>>>(cHashSet, iHashSet, dev_cache);

    checkCuda( cudaMemcpyToSymbol(dev_guideTable, guideTable, GTrows * GTcolumns * sizeof(UINT64)) );
    checkCuda( cudaMemcpy(dev_cache, cache, 2 * alphabet.size() * sizeof(UINT64), cudaMemcpyHostToDevice) );

    bool onTheFly = false, lastRound = false;
    int shortageCost = -1;

    // Enumeration
    for (REcost = c1 + 1; REcost <= maxCost; ++REcost) {

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
                    y = x + std::min(temp_cache_capacity - 1, qIdx2 - x);
                    qN = (y - x + 1);
                    #ifndef MEASUREMENT_MODE
                        printf("Cost %-2d | (Q) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n", REcost, allREs, lastIdx, qN);
                    #endif
                    int qBlc = (qN + 1023) / 1024;
                    QuestionMark<hash_set_t><<<qBlc, 1024>>>(x, y, onTheFly, cHashSet, iHashSet, dev_cache, dev_temp_cache, dev_temp_left_indices, dev_temp_right_indices, lPosBits, hPosBits, lNegBits, hNegBits, dev_result);
                    checkCuda( cudaPeekAtLastError() );
                    checkCuda( cudaMemcpy(result, dev_result, sizeof(int), cudaMemcpyDeviceToHost) );
                    allREs += qN;
                    if (*result != -1) {startPoints[REcost * 4 + 1] = INT_MAX; goto exitEnumeration;}
                    if (!onTheFly) storeUniqueREs(qN, lastIdx, cache_capacity, onTheFly, dev_cache, dev_temp_cache, dev_left_indices, dev_right_indices, dev_temp_left_indices, dev_temp_right_indices);
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
                    y = x + std::min(temp_cache_capacity - 1, sIdx2 - x);
                    sN = (y - x + 1);
                    #ifndef MEASUREMENT_MODE
                        printf("Cost %-2d | (S) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n", REcost, allREs, lastIdx, sN);
                    #endif
                    int sBlc = (sN + 1023) / 1024;
                    Star<hash_set_t><<<sBlc, 1024>>>(x, y, onTheFly, cHashSet, iHashSet, dev_cache, dev_temp_cache, dev_temp_left_indices, dev_temp_right_indices, nonatomicIndex, GTrows, GTcolumns, lPosBits, hPosBits, lNegBits, hNegBits, dev_result);
                    checkCuda( cudaPeekAtLastError() );
                    checkCuda( cudaMemcpy(result, dev_result, sizeof(int), cudaMemcpyDeviceToHost) );
                    allREs += sN;
                    if (*result != -1) {startPoints[REcost * 4 + 2] = INT_MAX; goto exitEnumeration;}
                    if (!onTheFly) storeUniqueREs(sN, lastIdx, cache_capacity, onTheFly, dev_cache, dev_temp_cache, dev_left_indices, dev_right_indices, dev_temp_left_indices, dev_temp_right_indices);
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
                    y = x + std::min(temp_cache_capacity / (2 * (cIdx2 - cIdx1 + 1)) - 1, cIdx4 - x); // 2 is for concat only for lr and rl
                    cN = (y - x + 1) * (cIdx2 - cIdx1 + 1);
                    #ifndef MEASUREMENT_MODE
                        printf("Cost %-2d | (C) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n", REcost, allREs, lastIdx, 2 * cN);
                    #endif
                    int cBlc = (cN + 1023) / 1024;
                    Concat<hash_set_t><<<cBlc, 1024>>>(cIdx1, cIdx2, x, y, onTheFly, cHashSet, iHashSet, dev_cache, dev_temp_cache, dev_temp_left_indices, dev_temp_right_indices, nonatomicIndex, GTrows, GTcolumns, lPosBits, hPosBits, lNegBits, hNegBits, dev_result);
                    checkCuda( cudaPeekAtLastError() );
                    checkCuda( cudaMemcpy(result, dev_result, sizeof(int), cudaMemcpyDeviceToHost) );
                    allREs += 2 * cN;
                    if (*result != -1) {startPoints[REcost * 4 + 3] = INT_MAX; goto exitEnumeration;}
                    if (!onTheFly) storeUniqueREs(2 * cN, lastIdx, cache_capacity, onTheFly, dev_cache, dev_temp_cache, dev_left_indices, dev_right_indices, dev_temp_left_indices, dev_temp_right_indices);
                    x = y + 1;
                } while (y < cIdx4);
            }

        }
        startPoints[REcost * 4 + 3] = lastIdx;

        // Or
        if (c1 + c5 < c2 && 2 * c1 <= REcost - c5) { // Where C(x + Epsilon) < C(x?)

            int oIdx1 = startPoints[(REcost - c1 - c5) * 4];
            int oIdx2 = startPoints[(REcost - c1 - c5 + 1) * 4] - 1;
            int oN = oIdx2 - oIdx1 + 1;

            if (oN){
                int x = oIdx1, y;
                do {
                    y = x + std::min(temp_cache_capacity - 1, oIdx2 - x);
                    oN = (y - x + 1);
                    #ifndef MEASUREMENT_MODE
                        printf("Cost %-2d | (O) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n", REcost, allREs, lastIdx, oN);
                    #endif
                    int qBlc = (oN + 1023) / 1024;
                    OrEpsilon<hash_set_t><<<qBlc, 1024>>>(x, y, onTheFly, cHashSet, iHashSet, dev_cache, dev_temp_cache, dev_temp_left_indices, dev_temp_right_indices, lPosBits, hPosBits, lNegBits, hNegBits, dev_result);
                    checkCuda( cudaPeekAtLastError() );
                    checkCuda( cudaMemcpy(result, dev_result, sizeof(int), cudaMemcpyDeviceToHost) );
                    allREs += oN;
                    if (*result != -1) {startPoints[(REcost + 1) * 4] = INT_MAX; goto exitEnumeration;}
                    if (!onTheFly) storeUniqueREs(oN, lastIdx, cache_capacity, onTheFly, dev_cache, dev_temp_cache, dev_left_indices, dev_right_indices, dev_temp_left_indices, dev_temp_right_indices);
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
                    y = x + std::min(temp_cache_capacity / (oIdx2 - oIdx1 + 1) - 1, oIdx4 - x);
                    oN = (y - x + 1) * (oIdx2 - oIdx1 + 1);
                    #ifndef MEASUREMENT_MODE
                        printf("Cost %-2d | (O) | AllREs: %-11lu | StoredREs: %-10d | ToBeChecked: %-10d \n", REcost, allREs, lastIdx, oN);
                    #endif
                    int oBlc = (oN + 1023) / 1024;
                    Or<hash_set_t><<<oBlc, 1024>>>(oIdx1, oIdx2, x, y, onTheFly, cHashSet, iHashSet, dev_cache, dev_temp_cache, dev_temp_left_indices, dev_temp_right_indices, lPosBits, hPosBits, lNegBits, hNegBits, dev_result);
                    checkCuda( cudaPeekAtLastError() );
                    checkCuda( cudaMemcpy(result, dev_result, sizeof(int), cudaMemcpyDeviceToHost) );
                    allREs += oN;
                    if (*result != -1) {startPoints[(REcost + 1) * 4] = INT_MAX; goto exitEnumeration;}
                    if (!onTheFly) storeUniqueREs(oN, lastIdx, cache_capacity, onTheFly, dev_cache, dev_temp_cache, dev_left_indices, dev_right_indices, dev_temp_left_indices, dev_temp_right_indices);
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
    if (*result != -1) output = REtoString(*result, lastIdx, alphabet, startPoints, 
    dev_left_indices, dev_right_indices, dev_temp_left_indices, dev_temp_right_indices);

    cudaFree(dev_cache);
    cudaFree(dev_result);
    cudaFree(dev_temp_cache);
    cudaFree(dev_left_indices);
    cudaFree(dev_right_indices);
    cudaFree(dev_temp_left_indices);
    cudaFree(dev_temp_right_indices);

    return output;
}

std::set<std::string, strLenComparison> infixesOf (const std::string &word){
    std::set<std::string, strLenComparison> ic;
    for (int len = 0; len <= word.length(); ++len) {
        for (int index = 0; index < word.length() - len + 1 ; ++index) {
            ic.insert(word.substr(index, len));
        }
    }
    return ic;
}

bool initialiseGuideTable(std::set<std::string, strLenComparison> &ic, int &GTrows, int &GTcolumns,
                             UINT64 &lPosBits, UINT64 &hPosBits, UINT64 &lNegBits, UINT64 &hNegBits, UINT64 **guideTable,
                             const std::vector<std::string> &pos, const std::vector<std::string> &neg) {

    for (const std::string& word : pos) {
        std::set<std::string, strLenComparison> set1 = infixesOf(word);
        ic.insert(set1.begin(), set1.end());
    }

    for (const std::string& word : neg) {
        std::set<std::string, strLenComparison> set1 = infixesOf(word);
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

    GTrows = static_cast<int> (gt.size());
    GTcolumns = static_cast<int> (gt.back().size());

    if (GTrows > 2 * 63) {
        printf("Your input needs %u bits which exceeds 126 bits ", GTrows);
        printf("(current version).\nPlease use less/shorter words and run the code again.\n");
        return false;
    }

    *guideTable = new UINT64 [GTrows * GTcolumns];

    for (int i = 0; i < GTrows; ++i) {
        for (int j = 0; j < gt.at(i).size(); ++j) {
            (*guideTable)[i * GTcolumns + j] = gt.at(i).at(j);
        }
    }

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

void initialiseAlphabet(std::set<char> &alphabet, const std::vector<std::string> &pos, const std::vector<std::string> &neg) {
    for (auto & word : pos) for (auto ch : word) alphabet.insert(ch);
    for (auto & word : neg) for (auto ch : word) alphabet.insert(ch);
}

bool readFile(const std::string& fileName, std::vector<std::string> &pos, std::vector<std::string> &neg) {

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
                    printf("%s is in both Pos and Neg examples", line.c_str());
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

    std::string fileName = argv[1];
    std::vector<std::string> pos, neg;
    if (!readFile(fileName, pos, neg)) return 0;
    unsigned short costFun[5];
    for (int i = 0; i < 5; i++) costFun[i] = std::atoi(argv[i + 2]);
    unsigned short maxCost = std::atoi(argv[7]);

    auto start = std::chrono::high_resolution_clock::now();

    std::string result;
    std::set<char> alphabet;
    initialiseAlphabet(alphabet, pos, neg);
    int GTrows, GTcolumns, REcost = costFun[0];
    std::set<std::string, strLenComparison> ic = {""};
    UINT64 allREs = 1, lPosBits{}, hPosBits{}, lNegBits{}, hNegBits{}, *guideTable;

    if (!initialiseGuideTable(ic, GTrows, GTcolumns, lPosBits, hPosBits, 
    lNegBits, hNegBits, &guideTable, pos, neg)) return 0;
    
    if (pos.empty()) result = "empty";
    else if ((pos.size() == 1) && (pos.at(0).empty())) {result = ""; allREs++;}
    else result = enumerateAndContainsCheckREc(alphabet, maxCost, GTrows, 
        GTcolumns, lPosBits, hPosBits, lNegBits, hNegBits, costFun, guideTable, allREs, REcost);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    printf("\nPositive: "); for (const auto& p : pos) printf("\"%s\" ", p.c_str());
    printf("\nNegative: "); for (const auto& n : neg) printf("\"%s\" ", n.c_str());
    printf("\nCost Function: %u, %u, %u, %u, %u", costFun[0], costFun[1], costFun[2], costFun[3], costFun[4]);
    printf("\nSize of IC: %u", GTrows);
    printf("\nCost of Final RE: %d", REcost);
    printf("\nNumber of All REs: %lu", allREs);
    printf("\nRunning Time: %f s", (double) duration * 0.000001);
    printf("\n\nRE: \"%s\"\n", result.c_str());

    return 0;
    
}
