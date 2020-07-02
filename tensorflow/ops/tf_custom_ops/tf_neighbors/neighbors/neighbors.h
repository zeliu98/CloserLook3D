

#include "../../cpp_utils/cloud/cloud.h"
#include "../../cpp_utils/nanoflann/nanoflann.hpp"

#include <set>
#include <cstdint>
#include <cstdlib>
#include <time.h>
using namespace std;


void ordered_neighbors(vector<PointXYZ>& queries,
                        vector<PointXYZ>& supports,
                        vector<int>& neighbors_indices,
                        float radius);

void batch_ordered_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius);

void batch_nanoflann_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius);

void batch_nanoflann_neighbors_pad(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius);