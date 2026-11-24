#pragma once
#include <tiny-cuda-nn/common.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace ncr {
template<typename T>
std::vector<T> diagnosticDownload(const T* data, size_t count)
{
    std::vector<T> values(count);
    if (count) CUDA_CHECK_THROW(cudaMemcpy(values.data(), data, count*sizeof(T), cudaMemcpyDeviceToHost));
    return values;
}

template<typename T>
tcnn::json outputStatistics(const T* data, uint32_t count, uint32_t stride, uint32_t channels)
{
    auto values=diagnosticDownload(data,size_t(count)*stride);
    std::vector<double> sums(channels,0.0), lums;
    size_t nonfinite=0;
    for(uint32_t i=0;i<count;++i) {
        bool finite=true;
        for(uint32_t c=0;c<channels;++c) {
            const double x=double(values[size_t(i)*stride+c]);
            if(!std::isfinite(x)) { ++nonfinite; finite=false; } else sums[c]+=x;
        }
        if(finite) lums.push_back(.2126*double(values[size_t(i)*stride])+.7152*double(values[size_t(i)*stride+1])+.0722*double(values[size_t(i)*stride+2]));
    }
    double sum=0, square=0;
    for(double x:lums) { sum+=x; square+=x*x; }
    for(double& x:sums) x/=std::max(1u,count);
    std::sort(lums.begin(),lums.end());
    const double n=std::max(size_t(1),lums.size()), mean=sum/n;
    return {{"count",count},{"nonfinite",nonfinite},{"channel_mean",sums},
        {"luminance_mean",mean},{"luminance_std",std::sqrt(std::max(0.0,square/n-mean*mean))},
        {"luminance_p95",lums.empty()?0.0:lums[size_t(.95*(lums.size()-1))]}};
}

template<typename T>
tcnn::json gradientStatistics(const T* data, size_t count, bool active, float lossScale)
{
    if(!active) return {{"active",false}}; // Do not report stale gradients of skipped branches.
    auto values=diagnosticDownload(data,count);
    double square=0, maxAbs=0;
    size_t nonfinite=0, nonzero=0, nan=0, inf=0;
    for(auto value:values) {
        const double x=double(value)/lossScale;
        if(!std::isfinite(x)) { ++nonfinite; nan+=std::isnan(x); inf+=std::isinf(x); continue; }
        square+=x*x; maxAbs=std::max(maxAbs,std::abs(x)); nonzero+=x!=0;
    }
    return {{"active",true},{"count",count},{"l2",std::sqrt(square)},
        {"rms",std::sqrt(square/std::max(size_t(1),count))},{"max_abs",maxAbs},
        {"nonzero",nonzero},{"nonfinite",nonfinite},{"nan",nan},{"inf",inf},{"norms_use_finite_entries_only",true},{"removed_loss_scale",lossScale}};
}
template<typename T>
tcnn::json inputColumnStatistics(const T* data, uint32_t count, uint32_t width)
{
    auto values=diagnosticDownload(data,size_t(count)*width);
    std::vector<uint32_t> nonfinite(width,0);
    uint32_t badRows=0;
    tcnn::json first=nullptr;
    for(uint32_t r=0;r<count;++r) {
        bool bad=false;
        for(uint32_t c=0;c<width;++c) if(!std::isfinite(float(values[size_t(r)*width+c]))) { ++nonfinite[c]; bad=true; }
        if(bad) {
            ++badRows;
            if(first.is_null()) {
                std::vector<float> row(width);
                for(uint32_t c=0;c<width;++c) row[c]=float(values[size_t(r)*width+c]);
                first={{"row",r},{"values",row}};
            }
        }
    }
    return {{"bad_rows",badRows},{"nonfinite_by_column",nonfinite},{"first_bad_row",first}};
}

} // namespace ncr
