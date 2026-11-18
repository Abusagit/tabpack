// Experimental BF16 grouped GEMM. Row-major products are expressed as C^T=B^T A^T.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <dlfcn.h>

struct Plan {
  std::vector<int> sizes, ld, counts;
  std::vector<cublasOperation_t> normal, transpose;
  std::vector<float> one, zero, a, b, c;
  torch::Tensor x, A, B, y, pointers;
  using Gemm = decltype(&cublasGemmGroupedBatchedEx);
  Gemm gemm;
  int version;
  Plan(torch::Tensor input, std::vector<int> widths, std::vector<int> members) : sizes(widths) {
    TORCH_CHECK(input.is_cuda() && input.dim()==3 && input.size(1)==input.size(2), "CUDA square pack required");
    TORCH_CHECK(members.size()==sizes.size() && !sizes.empty(), "member/width count mismatch");
    c10::cuda::CUDAGuard guard(input.device());
    gemm = reinterpret_cast<Gemm>(dlsym(RTLD_DEFAULT, "cublasGemmGroupedBatchedEx"));
    TORCH_CHECK(gemm, "loaded cuBLAS does not export cublasGemmGroupedBatchedEx (requires CUDA >=12.5)");
    cublasGetVersion(at::cuda::getCurrentCUDABlasHandle(), &version);
    auto opts=input.options().dtype(torch::kBFloat16);
    x=torch::zeros(input.sizes(),opts); A=torch::zeros_like(x);
    B=torch::zeros_like(x); y=torch::zeros_like(x);
    int n=sizes.size(), stride=input.size(2);
    ld.assign(n,stride); counts.assign(n,1);
    normal.assign(n,CUBLAS_OP_N); transpose.assign(n,CUBLAS_OP_T);
    one.assign(n,1); zero.assign(n,0); a.assign(n,3.4445f);
    b.assign(n,-4.7750f); c.assign(n,2.0315f);
    auto host=torch::empty({4*n},torch::TensorOptions().dtype(torch::kInt64));
    auto p=host.data_ptr<int64_t>();
    std::vector<torch::Tensor> buffers={x,A,B,y};
    for(int j=0;j<4;j++) for(int i=0;i<n;i++) {
      TORCH_CHECK(sizes[i]>0 && sizes[i]<=stride,"invalid width");
      TORCH_CHECK(members[i]>=0 && members[i]<input.size(0), "invalid member index");
      p[j*n+i]=reinterpret_cast<int64_t>(buffers[j].data_ptr())+members[i]*stride*stride*2;
    }
    pointers=host.to(input.device());
  }
  void multiply(int left,int right,int out,bool transpose_right,
                std::vector<float>& alpha,std::vector<float>& beta) {
    auto handle=at::cuda::getCurrentCUDABlasHandle();
    // PyTorch owns the handle and its stream; do not change its pointer/math modes.
    cublasPointerMode_t mode;
    cublasGetPointerMode(handle,&mode);
    TORCH_CHECK(mode==CUBLAS_POINTER_MODE_HOST,"host pointer mode required");
    int n=sizes.size();
    auto ptr=pointers.data_ptr<int64_t>();
    auto status=gemm(handle,transpose_right?transpose.data():normal.data(),normal.data(),
      sizes.data(),sizes.data(),sizes.data(),alpha.data(),
      reinterpret_cast<const void* const*>(ptr+right*n),CUDA_R_16BF,ld.data(),
      reinterpret_cast<const void* const*>(ptr+left*n),CUDA_R_16BF,ld.data(),beta.data(),
      reinterpret_cast<void* const*>(ptr+out*n),CUDA_R_16BF,ld.data(),n,counts.data(),CUBLAS_COMPUTE_32F);
    TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"BF16 grouped GEMM failed, cuBLAS status=",int(status),", version=",version);
  }
  torch::Tensor run(torch::Tensor normalized,int steps) {
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(normalized.device()==x.device() && normalized.sizes()==x.sizes() && normalized.scalar_type()==torch::kBFloat16,"plan input mismatch");
    x.copy_(normalized);
    int current=0, next=3;
    for(int i=0;i<steps;i++) {
      multiply(current,current,1,true,one,zero); // X X^T
      B.copy_(A);
      multiply(1,1,2,false,c,b);
      auto source=current==0?x:y;
      auto target=next==0?x:y;
      target.copy_(source);
      multiply(2,current,next,false,one,a);
      std::swap(current,next);
    }
    return current==0?x:y;
  }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<Plan>(m,"Plan").def(pybind11::init<torch::Tensor,std::vector<int>,std::vector<int>>())
    .def("run",&Plan::run).def_readonly("version",&Plan::version);
}
