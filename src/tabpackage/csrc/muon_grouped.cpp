// Experimental BF16 grouped GEMM. Row-major products are expressed as C^T=B^T A^T.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <c10/cuda/CUDAStream.h>
#include <dlfcn.h>
#include <map>

struct Plan {
  std::vector<int> sizes, columns, ld_x, ld_gram, counts;
  std::vector<cublasOperation_t> normal, transpose;
  std::vector<float> one, zero, a, b, c;
  torch::Tensor x, A, B, y, pointers;
  using Gemm = decltype(&cublasGemmGroupedBatchedEx);
  Gemm gemm;
  int version, input_columns, workspace_columns;
  struct LtCase {
    cublasLtMatmulDesc_t desc=nullptr;
    cublasLtMatrixLayout_t left=nullptr,right=nullptr,c=nullptr,d=nullptr;
    cublasLtMatmulAlgo_t algorithm{};
  };
  cublasLtHandle_t lt=nullptr;
  std::map<int,LtCase> lt_cases;
  torch::Tensor lt_workspace;
  std::vector<int> member_ids;
  std::string wide_backend="grouped";
  Plan(torch::Tensor input, std::vector<int> widths, std::vector<int> members, bool rectangular=false,
       std::string wide_strategy="auto") : sizes(widths), member_ids(members) {
    TORCH_CHECK(wide_strategy=="auto" || wide_strategy=="grouped" || wide_strategy=="lt", "unknown BX backend");
    TORCH_CHECK(input.is_cuda() && input.dim()==3, "CUDA matrix pack required");
    TORCH_CHECK(rectangular ? input.size(1)<=input.size(2) : input.size(1)==input.size(2), "unsupported orientation");
    TORCH_CHECK(members.size()==sizes.size() && !sizes.empty(), "member/width count mismatch");
    c10::cuda::CUDAGuard guard(input.device());
    gemm = reinterpret_cast<Gemm>(dlsym(RTLD_DEFAULT, "cublasGemmGroupedBatchedEx"));
    TORCH_CHECK(gemm, "loaded cuBLAS does not export cublasGemmGroupedBatchedEx (requires CUDA >=12.5)");
    cublasGetVersion(at::cuda::getCurrentCUDABlasHandle(), &version);
    auto opts=input.options().dtype(torch::kBFloat16);
    input_columns=input.size(2);
    bool try_lt=rectangular && (wide_strategy=="lt" || (wide_strategy=="auto" && input_columns>input.size(1)));
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,input.get_device());
    // Lt avoids the slow grouped BX kernel, so align8 is now useful for Maps Gram too.
    bool align_lt=try_lt && properties.major>=8;
    workspace_columns=rectangular && (input_columns%2 || align_lt) ? ((input_columns+7)/8)*8 : input_columns;
    int n=sizes.size(), rows=input.size(1), cols=workspace_columns;
    x=torch::zeros({input.size(0),rows,cols},opts); y=torch::zeros_like(x);
    A=torch::zeros({input.size(0),rows,rows},opts); B=torch::zeros_like(A);
    columns=rectangular ? std::vector<int>(n,cols) : sizes;
    ld_x.assign(n,cols); ld_gram.assign(n,rows); counts.assign(n,1);
    normal.assign(n,CUBLAS_OP_N); transpose.assign(n,CUBLAS_OP_T);
    one.assign(n,1); zero.assign(n,0); a.assign(n,3.4445f);
    b.assign(n,-4.7750f); c.assign(n,2.0315f);
    auto host=torch::empty({4*n},torch::TensorOptions().dtype(torch::kInt64));
    auto p=host.data_ptr<int64_t>();
    std::vector<torch::Tensor> buffers={x,A,B,y};
    for(int j=0;j<4;j++) for(int i=0;i<n;i++) {
      TORCH_CHECK(sizes[i]>0 && sizes[i]<=rows,"invalid width");
      TORCH_CHECK(members[i]>=0 && members[i]<input.size(0), "invalid member index");
      int64_t matrix_elements=(j==0 || j==3) ? int64_t(rows)*cols : int64_t(rows)*rows;
      p[j*n+i]=reinterpret_cast<int64_t>(buffers[j].data_ptr())+members[i]*matrix_elements*2;
    }
    pointers=host.to(input.device());
    if(try_lt) {
      if(initialize_lt()) wide_backend="lt";
      else lt_workspace=torch::Tensor();
    }
  }
  ~Plan() {
    for(auto& entry:lt_cases) {
      auto& c=entry.second;
      if(c.desc) cublasLtMatmulDescDestroy(c.desc);
      if(c.left) cublasLtMatrixLayoutDestroy(c.left);
      if(c.right) cublasLtMatrixLayoutDestroy(c.right);
      if(c.c) cublasLtMatrixLayoutDestroy(c.c);
      if(c.d) cublasLtMatrixLayoutDestroy(c.d);
    }
    if(lt) cublasLtDestroy(lt);
  }
  static uint32_t alignment(const void* ptr) {
    uintptr_t value=reinterpret_cast<uintptr_t>(ptr);
    uint32_t result=1;
    while(result<256 && value%(2*result)==0) result*=2;
    return result;
  }
  bool initialize_lt() {
    if(cublasLtCreate(&lt)!=CUBLAS_STATUS_SUCCESS) return false;
    lt_workspace=torch::empty({64*1024*1024},x.options().dtype(torch::kUInt8));
    uint32_t ax=256,ab=256;
    int rows=x.size(1),cols=x.size(2);
    for(int i:member_ids) {
      ax=std::min(ax,alignment(x.data_ptr<at::BFloat16>()+int64_t(i)*rows*cols));
      ax=std::min(ax,alignment(y.data_ptr<at::BFloat16>()+int64_t(i)*rows*cols));
      ab=std::min(ab,alignment(B.data_ptr<at::BFloat16>()+int64_t(i)*rows*rows));
    }
    for(int width:sizes) {
      if(lt_cases.count(width)) continue;
      auto& c=lt_cases[width];
      if(cublasLtMatmulDescCreate(&c.desc,CUBLAS_COMPUTE_32F,CUDA_R_32F)!=CUBLAS_STATUS_SUCCESS ||
         cublasLtMatrixLayoutCreate(&c.left,CUDA_R_16BF,cols,width,cols)!=CUBLAS_STATUS_SUCCESS ||
         cublasLtMatrixLayoutCreate(&c.right,CUDA_R_16BF,width,width,rows)!=CUBLAS_STATUS_SUCCESS ||
         cublasLtMatrixLayoutCreate(&c.c,CUDA_R_16BF,cols,width,cols)!=CUBLAS_STATUS_SUCCESS ||
         cublasLtMatrixLayoutCreate(&c.d,CUDA_R_16BF,cols,width,cols)!=CUBLAS_STATUS_SUCCESS) return false;
      cublasLtMatmulPreference_t pref;
      if(cublasLtMatmulPreferenceCreate(&pref)!=CUBLAS_STATUS_SUCCESS) return false;
      size_t bytes=lt_workspace.numel();
      cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes));
      cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,&ax,sizeof(ax));
      cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,&ab,sizeof(ab));
      cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,&ax,sizeof(ax));
      cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,&ax,sizeof(ax));
      cublasLtMatmulHeuristicResult_t result; int count=0;
      auto status=cublasLtMatmulAlgoGetHeuristic(lt,c.desc,c.left,c.right,c.c,c.d,pref,1,&result,&count);
      cublasLtMatmulPreferenceDestroy(pref);
      if(status!=CUBLAS_STATUS_SUCCESS || count!=1 || result.state!=CUBLAS_STATUS_SUCCESS) return false;
      c.algorithm=result.algo;
    }
    return true;
  }
  bool wide_lt(torch::Tensor source,torch::Tensor target) {
    float alpha=1,beta=3.4445f;
    int rows=x.size(1),cols=x.size(2);
    for(size_t j=0;j<sizes.size();j++) {
      int i=member_ids[j]; auto& c=lt_cases.at(sizes[j]);
      auto px=source.data_ptr<at::BFloat16>()+int64_t(i)*rows*cols;
      auto pb=B.data_ptr<at::BFloat16>()+int64_t(i)*rows*rows;
      auto py=target.data_ptr<at::BFloat16>()+int64_t(i)*rows*cols;
      auto status=cublasLtMatmul(lt,c.desc,&alpha,px,c.left,pb,c.right,&beta,px,c.c,py,c.d,
        &c.algorithm,lt_workspace.data_ptr(),lt_workspace.numel(),
        c10::cuda::getCurrentCUDAStream(x.get_device()).stream());
      if(status!=CUBLAS_STATUS_SUCCESS) return false;
    }
    return true;
  }
  void multiply(int left,int right,int out,bool transpose_right,
                std::vector<float>& alpha,std::vector<float>& beta,
                std::vector<int>& m,std::vector<int>& n_cols,std::vector<int>& k) {
    auto handle=at::cuda::getCurrentCUDABlasHandle();
    // PyTorch owns the handle and its stream; do not change its pointer/math modes.
    cublasPointerMode_t mode;
    cublasGetPointerMode(handle,&mode);
    TORCH_CHECK(mode==CUBLAS_POINTER_MODE_HOST,"host pointer mode required");
    int n=sizes.size();
    auto ptr=pointers.data_ptr<int64_t>();
    auto& ld_left=(left==0 || left==3) ? ld_x : ld_gram;
    auto& ld_right=(right==0 || right==3) ? ld_x : ld_gram;
    auto& ld_out=(out==0 || out==3) ? ld_x : ld_gram;
    auto status=gemm(handle,transpose_right?transpose.data():normal.data(),normal.data(),
      n_cols.data(),m.data(),k.data(),alpha.data(),
      reinterpret_cast<const void* const*>(ptr+right*n),CUDA_R_16BF,ld_right.data(),
      reinterpret_cast<const void* const*>(ptr+left*n),CUDA_R_16BF,ld_left.data(),beta.data(),
      reinterpret_cast<void* const*>(ptr+out*n),CUDA_R_16BF,ld_out.data(),n,counts.data(),CUBLAS_COMPUTE_32F);
    TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"BF16 grouped GEMM failed, cuBLAS status=",int(status),", version=",version);
  }
  torch::Tensor run(torch::Tensor normalized,int steps) {
    c10::cuda::CUDAGuard guard(x.device());
    auto input_view=x.narrow(2,0,input_columns);
    TORCH_CHECK(normalized.device()==x.device() && normalized.sizes()==input_view.sizes() && normalized.scalar_type()==torch::kBFloat16,"plan input mismatch");
    input_view.copy_(normalized);
    if(workspace_columns>input_columns)
      x.narrow(2,input_columns,workspace_columns-input_columns).zero_();
    int current=0, next=3;
    for(int i=0;i<steps;i++) {
      multiply(current,current,1,true,one,zero,sizes,sizes,columns); // X X^T
      B.copy_(A);
      multiply(1,1,2,false,c,b,sizes,sizes,sizes);
      auto source=current==0?x:y;
      auto target=next==0?x:y;
      // Lt reads C=source and writes D=target, so a full wide copy is unnecessary.
      if(wide_backend!="lt" || !wide_lt(source,target)) {
        wide_backend="grouped"; // Safe fallback if no supported Lt algorithm exists.
        target.copy_(source);
        multiply(2,current,next,false,one,a,sizes,columns,sizes);
      }
      std::swap(current,next);
    }
    return (current==0?x:y).narrow(2,0,input_columns);
  }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<Plan>(m,"Plan").def(pybind11::init<torch::Tensor,std::vector<int>,std::vector<int>,bool,std::string>(),
      pybind11::arg("input"),pybind11::arg("widths"),pybind11::arg("members"),pybind11::arg("rectangular")=false,
      pybind11::arg("wide_strategy")="auto")
    .def("run",&Plan::run).def_readonly("version",&Plan::version)
    .def_readonly("workspace_columns",&Plan::workspace_columns)
    .def_readonly("wide_backend",&Plan::wide_backend);
}
